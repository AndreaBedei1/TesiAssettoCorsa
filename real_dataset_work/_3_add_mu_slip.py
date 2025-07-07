import pandas as pd
import numpy as np

# Parameters
wheel_radius = 0.31
epsilon = 1e-5
lf = 1.765  # Distance from CoG to front axle
lr = 1.349  # Distance from CoG to rear axle
ax_threshold = 0.5  # Threshold for longitudinal acceleration (m/s²)
CIRCUIT = ""  # Circuit name for threshold adjustments

# Load CSV
df = pd.read_csv("1_reduced_file_3.csv")

# Drop unnecessary timestamp columns
columns_to_drop = [
    "/controller/control.timestamp.nanoseconds",
    "/eav24_badenia/ride_front.timestamp.nanoseconds",
    "/eav24_badenia/ride_rear.timestamp.nanoseconds",
    "/eav24_badenia/tpms_front.timestamp.nanoseconds",
    "/eav24_badenia/tpms_rear.timestamp.nanoseconds",
    "/eav24_badenia/wheel_load.timestamp.nanoseconds",
    "/observer/ego_loc.timestamp.nanoseconds",
    "/observer/ego_state.timestamp.nanoseconds",
    "/observer/perception_ego_state.timestamp_nanoseconds",
    "/rmpc/debug.timestamp.nanoseconds",
]
df = df.drop(columns=columns_to_drop, errors='ignore')

# Extract data
vx = df["/observer/ego_state.velocity.x"]
vy = df["/observer/ego_state.velocity.y"]

# Remove near-zero vx for stability
vx[vx.abs() < 1] = np.nan

# Wheel speeds
w_fl = df["/observer/ego_state.wheels_speed.fl"]
w_fr = df["/observer/ego_state.wheels_speed.fr"]
w_rl = df["/observer/ego_state.wheels_speed.rl"]
w_rr = df["/observer/ego_state.wheels_speed.rr"]

# Vertical load
Fz_front = df["/rmpc/debug.predicted_fzf"]
Fz_rear = df["/rmpc/debug.predicted_fzr"]
# Lateral forces
Fy_front = df["/rmpc/debug.predicted_fyf"]
Fy_rear = df["/rmpc/debug.predicted_fyr"]
# Longitudinal forces
Fx_front = df["/rmpc/debug.predicted_fxf"]
Fx_rear = df["/rmpc/debug.predicted_fxr"]

# μ estimate (friction coefficient)
df["mu_front"] = np.sqrt(Fx_front**2 + Fy_front**2) / (Fz_front + epsilon)
df["mu_rear"] = np.sqrt(Fx_rear**2 + Fy_rear**2) / (Fz_rear + epsilon)

# Slip ratios
slip_fl = (w_fl * wheel_radius - vx) / (vx.abs() + epsilon)
slip_fr = (w_fr * wheel_radius - vx) / (vx.abs() + epsilon)
slip_rl = (w_rl * wheel_radius - vx) / (vx.abs() + epsilon)
slip_rr = (w_rr * wheel_radius - vx) / (vx.abs() + epsilon)

# Limita i valori di slip_ratio_front, slip_ratio_rear e slip_ratio_mean
df["slip_ratio_front"] = np.clip((slip_fl + slip_fr) / 2, -1.25, 1.25)
df["slip_ratio_rear"] = np.clip((slip_rl + slip_rr) / 2, -1.25, 1.25)
df["slip_ratio_mean"] = (df["slip_ratio_front"] + df["slip_ratio_rear"]) / 2


# Slip angle (body)
df["slip_angle_body"] = np.arctan2(vy, vx)  # In radians

# Slip angles front/rear
# Limita i valori di slip_angle_front e slip_angle_rear
df["slip_angle_front"] = np.clip(df["/rmpc/debug.predicted_alpha_f"], -0.30, 0.30)
df["slip_angle_rear"] = np.clip(df["/rmpc/debug.predicted_alpha_r"], -0.30, 0.30)

if CIRCUIT == "ABU_DHABI":
    threshold_under=0.1 
    threshold_over=0.07
else:
    threshold_under=0.1
    threshold_over=0.045


# Simple Classification based on threshold
def classify_steering(slip_angle_front, slip_angle_rear):
    difference = slip_angle_front - slip_angle_rear
    if difference > threshold_over:
        return "oversteer"
    elif difference < -threshold_under:
        return "understeer"
    return "neutral"

df["steering_condition"] = df.apply(
    lambda row: classify_steering(row["slip_angle_front"], row["slip_angle_rear"]),
    axis=1
)

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculate dynamic weights based on longitudinal velocity
def calculate_weights(vx):
    normalized_vx = np.clip(vx / 71.0, 0, 1)  # Normalize velocity (max 71 m/s)
    weight_angles = 0.7 + 0.2 * normalized_vx  # Higher weight for angles at high speed
    weight_slip = 0.2 - 0.1 * normalized_vx   # Lower weight for slip at high speed
    weight_mu = 0.1                           # Constant weight for mu
    return weight_angles, weight_slip, weight_mu

df["angle_base_risk"] = sigmoid(
    np.abs(df["slip_angle_front"] - df["slip_angle_rear"]) / 0.02
)

# Calculate risk for braking and acceleration
def calculate_risk(row):
    # Extract relevant data
    ax = row.get("/rmpc/debug.predicted_ax", 0)  # Longitudinal acceleration (m/s²)
    slip_ratio_front = row["slip_ratio_front"]
    slip_ratio_rear = row["slip_ratio_rear"]
    slip_angle_front = row["slip_angle_front"]
    slip_angle_rear = row["slip_angle_rear"]
    mu_front = row["mu_front"]
    mu_rear = row["mu_rear"]
    Fz_front = row["/rmpc/debug.predicted_fzf"]
    Fz_rear = row["/rmpc/debug.predicted_fzr"]
    vx = row["/observer/ego_state.velocity.x"]

    # Dynamic weights
    weight_angles, weight_slip, weight_mu = calculate_weights(vx)

    # Thresholds for critical slip ratio and slip angle
    slip_ratio_threshold = 0.1  # Typical value for tire grip limit
    slip_angle_threshold = 0.1  # ~5.7 degrees in radians

    # Risk during braking (ax < -ax_threshold)
    if ax < -ax_threshold:
        # Risk of front wheel lockup
        slip_risk_front = np.abs(slip_ratio_front) / (slip_ratio_threshold * mu_front + epsilon)
        # Normalize by mu and vertical load
        force_risk_front = np.abs(row["/rmpc/debug.predicted_fxf"]) / (mu_front * Fz_front + epsilon)
        # Slip angle risk
        angle_risk = np.abs(slip_angle_front) / slip_angle_threshold
        # Combine risks
        braking_risk = (
            weight_slip * slip_risk_front +
            weight_mu * force_risk_front +
            weight_angles * angle_risk
        )
        return sigmoid(braking_risk)

    # Risk during acceleration (ax > ax_threshold)
    elif ax > ax_threshold:
        # Risk of rear wheel spin
        slip_risk_rear = np.abs(slip_ratio_rear) / (slip_ratio_threshold * mu_rear + epsilon)
        # Normalize by mu and vertical load
        force_risk_rear = np.abs(row["/rmpc/debug.predicted_fxr"]) / (mu_rear * Fz_rear + epsilon)
        # Slip angle risk
        angle_risk = np.abs(slip_angle_rear) / slip_angle_threshold
        # Combine risks
        accel_risk = (
            weight_slip * slip_risk_rear +
            weight_mu * force_risk_rear +
            weight_angles * angle_risk
        )
        return sigmoid(accel_risk)

    # Neutral case (-ax_threshold <= ax <= ax_threshold)
    else:
        # Use average slip and angle risks
        slip_risk = (np.abs(slip_ratio_front) + np.abs(slip_ratio_rear)) / (2 * slip_ratio_threshold)
        angle_risk = (np.abs(slip_angle_front) + np.abs(slip_angle_rear)) / (2 * slip_angle_threshold)
        force_risk = (
            np.abs(row["/rmpc/debug.predicted_fxf"]) / (mu_front * Fz_front + epsilon) +
            np.abs(row["/rmpc/debug.predicted_fxr"]) / (mu_rear * Fz_rear + epsilon)
        ) / 2
        neutral_risk = (
            weight_slip * slip_risk +
            weight_mu * force_risk +
            weight_angles * angle_risk
        )
        return sigmoid(neutral_risk)

# Apply risk calculation
df["general_risk"] = df.apply(calculate_risk, axis=1)

def expand_to_full_range(series):
    return (series - 0.5) * 2
df["angle_base_risk"] = expand_to_full_range(df["angle_base_risk"])
df["general_risk"] = expand_to_full_range(df["general_risk"])

# Replace infs and NaNs for safety
df = df.replace([np.inf, -np.inf], np.nan)

# Save
df.to_csv("2_output_with_mu_slip_angle_3.csv", index=False)
print("Saved as '2_output_with_mu_slip_angle_4.csv'")