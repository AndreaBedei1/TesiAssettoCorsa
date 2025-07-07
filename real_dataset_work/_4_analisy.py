import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("2_output_with_mu_slip_angle_3.csv")
df = df[df["/observer/ego_state.velocity.x"] > 1]


if "timestamp" in df.columns:
    time = (df["timestamp"] - df["timestamp"].min())
else:
    time = df.index

# === Plot 1: Velocity X with Classification ===
plt.figure(figsize=(12, 6))
colors = {
    "neutral": "blue",
    "understeer": "green",
    "oversteer": "red"
}
markers = {
    "neutral": "-",
    "understeer": "^",
    "oversteer": "v",
}
neutral_group = df[df["steering_condition"] == "neutral"]
plt.plot(
    neutral_group.index,
    neutral_group["/observer/ego_state.velocity.x"],
    label="Neutral",
    color=colors["neutral"],
    linestyle='-',
    linewidth=0.8
)
for condition, group in df[df["steering_condition"] != "neutral"].groupby("steering_condition"):
    plt.scatter(
        group.index,
        group["/observer/ego_state.velocity.x"],
        label=condition.capitalize(),
        color=colors.get(condition, "gray"),
        marker=markers.get(condition, "o"),
        s=100
    )

plt.title("Velocity X with Steering Conditions Highlighted")
plt.xlabel("Time (samples)")
plt.ylabel("Velocity X [m/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# === Plot 2: General Risk Indicator (Slip Angle Only) ===
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["angle_base_risk"],
    label="General Risk (Slip Angle Only)",
    color="purple",
    linestyle='-',
    linewidth=1.5
)
plt.title("General Risk Indicator Over Time (Slip Angle Only)")
plt.xlabel("Time (samples)")
plt.ylabel("Risk Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 3: General Risk Indicator with Dynamic Weights ===
plt.figure(figsize=(12, 6))
plt.plot(
    df.index,
    df["general_risk"],
    label="General Risk",
    color="purple",
    linestyle='-',
    linewidth=1.5
)
plt.title("General Risk Indicator Over Time with Dynamic Weights")
plt.xlabel("Time (samples)")
plt.ylabel("Risk Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 4: All Metrics in One Plot ===
plt.figure(figsize=(14, 7))
plt.plot(time, df["slip_ratio_front"], label="Slip Ratio Front", color="tab:green")
plt.plot(time, df["slip_ratio_rear"], label="Slip Ratio Rear", color="tab:red")
plt.plot(time, df["slip_ratio_mean"], label="Slip Ratio Mean", color="tab:purple")
plt.plot(time, df["slip_angle_body"], label="Slip Angle (Body)", color="tab:orange")
plt.plot(time, df["slip_angle_front"], label="Slip Angle (Front Axle)", color="tab:cyan")
plt.plot(time, df["slip_angle_rear"], label="Slip Angle (Rear Axle)", color="tab:brown")
plt.title("All Metrics in One Plot")
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Plot 5: Velocity X vs Mu (Front & Rear) ===
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Velocity X [m/s]", color="tab:blue")
line1 = ax1.plot(
    df.index,
    df["/observer/ego_state.velocity.x"],
    label="Velocity X",
    color="tab:blue"
)
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Mu (Coefficient of Friction)", color="tab:red")
line2 = ax2.plot(
    df.index,
    df["mu_front"],
    label="Mu Front",
    color="tab:red"
)
line3 = ax2.plot(
    df.index,
    df["mu_rear"],
    label="Mu Rear",
    color="tab:green"
)
ax2.tick_params(axis='y', labelcolor="tab:red")

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Velocity X vs Mu (Front & Rear)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 6: Velocity X vs Slip Ratio (Front & Rear) ===
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Velocity X [m/s]", color="tab:blue")
line1 = ax1.plot(
    df.index,
    df["/observer/ego_state.velocity.x"],
    label="Velocity X",
    color="tab:blue"
)
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Slip Ratio", color="tab:red")
line2 = ax2.plot(
    df.index,
    df["slip_ratio_front"],
    label="Slip Ratio Front",
    color="tab:red"
)
line3 = ax2.plot(
    df.index,
    df["slip_ratio_rear"],
    label="Slip Ratio Rear",
    color="tab:green"
)
ax2.tick_params(axis='y', labelcolor="tab:red")

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Velocity X vs Slip Ratio (Front & Rear)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 7: Velocity X vs Slip Angle (Front & Rear) ===
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax1.set_xlabel("Time (samples)")
ax1.set_ylabel("Velocity X [m/s]", color="tab:blue")
line1 = ax1.plot(
    df.index,
    df["/observer/ego_state.velocity.x"],
    label="Velocity X",
    color="tab:blue"
)
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Slip Angle (rad)", color="tab:red")
line2 = ax2.plot(
    df.index,
    df["slip_angle_front"],
    label="Slip Angle Front",
    color="tab:red"
)
line3 = ax2.plot(
    df.index,
    df["slip_angle_rear"],
    label="Slip Angle Rear",
    color="tab:green"
)
ax2.tick_params(axis='y', labelcolor="tab:red")
lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right")

plt.title("Velocity X vs Slip Angle (Front & Rear)")
plt.grid(True)
plt.tight_layout()
plt.show()