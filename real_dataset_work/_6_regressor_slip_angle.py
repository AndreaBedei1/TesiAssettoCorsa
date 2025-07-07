import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib

RETRAIN = True 
FRONT_TREE_MODEL_PATH = "./models/rf_front_model.pkl"
REAR_TREE_MODEL_PATH = "./models/rf_rear_model.pkl"
FRONT_LSTM_MODEL_PATH = "./models/lstm_model_front.keras"
REAR_LSTM_MODEL_PATH = "./models/lstm_model_rear.keras"

TRAIN_FILES = [
    "2_output_with_mu_slip_angle_1.csv",
    "2_output_with_mu_slip_angle_3.csv",
    "2_output_with_mu_slip_angle_4.csv"
]
TEST_FILE = "2_output_with_mu_slip_angle_2.csv"
CIRCUIT_TEST = "" # "ABU_DHABI" or "" for default
SHUFFLE = True  

if CIRCUIT_TEST == "ABU_DHABI":
    threshold_under=0.1 
    threshold_over=0.07
    balance_threshold_under = 0.05
    balance_threshold_over = 0.050
else:
    threshold_under=0.1
    threshold_over=0.045
    balance_threshold_under = 0.06
    balance_threshold_over = 0.037

# Add timestamp column to each DataFrame, resetting it to a range from 0 to n-1, this due to tire consumption
def load_and_reset_timestamp(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = range(len(df)) 
    return df

train_dfs = [load_and_reset_timestamp(file) for file in TRAIN_FILES]
df_train = pd.concat(train_dfs, ignore_index=True)

def balance_dataset(df, oversample_factor=700, undersample_frac=0.2):
    df = df.copy()

    high_front = df[df["slip_angle_front"].abs() > 0.07]
    high_rear = df[df["slip_angle_rear"].abs() > 0.04]
    neutral = df[
        (df["slip_angle_front"].abs() <= balance_threshold_over) &
        (df["slip_angle_rear"].abs() <= balance_threshold_under)
    ]

    neutral_sampled = neutral.sample(frac=undersample_frac, random_state=42)

    high_front_oversampled = pd.concat([high_front] * oversample_factor, ignore_index=True)
    high_rear_oversampled = pd.concat([high_rear] * oversample_factor, ignore_index=True)

    balanced_df = pd.concat([neutral_sampled, high_front_oversampled, high_rear_oversampled], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced_df

df_train = balance_dataset(df_train)
df_test = load_and_reset_timestamp(TEST_FILE)

if TEST_FILE == "2_output_with_mu_slip_angle_2.csv":
    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test["slip_angle_front"], label="True slip_angle_front (original)")
    plt.legend()
    plt.title("Original Data")
    plt.xlabel("Index")
    plt.ylabel("Slip Angle Front")
    plt.grid()
    plt.show()

    df_test = df_test.iloc[:-29000]  # Remove last 29000 rows only for test file 2

    plt.figure(figsize=(12, 6))
    plt.plot(df_test.index, df_test["slip_angle_front"], label="True slip_angle_front (modified)")
    plt.legend()
    plt.title("Modified Data (Last 7000 Rows Removed)")
    plt.xlabel("Index")
    plt.ylabel("Slip Angle Front")
    plt.grid()
    plt.show()


delete_cols = [
    "/rmpc/debug.predicted_alpha_f",
    "/rmpc/debug.predicted_alpha_r",
    "slip_angle_body",
    "general_risk",
    "angle_base_risk",
    "slip_angle_rear",
    "slip_angle_front",
    "steering_condition"
]

df_train = df_train.reset_index(drop=True)
df_train = df_train.dropna(subset=delete_cols)
df_test = df_test.reset_index(drop=True)
df_test = df_test.dropna(subset=delete_cols)

if SHUFFLE:
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

X_train = df_train.drop(columns=["slip_angle_front", "slip_angle_rear"] + delete_cols)
y_front_train = df_train["slip_angle_front"]
y_rear_train = df_train["slip_angle_rear"]

X_test = df_test.drop(columns=["slip_angle_front", "slip_angle_rear"] + delete_cols)
y_front_test = df_test["slip_angle_front"]
y_rear_test = df_test["slip_angle_rear"]

### XGB Regressor ###
xgb_front = XGBRegressor(random_state=42, n_jobs=-1, verbosity=1)
if not os.path.exists(FRONT_TREE_MODEL_PATH) or RETRAIN:
    xgb_front.fit(X_train, y_front_train)
    joblib.dump(xgb_front, FRONT_TREE_MODEL_PATH)
else:
    xgb_front = joblib.load(FRONT_TREE_MODEL_PATH)
y_front_pred_rf = xgb_front.predict(X_test)

xgb_rear = XGBRegressor(random_state=42, n_jobs=-1, verbosity=1)
if not os.path.exists(REAR_TREE_MODEL_PATH) or RETRAIN:
    xgb_rear.fit(X_train, y_rear_train)
    joblib.dump(xgb_rear, REAR_TREE_MODEL_PATH)
else:
    xgb_rear = joblib.load(REAR_TREE_MODEL_PATH)
y_rear_pred_rf = xgb_rear.predict(X_test)

print("XGB - slip_angle_front:")
print("MSE:", mean_squared_error(y_front_test, y_front_pred_rf))
print("R2:", r2_score(y_front_test, y_front_pred_rf))

print("XGB - slip_angle_rear:")
print("MSE:", mean_squared_error(y_rear_test, y_rear_pred_rf))
print("R2:", r2_score(y_rear_test, y_rear_pred_rf))

def analyze_predictions_tree(model, target_col, feature_cols, df):
    X = df[feature_cols]
    y_true = df[target_col]
    y_pred = model.predict(X)

    mse = mean_squared_error(y_true, y_pred)
    print(f"Analysis for {target_col}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {np.sqrt(mse)}")
    print(f"R2 Score: {r2_score(y_true, y_pred)}")
    print("-" * 15)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, y_true, label=f"True {target_col}", color="blue", linestyle="-", linewidth=1.5)
    plt.plot(df.index, y_pred, label=f"Predicted {target_col}", color="orange", linestyle="--", linewidth=1.5)
    plt.title(f"True vs Predicted {target_col}")
    plt.xlabel("Index")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

analyze_predictions_tree(xgb_front, "slip_angle_front", X_test.columns.tolist(), df_test)
analyze_predictions_tree(xgb_rear, "slip_angle_rear", X_test.columns.tolist(), df_test)

def classify_steering(slip_angle_front, slip_angle_rear):
    difference = slip_angle_front - slip_angle_rear
    if difference > threshold_over:
        return "oversteer"
    elif difference < -threshold_under:
        return "understeer"
    return "neutral"
y_front_pred = xgb_front.predict(X_test)
y_rear_pred = xgb_rear.predict(X_test)
df_test["steering_condition_regressor"] = [
    classify_steering(front, rear)
    for front, rear in zip(y_front_pred, y_rear_pred)
]

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

neutral_group = df_test[df_test["steering_condition_regressor"] == "neutral"]
plt.plot(
    neutral_group.index,
    neutral_group["/observer/ego_state.velocity.x"],
    label="Neutral",
    color=colors["neutral"],
    linestyle='-',
    linewidth=0.8
)

for condition, group in df_test[df_test["steering_condition_regressor"] != "neutral"].groupby("steering_condition_regressor"):
    plt.scatter(
        group.index,
        group["/observer/ego_state.velocity.x"],
        label=condition.capitalize(),
        color=colors.get(condition, "gray"),
        marker=markers.get(condition, "o"),
        s=100
    )

plt.title("Velocity X with Steering Conditions Highlighted (Predicted)")
plt.xlabel("Time (samples)")
plt.ylabel("Velocity X [m/s]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()