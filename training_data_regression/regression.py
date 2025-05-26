import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from preprocess_dataset import fix_dataset, load_telemetry_data

df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)

feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "avg_wheel_slip_front", "avg_wheel_slip_rear"]]
X = df[feature_cols].to_numpy()
y_front = df["avg_wheel_slip_front"].to_numpy()
y_rear = df["avg_wheel_slip_rear"].to_numpy()

X_train, X_temp, y_front_train, y_front_temp, y_rear_train, y_rear_temp = train_test_split(
    X, y_front, y_rear, test_size=0.3, random_state=42
)
X_val, X_test, y_front_val, y_front_test, y_rear_val, y_rear_test = train_test_split(
    X_temp, y_front_temp, y_rear_temp, test_size=0.5, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
os.makedirs("./models", exist_ok=True)
joblib.dump(scaler, "./models/scaler.pkl")

def weighted_mse_xgb(y_true, y_pred):
    residual = y_true - y_pred
    grad = -2 * residual * y_true 
    hess = 2 * y_true
    return grad, hess

model_front = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, objective=weighted_mse_xgb
)
model_front.fit(X_train, y_front_train, eval_set=[(X_val, y_front_val)], verbose=False)

model_rear = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, objective=weighted_mse_xgb)
model_rear.fit(X_train, y_rear_train, eval_set=[(X_val, y_rear_val)], verbose=False)

joblib.dump(model_front, "./models/model_front.pkl")
joblib.dump(model_rear, "./models/model_rear.pkl")

y_front_pred = model_front.predict(X_test)
y_rear_pred = model_rear.predict(X_test)

mse_front = mean_squared_error(y_front_test, y_front_pred)
mse_rear = mean_squared_error(y_rear_test, y_rear_pred)

def weighted_mse(y_true, y_pred):
    weights = y_true
    return np.mean(weights * (y_true - y_pred) ** 2)

weighted_mse_front = weighted_mse(y_front_test, y_front_pred)
weighted_mse_rear = weighted_mse(y_rear_test, y_rear_pred)

print(f"Mean Squared Error (Front): {mse_front}")
print(f"Mean Squared Error (Rear): {mse_rear}")
print(f"Weighted Mean Squared Error (Front): {weighted_mse_front}")
print(f"Weighted Mean Squared Error (Rear): {weighted_mse_rear}")

rmse_front = np.sqrt(mse_front)
rmse_rear = np.sqrt(mse_rear)
weighted_rmse_front = np.sqrt(weighted_mse_front)
weighted_rmse_rear = np.sqrt(weighted_mse_rear)

print(f"Root Mean Squared Error (Front): {rmse_front}")
print(f"Root Mean Squared Error (Rear): {rmse_rear}")
print(f"Weighted Root Mean Squared Error (Front): {weighted_rmse_front}")
print(f"Weighted Root Mean Squared Error (Rear): {weighted_rmse_rear}")

def plot_xgboost_training(history, title):
    results = history.evals_result()
    epochs = range(len(results["validation_0"]["rmse"]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, results["validation_0"]["rmse"], label="Validation RMSE")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_xgboost_training(model_front, "Training and Validation RMSE (Front)")
plot_xgboost_training(model_rear, "Training and Validation RMSE (Rear)")