import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, BatchNormalization
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


def create_model(input_dim):
    inputs = Input(shape=(input_dim,))
    
    x = Dense(128, activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    
    x = Dense(32, activation="relu")(x)
    x = BatchNormalization()(x)
    
    output_front = Dense(1, name="output_front")(x)
    output_rear = Dense(1, name="output_rear")(x)
    
    model = Model(inputs=inputs, outputs=[output_front, output_rear])
    model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={"output_front": "mse", "output_rear": "mse"},
    metrics={"output_front": ["mae"], "output_rear": ["mae"]}
    )
    
    return model

input_dim = X_train.shape[1]
model = create_model(input_dim)

history = model.fit(
    X_train,
    {"output_front": y_front_train, "output_rear": y_rear_train},
    validation_data=(X_val, {"output_front": y_front_val, "output_rear": y_rear_val}),
    epochs=50,
    batch_size=2048,
    verbose=0
)

model.save("./models/neural_network_model.h5")
y_front_pred, y_rear_pred = model.predict(X_test)
mse_front = mean_squared_error(y_front_test, y_front_pred)
mse_rear = mean_squared_error(y_rear_test, y_rear_pred)

def weighted_mse(y_true, y_pred):
    weights = y_true
    return np.mean(weights * (y_true - y_pred) ** 2)

weighted_mse_front = weighted_mse(y_front_test, y_front_pred)
weighted_mse_rear = weighted_mse(y_rear_test, y_rear_pred)

print(f"Weighted MSE (Front): {weighted_mse_front}")
print(f"Weighted MSE (Rear): {weighted_mse_rear}")
print(f"Mean Squared Error (Front): {mse_front}")
print(f"Mean Squared Error (Rear): {mse_rear}")

rmse_front = np.sqrt(mse_front)
rmse_rear = np.sqrt(mse_rear)
weighted_rmse_front = np.sqrt(weighted_mse_front)
weighted_rmse_rear = np.sqrt(weighted_mse_rear)

print(f"Root Mean Squared Error (Front): {rmse_front}")
print(f"Root Mean Squared Error (Rear): {rmse_rear}")
print(f"Weighted Root Mean Squared Error (Front): {weighted_rmse_front}")
print(f"Weighted Root Mean Squared Error (Rear): {weighted_rmse_rear}")

history.history["rmse"] = np.sqrt(history.history["loss"])
history.history["val_rmse"] = np.sqrt(history.history["val_loss"])

plt.figure(figsize=(12, 6))
plt.plot(history.history["rmse"], label="Training RMSE")
plt.plot(history.history["val_rmse"], label="Validation RMSE")
plt.title("Training and Validation RMSE")
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history["output_front_mae"], label="Training MAE (Front)")
plt.plot(history.history["val_output_front_mae"], label="Validation MAE (Front)")
plt.title("Training and Validation MAE (Front)")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history["output_rear_mae"], label="Training MAE (Rear)")
plt.plot(history.history["val_output_rear_mae"], label="Validation MAE (Rear)")
plt.title("Training and Validation MAE (Rear)")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

