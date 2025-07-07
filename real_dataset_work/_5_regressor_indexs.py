import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

train_files = [
    # "2_output_with_mu_slip_angle_2.csv",
    "2_output_with_mu_slip_angle_2.csv",
    # "2_output_with_mu_slip_angle_4.csv",
]
test_file = "2_output_with_mu_slip_angle_3.csv"
train_dataframes = [pd.read_csv(file) for file in train_files]
train_df = pd.concat(train_dataframes, ignore_index=True)
test_df = pd.read_csv(test_file)

SHUFFLE = True
if SHUFFLE:
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

delete_cols = ["general_risk", "angle_base_risk", "steering_condition"]
train_df = train_df.dropna(subset=delete_cols)
test_df = test_df.dropna(subset=delete_cols)

all_features = train_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
feature_cols = [col for col in train_df.columns if col in all_features and col not in delete_cols]

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_test = scaler.transform(test_df[feature_cols])

def compute_weights(y, threshold_percentile=90):
    threshold = np.percentile(y, threshold_percentile)
    weights = np.where(y > threshold, 10.0, 1.0)  
    return weights, threshold

def train_regressor(target_col, train_df, test_df, X_train, X_test):
    print(f"Training model for {target_col}...")
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Calcolo dei pesi
    weights, _ = compute_weights(y_train, threshold_percentile=90)
    
    # Addestramento del modello con pesi
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, sample_weight=weights)
    
    y_pred = model.predict(X_test)
    
    # Valutazione
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # MAE sui picchi (valori sopra il 90° percentile nel test set)
    test_threshold = np.percentile(y_test, 90)
    mask = y_test > test_threshold
    mae_peaks = mean_absolute_error(y_test[mask], y_pred[mask]) if mask.sum() > 0 else np.nan
    
    print(f"Model for {target_col}:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R2 Score: {r2}")
    print(f"MAE on peaks (>90th percentile): {mae_peaks}")
    print("-" * 15)
    
    return model, y_test, y_pred, test_threshold

# Addestramento per angle_base_risk
model_angle_base_risk, y_test_angle_base_risk, y_pred_angle_base_risk, threshold_angle = train_regressor(
    "angle_base_risk", train_df, test_df, X_train, X_test
)

# Addestramento per general_risk
model_general_risk, y_test_general_risk, y_pred_general_risk, threshold_general = train_regressor(
    "general_risk", train_df, test_df, X_train, X_test
)

def analyze_predictions(y_true, y_pred, target_col, threshold=None):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_true)), y_true, label=f"True {target_col}", color="blue", linestyle="-", linewidth=1.5)
    plt.plot(range(len(y_pred)), y_pred, label=f"Predicted {target_col}", color="orange", linestyle="--", linewidth=1.5)
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle=':', label='Peak threshold (90th percentile)')
    plt.title(f"True vs Predicted {target_col}")
    plt.xlabel("Index")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

analyze_predictions(y_test_angle_base_risk, y_pred_angle_base_risk, "angle_base_risk", threshold_angle)
analyze_predictions(y_test_general_risk, y_pred_general_risk, "general_risk", threshold_general)

plt.figure(figsize=(10, 6))
sns.histplot(train_df['angle_base_risk'], bins=50, label='angle_base_risk')
sns.histplot(train_df['general_risk'], bins=50, label='general_risk', alpha=0.5)
plt.title("Distribution of target variables")
plt.legend()
plt.show()