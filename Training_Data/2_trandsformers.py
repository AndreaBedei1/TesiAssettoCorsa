import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess_dataset import fix_dataset, load_telemetry_data
from custom_early_stop import CustomEarlyStopping
from tensorflow.keras import layers, Input, Model

# === Config ===
split_by_circuit = True
window_size = 3
df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)
result_encoder = LabelEncoder()
df["result"] = result_encoder.fit_transform(df["result"])
result_classes = result_encoder.classes_
df["track"] = LabelEncoder().fit_transform(df["track"])
df["driver"] = LabelEncoder().fit_transform(df["driver"])
df["temp"] = LabelEncoder().fit_transform(df["temp"])
df["time_idx"] = df.groupby(["track", "driver", "temp"]).cumcount()

feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "result"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "./models/2_transformers_scaler.pkl")

# === Sequence creation ===
group_cols = ["track", "driver", "temp"]
sequences, labels, groups_info = [], [], []

for name, group in df.groupby(group_cols):
    if len(group) < window_size:
        continue
    group = group.reset_index(drop=True)
    for i in range(len(group) - window_size):
        window = group.iloc[i:i + window_size]
        sequences.append(window[feature_cols].to_numpy())
        labels.append(group.iloc[i + window_size - 1]["result"])
        groups_info.append(name)

X_seq = np.array(sequences)
y_seq = np.array(labels)

# === Train/test split ===
track_seq = np.array([g[0] for g in groups_info])
if split_by_circuit:
    mask_train_val = track_seq != 2
    mask_test = track_seq == 2

    X_train_val = X_seq[mask_train_val]
    y_train_val = y_seq[mask_train_val]
    X_test = X_seq[mask_test]
    y_test = y_seq[mask_test]

    train_size = int(0.8 * len(X_train_val))
    X_train, y_train = X_train_val[:train_size], y_train_val[:train_size]
    X_val, y_val = X_train_val[train_size:], y_train_val[train_size:]
else:
    train_size = int(0.6 * len(X_seq))
    val_size = int(0.2 * len(X_seq))
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size + val_size], y_seq[train_size:train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]

# === Class weights ===
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# === Transformer model ===
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)

def build_model(input_shape, n_classes):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=64, dropout=0.2)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.2)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs, outputs)

model = build_model(X_train.shape[1:], n_classes=len(result_classes))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

custom_early_stopping = CustomEarlyStopping(validation_data=(X_val, y_val), patience=10)

# === Training ===
callbacks = [custom_early_stopping]
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=1024,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

model.save("./models/2_transformer_model.keras")

# === Evaluation ===
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes, target_names=result_classes, output_dict=True, zero_division=0)
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"Precision for label '{label}': {metrics['precision']:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred_classes, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=result_classes, yticklabels=result_classes)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# === Grafici loss/accuracy ===
plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
