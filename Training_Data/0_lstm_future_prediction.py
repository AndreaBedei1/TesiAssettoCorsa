import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess_dataset import fix_dataset, load_telemetry_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from custom_early_stop import CustomEarlyStopping
from tensorflow.keras import Input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

split_by_circuit = True
window_size = 2
future_offset = 2  

df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)

result_encoder = LabelEncoder()
df["result"] = result_encoder.fit_transform(df["result"])
result_classes = result_encoder.classes_
print("Mappatura delle classi:")
for i, class_name in enumerate(result_classes):
    print(f"{i}: {class_name}")

df["track"] = LabelEncoder().fit_transform(df["track"])
df["driver"] = LabelEncoder().fit_transform(df["driver"])
df["temp"] = LabelEncoder().fit_transform(df["temp"])
df["time_idx"] = df.groupby(["track", "driver", "temp"]).cumcount()

feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "result"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "./models/1_lstm_future_scaler.pkl")

group_cols = ["track", "driver", "temp"]
sequences, labels, groups_info = [], [], []

for name, group in df.groupby(group_cols):
    group = group.reset_index(drop=True)
    max_index = len(group) - (window_size + future_offset - 1)
    if max_index <= 0:
        continue
    for i in range(max_index):
        window = group.iloc[i : i + window_size]
        label_idx = i + window_size - 1 + future_offset
        sequences.append(window[feature_cols].to_numpy())
        labels.append(group.iloc[label_idx]["result"])
        groups_info.append(name)

X_seq = np.array(sequences)
y_seq = to_categorical(np.array(labels), num_classes=4)

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
    X_val, y_val = X_seq[train_size : train_size + val_size], y_seq[train_size : train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size :], y_seq[train_size + val_size :]

y_train_labels = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(enumerate(class_weights))

custom_early_stopping = CustomEarlyStopping(validation_data=(X_val, y_val), patience=7)

# Definizione del modello LSTM
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Addestramento
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=2048,
    callbacks=[custom_early_stopping],
    class_weight=class_weights_dict,
    verbose=1
)

model.save("./models/1_lstm_future_model.keras")

# Valutazione
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=result_classes, output_dict=True, zero_division=0)
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"Accuracy for label '{label}': {metrics['precision']:.2f}")

# Matrice di confusione normalizzata
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=result_classes, yticklabels=result_classes)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Andamento di Loss e Accuracy
plt.figure(figsize=(12, 12))
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()