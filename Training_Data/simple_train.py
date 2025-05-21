import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess_dataset import fix_dataset, load_telemetry_data
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from custom_early_stop import CustomEarlyStopping
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

split_by_circuit = True

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

X_full = df.drop(columns=["result"])
y_full = to_categorical(df["result"], num_classes=4)

feature_cols = [col for col in X_full.columns if col not in ["track", "driver", "temp"]]

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_full[feature_cols])
joblib.dump(scaler, "./models/0_simple_scaler.pkl")

X_full_scaled_df = pd.DataFrame(X_scaled_full, columns=feature_cols, index=df.index)

X_full_scaled_df["track"] = df["track"]
X_full_scaled_df["driver"] = df["driver"]
X_full_scaled_df["temp"] = df["temp"]
X_full_scaled_df["result"] = df["result"]

if split_by_circuit:
    df_train_val = X_full_scaled_df[X_full_scaled_df["track"] != 2]
    df_test = X_full_scaled_df[X_full_scaled_df["track"] == 2]

    X_train_val = df_train_val.drop(columns=["result", "track", "driver", "temp"])
    y_train_val = to_categorical(df_train_val["result"], num_classes=4)

    X_test = df_test.drop(columns=["result", "track", "driver", "temp"])
    y_test = to_categorical(df_test["result"], num_classes=4)

    train_size = int(0.8 * len(X_train_val))
    X_train, y_train = X_train_val[:train_size], y_train_val[:train_size]
    X_val, y_val = X_train_val[train_size:], y_train_val[train_size:]
else:
    train_size = int(0.6 * len(X_full_scaled_df))
    val_size = int(0.20 * len(X_full_scaled_df))

    df_model = X_full_scaled_df.drop(columns=["track", "driver", "temp"])

    X = df_model.drop(columns=["result"])
    y = to_categorical(df_model["result"], num_classes=4)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1)
)
class_weights_dict = dict(enumerate(class_weights))

custom_early_stopping = CustomEarlyStopping(validation_data=(X_val, y_val), patience=7)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=1024,
    callbacks=[custom_early_stopping],
    class_weight=class_weights_dict,
    verbose=0
)

model.save("./models/0_simple_cnn_model.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Overall Test accuracy: {test_acc:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=result_classes, output_dict=True, zero_division=0)
for label, metrics in report.items():
    if isinstance(metrics, dict):  # Skip overall metrics like 'accuracy'
        print(f"precision for label '{label}': {metrics['precision']:.2f}")

conf_matrix = confusion_matrix(y_true_classes, y_pred_classes, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=result_classes, yticklabels=result_classes)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

history = model.history
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
