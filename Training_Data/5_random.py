import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from preprocess_dataset import fix_dataset, load_telemetry_data
from custom_early_stop import CustomEarlyStopping
from keras.layers import BatchNormalization

import seaborn as sns
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf

num_threads = tf.config.threading.get_intra_op_parallelism_threads()
print(f"Numero di thread disponibili: {num_threads}")

tf.config.threading.set_intra_op_parallelism_threads(0)  
tf.config.threading.set_inter_op_parallelism_threads(0)

import os
os.environ["OMP_NUM_THREADS"] = "0"
os.environ["OPENBLAS_NUM_THREADS"] = "0"
os.environ["MKL_NUM_THREADS"] = "0"
os.environ["VECLIB_MAXIMUM_THREADS"] = "0"
os.environ["NUMEXPR_NUM_THREADS"] = "0"

split_by_circuit = True
random_state = 42

# === Caricamento e preprocessing ===
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

# Encoding altre colonne categoriali
df["track"] = LabelEncoder().fit_transform(df["track"])
df["driver"] = LabelEncoder().fit_transform(df["driver"])
df["temp"] = LabelEncoder().fit_transform(df["temp"])

# Selezione feature numeriche + scaling
feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "result"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "./models/0_simple_scaler_random.pkl")

# === Split casuale dei dati ===
X = df[feature_cols]
y = to_categorical(df["result"], num_classes=4)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=df["result"])
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=np.argmax(y_temp, axis=1))

# === Data Augmentation ===
class_counts = np.sum(y_train, axis=0)
max_count = np.max(class_counts)

augmented_X, augmented_y = [], []
augmentation_limit = 10000

for i, count in enumerate(class_counts):
    if count < max_count:
        class_indices = np.where(np.argmax(y_train, axis=1) == i)[0]
        class_X = X_train.iloc[class_indices]
        class_y = y_train[class_indices]

        num_to_add = max_count - count
        num_added = 0

        while num_added < num_to_add and len(augmented_X) < augmentation_limit:
            augmented_X.append(class_X)
            augmented_y.append(class_y)
            num_added += len(class_X)
            class_X = class_X.sample(frac=1, replace=True, random_state=random_state)  # Shuffle

if augmented_X:
    augmented_X = pd.concat(augmented_X, axis=0)
    augmented_y = np.vstack(augmented_y)

    X_train = pd.concat([X_train, augmented_X], axis=0)
    y_train = np.vstack([y_train, augmented_y])

# Conta il numero di etichette per classe dopo la data augmentation
class_distribution = np.bincount(np.argmax(y_train, axis=1))
print("Distribuzione delle classi dopo la data augmentation:")
for i, count in enumerate(class_distribution):
    print(f"Classe {i} ({result_classes[i]}): {count} esempi")

# Shuffle finale
X_train, y_train = shuffle(X_train, y_train, random_state=random_state)

# === Modello ===
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
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
    epochs=100,
    batch_size=64,
    callbacks=[custom_early_stopping],
    class_weight=class_weights_dict,
    verbose=0
)

model.save("./models/0_simple_cnn_model_random_augmented.keras")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Overall Test accuracy: {test_acc:.2f}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

report = classification_report(y_true_classes, y_pred_classes, target_names=result_classes, output_dict=True, zero_division=0)
for label, metrics in report.items():
    if isinstance(metrics, dict):
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