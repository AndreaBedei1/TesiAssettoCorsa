import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


from preprocess_dataset import fix_dataset, load_telemetry_data
from custom_early_stop import CustomEarlyStoppingTorch
from r1 import ResNet1DTabular

split_by_circuit = True
batch_size = 1024
num_epochs = 100
patience = 7

df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)

result_encoder = LabelEncoder()
df["result"] = result_encoder.fit_transform(df["result"])
result_classes = result_encoder.classes_

os.makedirs("./models", exist_ok=True)
np.save("./models/1_resnet_classes.npy", result_classes)

df["track"] = LabelEncoder().fit_transform(df["track"])
df["driver"] = LabelEncoder().fit_transform(df["driver"])
df["temp"] = LabelEncoder().fit_transform(df["temp"])
df["time_idx"] = df.groupby(["track", "driver", "temp"]).cumcount()


feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "result"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "./models/3_resnet_scaler.pkl")

if split_by_circuit:
    df_train_val = df[df["track"] != 2].copy()
    df_test = df[df["track"] == 2].copy()

    X_train_val = df_train_val[feature_cols].to_numpy()
    y_train_val = df_train_val["result"].to_numpy()
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test["result"].to_numpy()

    split_idx = int(0.8 * len(X_train_val))
    X_train, y_train = X_train_val[:split_idx], y_train_val[:split_idx]
    X_val, y_val = X_train_val[split_idx:], y_train_val[split_idx:]
else:
    X = df[feature_cols].to_numpy()
    y = df["result"].to_numpy()
    split1 = int(0.6 * len(X))
    split2 = int(0.8 * len(X))
    X_train, y_train = X[:split1], y[:split1]
    X_val, y_val = X[split1:split2], y[split1:split2]
    X_test, y_test = X[split2:], y[split2:]

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet1DTabular(input_dim=X_train.shape[1], num_classes=len(result_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
early_stopper = CustomEarlyStoppingTorch(patience=patience)

# class_weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        correct_train += (preds.argmax(1) == yb).sum().item()
        total_train += yb.size(0)

    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(correct_train / total_train)

    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item()
            correct_val += (preds.argmax(1) == yb).sum().item()
            total_val += yb.size(0)

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct_val / total_val)

    print(f"Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

    early_stopper(model, val_loader, device)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

torch.save(model, "./models/3_resnet_full_model.pth")
print("Modello salvato come './models/3_resnet_full_model.pth'")

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        preds = model(xb)
        all_preds.extend(preds.argmax(dim=1).cpu().numpy())
        all_labels.extend(yb.numpy())

print("\nTest Classification Report:")
print(classification_report(all_labels, all_preds, target_names=result_classes))

conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=result_classes, yticklabels=result_classes)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
