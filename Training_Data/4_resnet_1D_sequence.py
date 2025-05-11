import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess_dataset import fix_dataset, load_telemetry_data
from custom_early_stop import CustomEarlyStoppingTorch

# === Config ===
split_by_circuit = True
window_size = 5
batch_size = 64
num_epochs = 100
patience = 8

# === Load and preprocess ===
df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)

# Encoding
result_encoder = LabelEncoder()
df["result"] = result_encoder.fit_transform(df["result"])
result_classes = result_encoder.classes_
df["track"] = LabelEncoder().fit_transform(df["track"])
df["driver"] = LabelEncoder().fit_transform(df["driver"])
df["temp"] = LabelEncoder().fit_transform(df["temp"])
df["time_idx"] = df.groupby(["track", "driver", "temp"]).cumcount()

# Feature scaling
feature_cols = [col for col in df.columns if col not in ["track", "driver", "temp", "result"]]
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, "./models/resnet1d_sequence_scaler.pkl")

# === Create sequences with windowing ===
group_cols = ["track", "driver", "temp"]
sequences, labels, track_ids = [], [], []

for name, group in df.groupby(group_cols):
    if len(group) < window_size:
        continue
        
    group = group.sort_values("time_idx").reset_index(drop=True)
    group_features = group[feature_cols].values
    group_results = group["result"].values
    
    for i in range(len(group) - window_size):
        sequences.append(group_features[i:i+window_size])
        labels.append(group_results[i + window_size - 1])
        track_ids.append(name[0])  # track is first in group_cols

X_seq = np.array(sequences)
y_seq = np.array(labels)
track_ids = np.array(track_ids)

# === Split dataset ===
if split_by_circuit:
    mask_train_val = track_ids != 2
    mask_test = track_ids == 2

    X_train_val = X_seq[mask_train_val]
    y_train_val = y_seq[mask_train_val]
    X_test = X_seq[mask_test]
    y_test = y_seq[mask_test]

    split_idx = int(0.8 * len(X_train_val))
    X_train, y_train = X_train_val[:split_idx], y_train_val[:split_idx]
    X_val, y_val = X_train_val[split_idx:], y_train_val[split_idx:]
else:
    split1 = int(0.6 * len(X_seq))
    split2 = int(0.8 * len(X_seq))
    X_train, y_train = X_seq[:split1], y_seq[:split1]
    X_val, y_val = X_seq[split1:split2], y_seq[split1:split2]
    X_test, y_test = X_seq[split2:], y_seq[split2:]

# === Class weights ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# === ResNet1D Model ===
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class ResNet1D(nn.Module):
    def __init__(self, input_features, seq_length, num_classes):
        super().__init__()
        self.in_channels = input_features
        self.seq_len = seq_length
        
        self.conv1 = nn.Conv1d(self.in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = [ResidualBlock1D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, seq_len)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)

# === Training Setup ===

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
input_features = X_train.shape[2]  # Numero di features
seq_length = X_train.shape[1]      # Lunghezza della sequenza (window_size)
model = ResNet1D(input_features=input_features, 
                 seq_length=seq_length, 
                 num_classes=len(result_classes)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
early_stopper = CustomEarlyStoppingTorch(patience=patience)

# === Training Loop ===
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.detach().item() 
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct_val / total_val
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    early_stopper(model, val_loader, device)
    if early_stopper.early_stop:
        print("Early stopping triggered.")
        break

# === Evaluation ===
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=result_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds, normalize='true')
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=result_classes, yticklabels=result_classes)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss over epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy over epochs')
plt.legend()
plt.tight_layout()
plt.show()

# Save model
torch.save(model.state_dict(), "./models/resnet1d_sequence_model.pth")