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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

split_by_circuit = True

df = load_telemetry_data("../data/dataset/vehicle_telemetry_*.csv")
if df.empty:
    print("Nessun file trovato. Controlla il pattern o la cartella.")
    exit()

df = fix_dataset(df)

result_encoder = LabelEncoder()
df["result"] = result_encoder.fit_transform(df["result"])


# Scala i dati
scaler = StandardScaler()
# Seleziona solo le feature numeriche
feature_cols = [col for col in df.columns if col not in ["temp", "track", "driver"]]

X_scaled = scaler.fit_transform(df[feature_cols])


# T-SNE
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1, verbose=1)
X_tsne = tsne.fit_transform(X_scaled)

df["tsne_1"] = X_tsne[:, 0]
df["tsne_2"] = X_tsne[:, 1]

df["result_name"] = df["result"].apply(lambda x: result_encoder.classes_[x])

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x="tsne_1", y="tsne_2", hue="result_name", palette="tab10", data=df, alpha=0.7
)
plt.title("T-SNE Visualization")
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.legend(title="Result", loc="best")
plt.grid()
plt.show()