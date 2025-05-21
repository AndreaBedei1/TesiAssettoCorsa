from sklearn.preprocessing import StandardScaler, LabelEncoder
from preprocess_dataset import fix_dataset, load_telemetry_data
import seaborn as sns
import matplotlib.pyplot as plt
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

scaler = StandardScaler()
feature_cols = [col for col in df.columns if col not in ["temp", "track", "driver"]]
X_scaled = scaler.fit_transform(df[feature_cols])

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