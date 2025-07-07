import pandas as pd
from glob import glob
import os

folder_path = "./csv_0"

all_files = glob(os.path.join(folder_path, "*.csv"))
dfs = []

for file in all_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

combined = combined.sort_values("timestamp")

pivoted = combined.pivot(index="timestamp", columns="topic", values="value")

pivoted = pivoted.sort_index()

pivoted = pivoted.interpolate(method="nearest", limit_direction="both")

final_df = pivoted.reset_index()

final_df.to_csv("0_output_unify_4.csv", index=False)
