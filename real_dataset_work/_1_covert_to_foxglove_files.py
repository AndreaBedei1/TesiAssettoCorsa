import pandas as pd
from glob import glob
import os

input_folder = "./csv_0"
output_folder = "./csv_0_long"

os.makedirs(output_folder, exist_ok=True)

all_files = glob(os.path.join(input_folder, "*.csv"))

for file in all_files:
    df = pd.read_csv(file)
    filename = os.path.basename(file)

    # Find the column that ends with 'timestamp_nanoseconds'
    timestamp_cols = [col for col in df.columns if col.endswith("timestamp_nanoseconds")]
    if not timestamp_cols:
        print(f"No timestamp column found in {filename}")
        continue

    timestamp_col = timestamp_cols[0]

    # Temporarily rename it to 'timestamp'
    df = df.rename(columns={timestamp_col: "timestamp"})

    # Convert to long format: columns 'topic' and 'value'
    df_long = df.melt(id_vars=["timestamp"], var_name="topic", value_name="value")

    # Export converted CSV
    out_file = os.path.join(output_folder, filename.replace(".csv", "_long.csv"))
    df_long.to_csv(out_file, index=False)
    print(f"Saved {out_file} ({len(df_long)} rows)")
