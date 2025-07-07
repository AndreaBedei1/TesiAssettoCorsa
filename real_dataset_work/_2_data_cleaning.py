import pandas as pd

input_file = '0_output_unify_4.csv'
output_file = '1_reduced_file_4.csv'
CHUNKS_SIZE = 10

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Group the data into chunks of 10 rows and calculate the mean for each group
reduced_df = df.groupby(df.index // CHUNKS_SIZE).mean()

# Save the reduced DataFrame to a new CSV file
reduced_df.to_csv(output_file, index=False)

print(f"Reduced file saved to {output_file}")