import pandas as pd

# Function to split the input CSV into smaller CSV files with 1000 rows each
def split_csv(input_csv, rows_per_split=1000):
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    num_splits = (total_rows // rows_per_split) + 1
    for i in range(num_splits):
        split_df = df[i*rows_per_split:(i+1)*rows_per_split]
        split_df.to_csv(f'input_csv_split_{i+1:02d}.csv', index=False)
    return num_splits

# Split the input CSV
input_csv = 'subject.csv'
split_csv(input_csv)
