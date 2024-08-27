import os
import pandas as pd

# Function to merge CSV files with a "description" column into a single CSV file
def merge_csv_files(directory):
    merged_data = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Check if the "description" column exists
                if "description" in df.columns:
                    # Append the "description" column to the merged data
                    merged_data.extend(df["description"].dropna().tolist())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Create a DataFrame with the merged data
    merged_df = pd.DataFrame(merged_data, columns=["description"])

    # Save the merged DataFrame to a new CSV file
    output_file = os.path.join(directory, "descriptions.csv")
    merged_df.to_csv(output_file, index=False)
    print(f"Merged CSV file saved to {output_file}")

# Directory containing the CSV files
directory = '.'  # Replace with the actual directory if needed

# Merge the CSV files
merge_csv_files(directory)
