import os
import csv

def merge_csv_files(directory, output_file):
    data = []
    headers = None

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('output_csv_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    file_headers = next(reader)
                    if headers is None:
                        headers = file_headers
                    elif headers != file_headers:
                        raise ValueError(f"Headers do not match in file: {file_path}")
                    data.extend(list(reader))

    # Write the merged data to the output CSV file
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)  # Write the headers
        writer.writerows(data)  # Write the data

# Example usage
directory = '.'
output_file = 'merged_output.csv'
merge_csv_files(directory, output_file)

print(f"Merged data has been written to {output_file}")
