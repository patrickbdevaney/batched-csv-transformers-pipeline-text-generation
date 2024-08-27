# All Rights Reserved
# 
# Copyright © 2024 Patrick Devaney
# 
# THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED COPYING, TRANSFERRING, OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM, IS STRICTLY PROHIBITED.
# 
# The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them for any purpose other than the purpose for which they were provided to you.
# 
# The software is provided “AS IS”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

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
