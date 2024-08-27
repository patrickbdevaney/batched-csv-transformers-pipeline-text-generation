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

import csv

def parse_csv(file_path):
    parsed_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text = row[1]  # Ignore the subject column and only use the dialogue column
            question, answer = None, None
            try:
                # Extract the first question and answer pair
                question = text.split('*2*')[0].split('*1*')[1].strip()
                answer = text.split('*2*')[1].strip()
            except (IndexError, ValueError):
                # Skip the row if it can't be parsed using the delimiting technique
                continue
            if question and answer:
                parsed_data.append((question, answer))
    return parsed_data

def write_parsed_csv(parsed_data, output_file_path):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])
        for question, answer in parsed_data:
            writer.writerow([question, answer])

# Example usage
input_file_path = 'merged_output.csv'
output_file_path = 'csv_parsed.csv'
parsed_data = parse_csv(input_file_path)
write_parsed_csv(parsed_data, output_file_path)

print(f"Parsed data has been written to {output_file_path}")
