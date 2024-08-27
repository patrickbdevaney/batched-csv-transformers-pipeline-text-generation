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

import pandas as pd
import re

# Function to parse descriptions from a given text
def parse_descriptions(text):
    # Find all descriptions enclosed in brackets
    descriptions = re.findall(r'\[([^\[\]]+)\]', text)
    # Filter descriptions with at least 3 words
    descriptions = [desc.strip() for desc in descriptions if len(desc.split()) >= 3]
    return descriptions

# Read the input CSV file
input_csv = 'descriptions.csv'
df = pd.read_csv(input_csv)

# Initialize a list to store parsed descriptions
parsed_descriptions = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    descriptions = parse_descriptions(row['description'])
    parsed_descriptions.extend(descriptions)

# Create a new DataFrame with the parsed descriptions
parsed_df = pd.DataFrame(parsed_descriptions, columns=['description'])

# Write the parsed descriptions to a new CSV file
output_csv = 'descriptions_parsed.csv'
parsed_df.to_csv(output_csv, index=False)

print(f"Parsed descriptions written to {output_csv}")