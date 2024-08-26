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