import csv

#alternate delimiter for *1* <question> *1* and *2* <answer> *2*

def parse_csv(file_path):
    parsed_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                subject, text = row
                question = text.split('*1*')[1].split('*1*')[0].strip()
                answer = text.split('*2*')[1].split('*2*')[0].strip()
                parsed_data.append((subject, question, answer))
            except (IndexError, ValueError):
                # Skip the row if it can't be parsed using the delimiting technique
                continue
    return parsed_data

def write_parsed_csv(parsed_data, output_file_path):
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject', 'Question', 'Answer'])
        for subject, question, answer in parsed_data:
            writer.writerow([subject, question, answer])

# Example usage
input_file_path = 'merged_output.csv'
output_file_path = 'csv_parsed.csv'
parsed_data = parse_csv(input_file_path)
write_parsed_csv(parsed_data, output_file_path)

print(f"Parsed data has been written to {output_file_path}")