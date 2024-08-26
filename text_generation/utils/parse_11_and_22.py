import csv

def parse_csv(file_path):
    parsed_data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text = row[1]  # Ignore the subject column and only use the dialogue column
            question, answer = None, None
            try:
                # First method
                question = text.split('*2*')[0].split('*1*')[1].strip()
                answer = text.split('*2*')[1].strip()
            except (IndexError, ValueError):
                try:
                    # Second method
                    question = text.split('*1*')[1].split('*1*')[0].strip()
                    answer = text.split('*2*')[1].split('*2*')[0].strip()
                except (IndexError, ValueError):
                    # Skip the row if both methods fail
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
