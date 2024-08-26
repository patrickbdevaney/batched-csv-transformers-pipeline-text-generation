import csv

def stringify_csv(input_file, output_file):
    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            
            for row in reader:
                writer.writerow(row)
                
        print(f"Stringified CSV file saved to {output_file}")
    except csv.Error as e:
        print(f"Error processing CSV file: {e}")

# Example usage
input_file = 'fantasydesc.csv'
output_file = 'stringified_fantasydesc.csv'
stringify_csv(input_file, output_file)