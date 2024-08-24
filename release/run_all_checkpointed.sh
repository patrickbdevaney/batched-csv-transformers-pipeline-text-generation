#!/bin/bash

# Loop through all input CSV split files
for input_file in input_csv_split_*.py
do
    # Extract the split number from the input file name
    split_num=$(echo $input_file | grep -oP '\d+')
    
    # Define the output CSV file name
    output_csv="output_csv_${split_num}.csv"
    
    # Check if the output CSV file already exists
    if [ -f "$output_csv" ]; then
        echo "$output_csv already exists. Skipping $input_file"
    else
        echo "Running $input_file"
        python $input_file
    fi
done
