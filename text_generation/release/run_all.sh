#!/bin/bash

# Execute all Python scripts sequentially
for script in input_csv_split_*.py; 
do 
    python "$script"; 
done
