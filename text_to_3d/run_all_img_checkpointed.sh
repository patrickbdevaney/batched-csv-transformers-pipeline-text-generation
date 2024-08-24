#!/bin/bash

for script in generate_images_split_*.py; do
    completion_file="output_img_desc_input_csv_split_${script:22:2}.txt"
    if [ ! -f "$completion_file" ]; then
        echo "Running $script..."
        python "$script"
    else
        echo "Skipping $script, already completed."
    fi
done