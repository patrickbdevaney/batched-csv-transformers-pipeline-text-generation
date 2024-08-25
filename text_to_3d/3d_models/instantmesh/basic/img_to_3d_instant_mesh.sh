#!/bin/bash

#assumes for simplcity you use instant mesh, images and this script are in
#the root of their repo
# Directory containing the generated images
image_dir="."

# Path to the run.py script and configuration file
run_script_path="./run.py"
config_file="configs/instant-mesh-large.yaml"

# Loop over all images in the directory
for image in "$image_dir"/*.png; do
    echo "Processing $image..."
    python "$run_script_path" "$config_file" "$image" --save_video
    echo "Finished processing $image"
done
