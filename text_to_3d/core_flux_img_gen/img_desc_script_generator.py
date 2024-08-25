import os

# Function to generate individual Python scripts for each segmented CSV file
def generate_python_scripts(num_splits):
    for i in range(1, num_splits + 1):
        script_content = (
            "import pandas as pd\n"
            "from diffusers import FluxPipeline\n"
            "import torch\n"
            "import gc\n"
            "import os\n"
            "from tqdm.auto import tqdm\n\n"
            "# Initialize the image generation pipeline\n"
            "print('Initializing the image generation pipeline...')\n"
            "model_id = 'black-forest-labs/FLUX.1-dev'\n"
            "pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)\n"
            "pipe.enable_model_cpu_offload()  # Save some VRAM by offloading the model to CPU\n"
            "print('Image generation pipeline initialized.')\n\n"
            "# Create a directory to save the generated images\n"
            "output_dir = 'generated_images'\n"
            "os.makedirs(output_dir, exist_ok=True)\n\n"
            "# Function to generate an image based on the description\n"
            "def generate_image(description, seed=42):\n"
            "    prompt = f'{description}'\n"
            "    image = pipe(\n"
            "        prompt,\n"
            "        output_type='pil',\n"
            "        num_inference_steps=4,\n"
            "        generator=torch.Generator('cpu').manual_seed(seed)\n"
            "    ).images[0]\n"
            "    return image\n\n"
            "# Load the dataset from the segmented CSV file\n"
            f"segment_file = 'img_desc_input_csv_split_{i:02d}.csv'\n"
            "print(f'Loading the dataset from {segment_file}...')\n"
            "dataset = pd.read_csv(segment_file)\n"
            "print('Dataset loaded successfully.')\n\n"
            "# Iterate over the dataset and generate images\n"
            "for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):\n"
            "    description = row['description']\n"
            "    image = generate_image(description)\n"
            "    image_path = os.path.join(output_dir, f'{segment_file.split('.')[0]}_image_{idx}.png')\n"
            "    image.save(image_path)\n"
            "    print(f'Image saved to {image_path}')\n\n"
            "# Write completion file\n"
            f"with open('output_img_desc_input_csv_split_{i:02d}.txt', 'w') as f:\n"
            "    f.write('1')\n"
            "print(f'Completion file written for {segment_file}')\n\n"
            "# Clear GPU memory\n"
            "torch.cuda.empty_cache()\n"
            "gc.collect()\n"
            "print('GPU memory cleared.')\n"
        )

        script_filename = f'generate_images_split_{i:02d}.py'
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)
        print(f"Python script {script_filename} generated.")

# Generate Python scripts for the number of splits
num_splits = 10  # Replace with the actual number of splits from the CSV splitting script
generate_python_scripts(num_splits)
