import os
import re
import pandas as pd
import torch
import gc
import gradio as gr
from tqdm.auto import tqdm
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

dtype = torch.bfloat16

# Function to find the number of splits based on the name format
def find_num_splits(directory):
    pattern = re.compile(r'img_desc_input_csv_split_(\d+)\.csv')
    max_split = 0
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            split_num = int(match.group(1))
            if split_num > max_split:
                max_split = split_num
    return max_split

# Function to generate individual Python scripts for each segmented CSV file
def generate_python_scripts(directory):
    num_splits = find_num_splits(directory)
    for i in range(1, num_splits + 1):
        script_content = (
            "import pandas as pd\n"
            "from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL\n"
            "from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel\n"
            "from diffusers.pipelines.flux.pipeline_flux import FluxPipeline\n"
            "from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast\n"
            "import torch\n"
            "import gc\n"
            "import os\n"
            "from tqdm.auto import tqdm\n"
            "from optimum.quanto import freeze, qfloat8, quantize\n\n"
            "# Set environment variables for local path\n"
            "os.environ['FLUX_DEV'] = '.'\n"
            "os.environ['AE'] = '.'\n\n"
            "dtype = torch.bfloat16\n"
            "bfl_repo = 'black-forest-labs/FLUX.1-schnell'\n"
            "revision = 'refs/pr/1'\n\n"
            "scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder='scheduler', revision=revision)\n"
            "text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)\n"
            "tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)\n"
            "text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder='text_encoder_2', torch_dtype=dtype, revision=revision)\n"
            "tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder='tokenizer_2', torch_dtype=dtype, revision=revision)\n"
            "vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder='vae', torch_dtype=dtype, revision=revision)\n"
            "transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder='transformer', torch_dtype=dtype, revision=revision)\n\n"
            "quantize(transformer, weights=qfloat8)\n"
            "freeze(transformer)\n"
            "quantize(text_encoder_2, weights=qfloat8)\n"
            "freeze(text_encoder_2)\n\n"
            "pipe = FluxPipeline(\n"
            "    scheduler=scheduler,\n"
            "    text_encoder=text_encoder,\n"
            "    tokenizer=tokenizer,\n"
            "    text_encoder_2=None,\n"
            "    tokenizer_2=tokenizer_2,\n"
            "    vae=vae,\n"
            "    transformer=None,\n"
            ")\n"
            "pipe.text_encoder_2 = text_encoder_2\n"
            "pipe.transformer = transformer\n"
            "pipe.enable_model_cpu_offload()\n\n"
            "# Create a directory to save the generated images\n"
            f"output_dir = 'generated_images_{i}'\n"
            "os.makedirs(output_dir, exist_ok=True)\n\n"
            "# Function to generate an image based on the description\n"
            "def generate_image(description, seed=42):\n"
            "    prompt = f'{description}'\n"
            "    generator = torch.Generator().manual_seed(seed)\n"
            "    image = pipe(\n"
            "        prompt=prompt,\n"
            "        width=1024 ,\n"
            "        height=1024,\n"
            "        num_inference_steps=4,\n"
            "        generator=generator,\n"
            "        guidance_scale=3.5,\n"
            "    ).images[0]\n"
            "    return image\n\n"
            "# Load the segmented CSV file from the local directory\n"
            f"directory = '{directory}'\n"
            f"segment_file = os.path.join(directory, 'img_desc_input_csv_split_{i:02d}.csv').replace('\\\\', '/')\n"
            "print(f'Loading the dataset from {segment_file}...')\n"
            "dataset = pd.read_csv(segment_file)\n"
            "print('Dataset loaded successfully.')\n\n"
            "# Iterate over the dataset and generate images\n"
            "for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):\n"
            "    description = row['description']\n"
            "    image = generate_image(description)\n"
            "    image_path = os.path.join(output_dir, f\"{segment_file.split('.')[0]}_image_{idx}.png\")\n"
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
        with open(os.path.join(directory, script_filename), 'w') as script_file:
            script_file.write(script_content)
        print(f"Python script {script_filename} generated.")

# Generate Python scripts for the number of splits
directory = '.'  # Replace with the actual directory if needed
generate_python_scripts(directory)