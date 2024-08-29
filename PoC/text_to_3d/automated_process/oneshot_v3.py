import os
import pandas as pd
import torch
import gc
from tqdm.auto import tqdm
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import re

dtype = torch.bfloat16

# Set environment variables for local path
os.environ['FLUX_DEV'] = '.'
os.environ['AE'] = '.'

bfl_repo = 'black-forest-labs/FLUX.1-schnell'
revision = 'refs/pr/1'

# Load models
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder='scheduler', revision=revision)
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder='text_encoder_2', torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder='tokenizer_2', torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder='vae', torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder='transformer', torch_dtype=dtype, revision=revision)

# Quantize the models
quantize(transformer, weights=qfloat8)
freeze(transformer)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

# Initialize the pipeline
pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer
)
pipe.enable_model_cpu_offload()

# Create a directory to save the generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

# Function to sanitize the filename
def sanitize_filename(filename):
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Truncate to 50 characters
    return sanitized[:50]

# Function to generate an image based on the description
def generate_image(description, seed=42):
    prompt = f'detailed photorealistic full shot of {description}'
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=4,
        generator=generator,
        guidance_scale=3.5,
    ).images[0]
    return image

# Load the CSV file from the local directory
csv_file = 'descriptions.csv'
print(f'Loading the dataset from {csv_file}...')
dataset = pd.read_csv(csv_file)
print('Dataset loaded successfully.')

# Iterate over the dataset and generate images
for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
    description = row['description']
    image = generate_image(description)
    sanitized_description = sanitize_filename(description)
    image_path = os.path.join(output_dir, f"image_{sanitized_description}_{idx}.png")
    try:
        image.save(image_path)
        print(f'Image saved to {image_path}')
    except Exception as e:
        print(f'Error saving image to {image_path}: {e}')

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print('GPU memory cleared.')
