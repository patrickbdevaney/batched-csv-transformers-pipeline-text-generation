import os
import pandas as pd
import torch
import gc
import re
import random
from tqdm.auto import tqdm
from collections import deque
from optimum.quanto import freeze, qfloat8, quantize
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import gradio as gr

dtype = torch.bfloat16

# Set environment variables for local path
os.environ['FLUX_DEV'] = '.'
os.environ['AE'] = '.'

bfl_repo = 'black-forest-labs/FLUX.1-schnell'
revision = 'refs/pr/1'

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder='scheduler', revision=revision)
text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=dtype)
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder='text_encoder_2', torch_dtype=dtype, revision=revision)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder='tokenizer_2', torch_dtype=dtype, revision=revision)
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder='vae', torch_dtype=dtype, revision=revision)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder='transformer', torch_dtype=dtype, revision=revision)

quantize(transformer, weights=qfloat8)
freeze(transformer)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=None,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=None,
)
pipe.text_encoder_2 = text_encoder_2
pipe.transformer = transformer
pipe.enable_model_cpu_offload()

# Create a directory to save the generated images
output_dir = 'generated_images'
os.makedirs(output_dir, exist_ok=True)

# Function to generate a detailed visual description prompt
def generate_description_prompt(subject, user_prompt):
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 100 words of {user_prompt} from {subject}. "
    try:
        generated_text = text_generator(prompt, max_length=230, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None

# Function to parse descriptions from a given text
def parse_descriptions(text):
    # Find all descriptions enclosed in brackets
    descriptions = re.findall(r'\[([^\[\]]+)\]', text)
    # Filter descriptions with at least 3 words
    descriptions = [desc.strip() for desc in descriptions if len(desc.split()) >= 3]
    return descriptions

# Seed words pool
seed_words = []

used_words = set()
paused = False

def generate_and_write_to_files(output_csv, output_txt, user_prompt, batch_size=100, max_iterations=1500):
    global paused
    descriptions = []
    description_queue = deque()
    iteration_count = 0

    # Initialize the text generation pipeline with 16-bit precision
    print("Initializing the text generation pipeline with 16-bit precision...")
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    print("Text generation pipeline initialized with 16-bit precision.")

    # Check if the CSV file already exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        descriptions = existing_df.to_dict('records')

    while iteration_count < max_iterations:
        if paused:
            break

        # Select a subject that has not been used
        available_subjects = [word for word in seed_words if word not in used_words]
        if not available_subjects:
            print("No more available subjects to use.")
            break

        subject = random.choice(available_subjects)
        generated_description = generate_description_prompt(subject, user_prompt)
        
        if generated_description:
            # Remove any offending symbols
            clean_description = generated_description.encode('ascii', 'ignore').decode('ascii')
            description_queue.append({'subject': subject, 'description': clean_description})

            # Print the generated description to the command line
            print(f"Generated description for subject '{subject}': {clean_description}")

            # Update used words and seed words
            used_words.add(subject)
            seed_words.append(clean_description)  # Add the generated description to the seed bank array

            # Batch write descriptions to CSV and TXT files
            if len(description_queue) >= batch_size:
                batch_write_to_files(description_queue, output_csv, output_txt)
                description_queue.clear()

            # Parse and append descriptions every 50 iterations
            if iteration_count % 50 == 0:
                parse_and_append_descriptions(output_csv, 'descriptions_parsed.csv')

        iteration_count += 1

    # Write any remaining descriptions in the queue
    if description_queue:
        batch_write_to_files(description_queue, output_csv, output_txt)

    # Clear GPU memory after 1500 iterations
    if iteration_count >= max_iterations:
        torch.cuda.empty_cache()
        gc.collect()
        print('GPU memory cleared.')

def batch_write_to_files(description_queue, output_csv, output_txt):
    # Convert the result to a pandas DataFrame and save to CSV
    result_df = pd.DataFrame(description_queue)
    result_df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))
    print(f"Batch of descriptions written to {output_csv}")

    # Save the generated texts to a .txt file with utf-8 encoding
    with open(output_txt, 'a', encoding='utf-8') as txt_file:
        for description in description_queue:
            txt_file.write(description['description'] + '\n')
    print(f"Batch of descriptions written to {output_txt}")

def parse_and_append_descriptions(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Initialize a list to store parsed descriptions
    parsed_descriptions = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        descriptions = parse_descriptions(row['description'])
        parsed_descriptions.extend(descriptions)

    # Create a new DataFrame with the parsed descriptions
    parsed_df = pd.DataFrame(parsed_descriptions, columns=['description'])

    # Write the parsed descriptions to a new CSV file
    parsed_df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))

    print(f"Parsed descriptions written to {output_csv}")

# Function to generate an image based on the description
def generate_image(description, seed=42):
    prompt = f'detailed photorealistic full shot of {description}'
    generator = torch.Generator().manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=1920,
        height=1080,
        num_inference_steps=4,
        generator=generator,
        guidance_scale=3.5,
    ).images[0]
    return image

# Function to generate images from parsed descriptions
def generate_images_from_parsed_descriptions():
    # Load the CSV file from the local directory
    csv_file = 'descriptions_parsed.csv'
    print(f'Loading the dataset from {csv_file}...')
    dataset = pd.read_csv(csv_file)
    print('Dataset loaded successfully.')

    # Iterate over the dataset and generate images
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        description = row['description']
        image = generate_image(description)
        image_path = os.path.join(output_dir, f"image_{idx}.png")
        image.save(image_path)
        print(f'Image saved to {image_path}')

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    print('GPU memory cleared.')

# Clear GPU memory when the process is closed
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory cleared.")

# Gradio UI
def gradio_interface(seed_input, user_prompt):
    global seed_words, paused
    seed_words = [word.strip() for word in seed_input.split(',')]
    paused = False
    generate_and_write_to_files('descriptions.csv', 'descriptions.txt', user_prompt)
    # Unload LLaMA model to free up VRAM
    del model
    del text_generator
    torch.cuda.empty_cache()
    gc.collect()
    print("LLaMA model unloaded and GPU memory cleared.")
    # Load Flux Schnell model and generate images
    generate_images_from_parsed_descriptions()

def pause_generation():
    global paused, seed_words, used_words
    paused = True
    seed_words = []
    used_words = set()
    print("Generation paused and seed word bank reset.")

# Function to display descriptions in batches of 20
def display_descriptions():
    descriptions_df = pd.read_csv('descriptions_parsed.csv', header=None, names=['description'])
    descriptions_list = descriptions_df['description'].tolist()
    batch_size = 20
    for i in range(0, len(descriptions_list), batch_size):
        yield descriptions_list[i:i + batch_size]

# Create Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter seed words separated by commas", label="Seed Words"),
        gr.inputs.Textbox(lines=2, placeholder="Enter your prompt", label="User Prompt")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Generated Description"),
        gr.Textbox(label="Batch of Generated Descriptions")
    ],
    live=True
)

# Add a pause button
iface.add_component(
    gr.Button("Pause", variant="primary", on_click=pause_generation)
)

# Function to handle the entire workflow
def run_workflow(seed_input, user_prompt):
    gradio_interface(seed_input, user_prompt)
    generate_images_from_parsed_descriptions()

# Create Gradio interface for the entire workflow
workflow_iface = gr.Interface(
    fn=run_workflow,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter seed words separated by commas", label="Seed Words"),
        gr.inputs.Textbox(lines=2, placeholder="Enter your prompt", label="User Prompt")
    ],
    outputs="text",
    live=True
)

# Run the Gradio interface
workflow_iface.launch()

# Clear GPU memory when the process is closed
try:
    generate_and_write_to_files('descriptions.csv', 'descriptions.txt', user_prompt)
finally:
    clear_gpu_memory()