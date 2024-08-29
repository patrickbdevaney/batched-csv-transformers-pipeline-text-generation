import json
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
import random
import os
from collections import deque

# Function to sanitize the description
def sanitize_description(description):
    return re.sub(r'[<>:"/\\|?*]', '_', description)

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

prompt = config['prompt']
seed_words = config['seed_words']

# Function to generate a detailed visual description prompt
def generate_description_prompt(subject):
    full_prompt = f"{prompt} different from {subject}"
    try:
        generated_text = text_generator(full_prompt, max_length=330, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(full_prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None

# Seed words pool
used_words = set()
parsed_descriptions = []

def generate_and_write_to_csv(output_csv, max_count=1):
    description_queue = deque()

    # Check if the CSV file already exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        parsed_descriptions.extend(existing_df['description'].tolist())

    while len(parsed_descriptions) < max_count:
        # Select a subject that has not been used
        available_subjects = [word for word in seed_words if word not in used_words]
        if not available_subjects:
            print("No more available subjects to use.")
            break

        subject = random.choice(available_subjects)
        generated_description = generate_description_prompt(subject)
        
        if generated_description:
            # Remove any offending symbols
            clean_description = sanitize_description(generated_description)
            description_queue.append({'description': clean_description})

            # Print the generated description to the command line
            print(f"Generated description for subject '{subject}': {clean_description}")

            # Update used words
            used_words.add(subject)

            # Parse and store the description
            parsed_description = re.search(r'\[(.*?)\]', clean_description)
            if parsed_description:
                parsed_descriptions.append(parsed_description.group(1))

            # Check if we have collected enough descriptions
            if len(parsed_descriptions) >= max_count:
                break

    # Write all collected descriptions to CSV
    if description_queue:
        result_df = pd.DataFrame(description_queue)
        result_df.to_csv(output_csv, index=False, mode='a', header=not os.path.exists(output_csv))
        print(f"Batch of descriptions written to {output_csv}")

# Main execution
generate_and_write_to_csv('descriptions.csv', max_count=1)

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print('GPU memory cleared.')
