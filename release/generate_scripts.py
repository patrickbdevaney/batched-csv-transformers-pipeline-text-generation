import os
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import re

# Function to create Python scripts for each split
def create_python_scripts():
    template = """
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import re

# Load the dataset from Hugging Face Datasets library
print("Loading the dataset from Hugging Face Datasets library...")
dataset = load_dataset('csv', data_files='./input_csv_split_SPLIT_NUM.csv')['train']
print("Dataset loaded successfully.")

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")

# Function to generate dialogue based on the subject
def generate_dialogue(subject):
    prompt = f"You are a character in an ancient fantasy roleplaying game. Write a brief question and answer about {subject}. Each should be one sentence and 50 words or less. Enclose the question in *1* <question> *1* and the answer in *2* <answer> *2*."
    generated_text = text_generator(prompt, max_length=256, truncation=True)[0]['generated_text']
    dialogue = re.sub(re.escape(prompt), "", generated_text).strip()
    return dialogue

# Process the dataset using the pipeline with KeyDataset
dialogues = []
for subject in tqdm(KeyDataset(dataset, "subject")):
    dialogue = generate_dialogue(subject)
    dialogues.append({'subject': subject, 'dialogue': dialogue})

# Convert the result to a pandas DataFrame and save to CSV
result_df = pd.DataFrame(dialogues)
result_df.to_csv('output_csv_SPLIT_NUM.csv', index=False)
print("All dialogues written to output_csv_SPLIT_NUM.csv")

# Save the generated texts to a .txt file
with open('output_txt_SPLIT_NUM.txt', 'w') as txt_file:
    for dialogue in dialogues:
        txt_file.write(dialogue['dialogue'] + '\\n')
print("All dialogues written to output_txt_SPLIT_NUM.txt")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared.")
"""

    # Walk the directory to get the names of the input CSV files
    input_files = [f for f in os.listdir() if f.startswith('input_csv_split_') and f.endswith('.csv')]
    for input_file in input_files:
        print("Processing file:", input_file)  # Print the filename being processed
        split_num = input_file.split('_')[-1].split('.')[0]
        script_content = template.replace('SPLIT_NUM', split_num)
        with open(f'input_csv_split_{split_num}.py', 'w') as f:
            f.write(script_content)

# Create Python scripts for each split
create_python_scripts()
