# All Rights Reserved
# 
# Copyright © 2024 Patrick Devaney
# 
# THE CONTENTS OF THIS PROJECT ARE PROPRIETARY AND CONFIDENTIAL. UNAUTHORIZED COPYING, TRANSFERRING, OR REPRODUCTION OF THE CONTENTS OF THIS PROJECT, VIA ANY MEDIUM, IS STRICTLY PROHIBITED.
# 
# The receipt or possession of the source code and/or any parts thereof does not convey or imply any right to use them for any purpose other than the purpose for which they were provided to you.
# 
# The software is provided “AS IS”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the software.

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
    prompt = f"Role: character from ancient fantasy RPG. in character Q&A about {subject}. Q: less than 15 words. A: less than 35 words. Enclose Q in *1* <question> *1* and A in *2* <answer> *2*. Do not include any commentary or notes on output."

    generated_text = text_generator(prompt, max_length=165, truncation=True)[0]['generated_text']
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

# Save the generated texts to a .txt file with utf-8 encoding
with open('output_txt_SPLIT_NUM.txt', 'w', encoding='utf-8') as txt_file:
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
        with open(f'input_csv_split_{split_num}.py', 'w', encoding='utf-8') as f:
            f.write(script_content)

# Create Python scripts for each split
create_python_scripts()
