# Import necessary libraries
import os
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

# Initialize the text generation pipeline with 8-bit precision
print("Initializing the text generation pipeline with 8-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 8-bit precision.")

# Function to generate dialogue based on the subject
def generate_dialogue(subject):
    prompt = f"You are a video game character in an ancient fantasy roleplaying game. Write a question and answer about {subject} as that character if you get a subject matter from a certain setting act like a character in that setting. separate question and answer with a comma. do not include the prompt in your response, only the question and answer themselves."
    dialogue = text_generator(prompt, max_length=256, truncation=True)[0]['generated_text']
    return dialogue

# Function to process all CSV files in a directory
def process_all_csv_files(directory):
    all_dialogues = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                df = pd.read_csv(file_path)
                for subject in df['subject']:
                    dialogue = generate_dialogue(subject)
                    all_dialogues.append({'subject': subject, 'Dialogue': dialogue})
    
    # Convert the result to a pandas DataFrame and save to a single CSV file
    result_df = pd.DataFrame(all_dialogues)
    result_df.to_csv('all_dialogues.csv', index=False)
    print("All dialogues written to a single CSV file.")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Specify the directory containing the batch CSV files
directory = './path_to_your_csv_files'
process_all_csv_files(directory)
