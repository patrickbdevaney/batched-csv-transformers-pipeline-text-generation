# Import necessary libraries
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from datasets import load_dataset
# Receives one 36000 row csv that I prepared, and outputs csvs in chunks. Outputting to a single file can work, but I think dividing them is a bit safer with large outputs 

# Load the dataset from Hugging Face Datasets library
print("Loading the dataset from Hugging Face Datasets library...")
dataset = load_dataset('csv', data_files='./subject.csv')
df = pd.DataFrame(dataset['train'])
print("Dataset loaded successfully.")

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

# Process the dataset in smaller batches to avoid running out of memory
batch_size = 25
total_batches = (len(df) + batch_size - 1) // batch_size
print(f"Total number of batches: {total_batches}")

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i + batch_size]
    dialogues = []
    for subject in batch_df['subject']:
        dialogue = generate_dialogue(subject)
        dialogues.append({'subject': subject, 'Dialogue': dialogue})
    
    # Convert the result to a pandas DataFrame and save to CSV
    result_df = pd.DataFrame(dialogues)
    batch_file = f'output_{i // batch_size + 1}.csv'
    result_df.to_csv(batch_file, index=False)
    print(f"Batch {i // batch_size + 1} written to {batch_file}")

print("All dialogues written to batch output CSV files.")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()