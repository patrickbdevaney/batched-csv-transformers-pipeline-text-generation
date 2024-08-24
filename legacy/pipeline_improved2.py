# Import necessary libraries
import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from datasets import load_dataset

#sequentiall pipeline and prompt improved for performance. need to try to configure datasets with transformers and pipeline for parallel

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
    prompt = f"You are a character in an ancient fantasy roleplaying game. Write a brief question and answer about {subject} as a character in that setting. Each should be one sentence and 50 words or less."
    dialogue = text_generator(prompt, max_length=135, truncation=True)[0]['generated_text']
    return dialogue

# Function to parse question and answer from the generated dialogue
def parse_dialogue(dialogue):
    # Split the dialogue based on the question mark
    parts = dialogue.split('?')
    question = parts[0].strip() + '?' if len(parts) > 0 else ""
    answer = parts[1].strip() if len(parts) > 1 else ""
    return question, answer

# Process the dataset in larger batches to utilize more VRAM
batch_size = 60  # more than double
total_batches = (len(df) + batch_size - 1) // batch_size
print(f"Total number of batches: {total_batches}")

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i + batch_size]
    dialogues = []
    for subject in batch_df['subject']:
        dialogue = generate_dialogue(subject)
        question, answer = parse_dialogue(dialogue)  # Parse the dialogue into question and answer
        dialogues.append({'subject': subject, 'Question': question, 'Answer': answer})
    
    # Convert the result to a pandas DataFrame and save to CSV
    result_df = pd.DataFrame(dialogues)
    batch_file = f'output_{i // batch_size + 1}.csv'
    result_df.to_csv(batch_file, index=False)
    print(f"Batch {i // batch_size + 1} written to {batch_file}")

print("All dialogues written to batch output CSV files.")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()