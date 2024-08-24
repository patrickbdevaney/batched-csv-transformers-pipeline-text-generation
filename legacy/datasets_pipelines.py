import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
from datasets import load_dataset
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

# Load the dataset from Hugging Face Datasets library
print("Loading the dataset from Hugging Face Datasets library...")
dataset = load_dataset('csv', data_files='./subject.csv')['train']
print("Dataset loaded successfully.")

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")

# Function to parse question and answer from the generated dialogue
def parse_dialogue(dialogue):
    parts = dialogue.split('?')
    question = parts[0].strip() + '?' if len(parts) > 0 else ""
    answer = parts[1].strip() if len(parts) > 1 else ""
    return question, answer

# Process the dataset using the pipeline with KeyDataset
dialogues = []
for out in tqdm(text_generator(KeyDataset(dataset, "subject"), max_length=135, truncation=True)):
    for generated_text in out:
        dialogue = generated_text['generated_text']
        subject = dialogue.split('about ')[1].split('.')[0].strip()
        question, answer = parse_dialogue(dialogue)
        dialogues.append({'subject': subject, 'Question': question, 'Answer': answer})

# Convert the result to a pandas DataFrame and save to CSV
result_df = pd.DataFrame(dialogues)
result_df.to_csv('output.csv', index=False)
print("All dialogues written to output.csv")

# Print the size of the DataFrame and average length of dialogues
print(f"Number of dialogues: {len(result_df)}")
average_length = result_df['Question'].apply(len).mean() + result_df['Answer'].apply(len).mean()
print(f"Average length of dialogues: {average_length} characters")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared.")
