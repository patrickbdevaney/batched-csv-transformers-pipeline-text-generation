# Import necessary libraries
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

# Function to generate new subjects based on the current list
def generate_new_subjects(current_subjects):
    prompt = f"Generate a new subject related to a fantasy roleplaying game setting. Current subjects: {', '.join(current_subjects)}"
    output = text_generator(prompt, max_length=1, truncation=True)[0]['generated_text']
    new_subject = output.strip()
    return new_subject

# Start with a few seed subjects
seed_subjects = ["dragon", "sword", "magic"]
all_subjects = set(seed_subjects)

# Generate new subjects iteratively with a retry mechanism
num_iterations = 100  # Adjust the number of iterations as needed
max_retries = 10  # Maximum number of retries for generating a unique subject

for _ in range(num_iterations):
    retries = 0
    while retries < max_retries:
        new_subject = generate_new_subjects(list(all_subjects))
        if new_subject not in all_subjects:
            all_subjects.add(new_subject)
            break
        else:
            retries += 1
            print(f"Duplicate subject found: {new_subject}")
    if retries == max_retries:
        print("Max retries reached, moving to the next iteration.")

# Convert the result to a pandas DataFrame and save to CSV
result_df = pd.DataFrame(list(all_subjects), columns=['subject'])
result_df.to_csv('subject.csv', index=False)
print("Generated subjects written to subject.csv")

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()