import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
import random
import os
from collections import deque

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")



# Function to generate a detailed visual description prompt
def generate_description_prompt(subject):
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 100 words of intricate varied architectural elements in ancient bronze age iron age celtic and insular monastic, minoan, etruscan, mycenaean, and medieval like frankish and venetian style different from {subject}. each example very different alternating frieze columns, portico, archway atrium, arcade, gate- modular elements to compose buildings colorful materials varied marble wood stone etc. The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=230, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["A high elven marble archway adorned with intricate vine patterns and sparkling crystal inlays, the arch is tall and slender, perfect for grand entrances.",
"A Venetian portico with a series of semi-circular arches supported by fluted columns, the marble is polished to a mirror finish and decorated with floral motifs.",
"A Norman castle portcullis made of wrought iron, featuring a lattice design and reinforced with rivets, the gate is both imposing and functional.",
"A Mycenaean column with a fluted shaft and a capital decorated with spiral patterns, the marble has a warm, golden hue.",
"An elven atrium with a domed ceiling and walls adorned with delicate filigree, the space is filled with natural light and features a central fountain.",
"A Venetian marble balcony with a balustrade of interlocking geometric patterns, the balcony overlooks a serene canal and is adorned with potted plants.",
"A Norman castle turret with crenellated battlements and arrow slits, the stone is rugged and weathered, perfect for defense.",
"A Minoan fresco depicting scenes of nature and mythology, the vibrant colors and intricate details bring the walls to life.",
"An elven marble column with a slender, fluted shaft and a capital adorned with leaf motifs, the column is both elegant and sturdy.",
"A Venetian archway with a pointed arch and intricate scrollwork, the marble is polished to a high sheen and decorated with gold leaf.",
"A Norman castle gatehouse with a heavy wooden door reinforced with iron bands, the gatehouse is designed for security and defense.",
"A Mycenaean throne room with a raised dais and a throne carved from marble, the walls are adorned with frescoes depicting scenes of royalty.",
"An elven marble staircase with a balustrade of delicate filigree, the steps are wide and shallow, perfect for graceful ascents.",
"A Venetian courtyard with a central fountain and marble benches, the space is surrounded by arched colonnades and filled with potted plants.",
"A Norman castle keep with thick stone walls and narrow windows, the keep is designed to withstand sieges and provide a last line of defense.",
"A Minoan column with a bulbous capital and a shaft decorated with spiral patterns, the marble has a warm, reddish hue.",
"An elven marble portico with a series of slender columns and a roof adorned with intricate carvings, the portico is perfect for grand entrances.",
"A Venetian marble bridge with a series of arches and a balustrade of interlocking geometric patterns, the bridge spans a serene canal.",
"A Norman castle courtyard with a central well and cobblestone paving, the courtyard is surrounded by high stone walls and towers.",
"A Mycenaean palace gate with a heavy wooden door reinforced with bronze bands, the gate is adorned with intricate carvings and motifs.",
"An elven marble atrium with a domed ceiling and walls adorned with delicate filigree, the space is filled with natural light and features a central fountain.",
"A Venetian marble colonnade with a series of fluted columns and a roof adorned with floral motifs, the colonnade is perfect for grand entrances.",
"A Norman castle drawbridge with a heavy wooden platform and iron chains, the drawbridge is designed for security and defense.",
"A Minoan fresco depicting scenes of nature and mythology, the vibrant colors and intricate details bring the walls to life.",
"An elven marble column with a slender, fluted shaft and a capital adorned with leaf motifs, the column is both elegant and sturdy.",
"A Venetian archway with a pointed arch and intricate scrollwork, the marble is polished to a high sheen and decorated with gold leaf.",
"A Norman castle gatehouse with a heavy wooden door reinforced with iron bands, the gatehouse is designed for security and defense.",
"A Mycenaean throne room with a raised dais and a throne carved from marble, the walls are adorned with frescoes depicting scenes of royalty.",
"An elven marble staircase with a balustrade of delicate filigree, the steps are wide and shallow, perfect for graceful ascents.",
"A Venetian courtyard with a central fountain and marble benches, the space is surrounded by arched colonnades and filled with potted plants.",
"A Norman castle keep with thick stone walls and narrow windows, the keep is designed to withstand sieges and provide a last line of defense.",
"A Minoan column with a bulbous capital and a shaft decorated with spiral patterns, the marble has a warm, reddish hue.",
"An elven marble portico with a series of slender columns and a roof adorned with intricate carvings, the portico is perfect for grand entrances.",
"A Venetian marble bridge with a series of arches and a balustrade of interlocking geometric patterns, the bridge spans a serene canal.",
"A Norman castle courtyard with a central well and cobblestone paving, the courtyard is surrounded by high stone walls and towers.",
"A Mycenaean palace gate with a heavy wooden door reinforced with bronze bands, the gate is adorned with intricate carvings and motifs.",
"An elven marble atrium with a domed ceiling and walls adorned with delicate filigree, the space is filled with natural light and features a central fountain.",
"A Venetian marble colonnade with a series of fluted columns and a roof adorned with floral motifs, the colonnade is perfect for grand entrances.",
"A Norman castle drawbridge with a heavy wooden platform and iron chains, the drawbridge is designed for security and defense.",
"A Minoan fresco depicting scenes of nature and mythology, the vibrant colors and intricate details bring the walls to life.",
"An elven marble column with a slender, fluted shaft and a capital adorned with leaf motifs, the column is both elegant and sturdy.",
"A Venetian archway with a pointed arch and intricate scrollwork, the marble is polished to a high sheen and decorated with gold leaf.",
"A Norman castle gatehouse with a heavy wooden door reinforced with iron bands, the gatehouse is designed for security and defense.",
"A Mycenaean throne room with a raised dais and a throne carved from marble, the walls are adorned with frescoes depicting scenes of royalty.",
"An elven marble staircase with a balustrade of delicate filigree, the steps are wide and shallow, perfect for graceful ascents.",
"A Venetian courtyard with a central fountain and marble benches, the space is surrounded by arched colonnades and filled with potted plants.",
"A Norman castle keep with thick stone walls and narrow windows, the keep is designed to withstand sieges and provide a last line of defense.",
"A Minoan column with a bulbous capital and a shaft decorated with spiral patterns, the marble has a warm, reddish hue.",
"An elven marble portico with a series of slender columns and a roof adorned with intricate carvings, the portico is perfect for grand entrances.",
"A Venetian marble bridge with a series of arches and a balustrade of interlocking geometric patterns, the bridge spans a serene canal.",
"A Norman castle courtyard with a central well and cobblestone paving, the courtyard is surrounded by high stone walls and towers.",
"A Mycenaean palace gate with a heavy wooden door reinforced with bronze bands, the gate is adorned with intricate carvings and motifs.",
"An elven marble atrium with a domed ceiling and walls adorned with delicate filigree, the space is filled with natural light and features a central fountain.",
"A Venetian marble colonnade with a series of fluted columns and a roof adorned with floral motifs, the colonnade is perfect for grand entrances.",
"A Norman castle drawbridge with a heavy wooden platform and iron chains, the drawbridge is designed for security and defense.",
"A Minoan fresco depicting scenes of nature and mythology, the vibrant colors and intricate details bring the walls to life.",
"An elven marble column with a slender, fluted shaft and a capital adorned with leaf motifs, the column is both elegant and sturdy.",
"A Venetian archway with a pointed arch and intricate scrollwork, the marble is polished to a high sheen and decorated with gold leaf.",
"A Norman castle gatehouse with a heavy wooden door reinforced with iron bands, the gatehouse is designed for security and defense.",
"A Mycenaean throne room with a raised dais and a throne carved from marble, the walls are adorned with frescoes depicting scenes of royalty.",
"An elven marble staircase with a balustrade of delicate filigree, the steps are wide and shallow, perfect for graceful ascents.",
"A Venetian courtyard with a central fountain and marble benches, the space is surrounded by arched colonnades and filled with potted plants.",
"A Norman castle keep with thick stone walls and narrow windows, the keep is designed to withstand sieges and provide a last line of defense.",
"A Minoan column with a bulbous capital and a shaft decorated with spiral patterns, the marble has a warm, reddish hue.",
"An elven marble portico with a series of slender columns and a roof adorned with intricate carvings, the portico is perfect for grand entrances.",
"A Venetian marble bridge with a series of arches and a balustrade of interlocking geometric patterns, the bridge spans a serene canal.",
"A Norman castle courtyard with a central well and cobblestone paving, the courtyard is surrounded by high stone walls and towers.",
"A Mycenaean palace gate with a heavy wooden door reinforced with bronze bands, the gate is adorned with intricate carvings and motifs.",
"An elven marble atrium with a domed ceiling and walls adorned with delicate filigree, the space is filled with natural light and features a central fountain.",
"A Venetian marble colonnade with a series of fluted columns and a roof adorned with floral motifs, the colonnade is perfect for grand entrances."]

used_words = set()

def generate_and_write_to_files(output_csv, output_txt, batch_size=100):
    descriptions = []
    description_queue = deque()

    # Check if the CSV file already exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        descriptions = existing_df.to_dict('records')

    while True:
        # Select a subject that has not been used
        available_subjects = [word for word in seed_words if word not in used_words]
        if not available_subjects:
            print("No more available subjects to use.")
            break

        subject = random.choice(available_subjects)
        generated_description = generate_description_prompt(subject)
        
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

    # Write any remaining descriptions in the queue
    if description_queue:
        batch_write_to_files(description_queue, output_csv, output_txt)

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

# Clear GPU memory when the process is closed
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory cleared.")

# Run the function indefinitely
try:
    generate_and_write_to_files('descriptions.csv', 'descriptions.txt')
finally:
    clear_gpu_memory()