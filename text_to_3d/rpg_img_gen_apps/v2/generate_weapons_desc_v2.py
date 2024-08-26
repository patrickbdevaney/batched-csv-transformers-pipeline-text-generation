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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 100 words of intricate varied weapons from various ancient bronze age europe mythological fantasy RPG settings different from {subject}. each example very different alternating weapon type staves bows crossbows all melee weapon types, colorful material and shape description and style/influence (e.g. fantasy elf, demon, god, atlantean, dragon, dwarf). The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=230, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["Celtic longsword with a blade of emerald metal, its hilt wrapped in green leather and adorned with a sapphire. The blade has a dark star-studded galactic pattern, making it look otherworldly",
"Germanic spear with a shaft of cobalt sapphire, tipped with a blood-red steel head. The spear is lightweight yet incredibly durable, with intricate runes etched along its length",
"Hittite spatha with a blade of obsidian, its hilt crafted from polished ebony and inlaid with gold filigree. The weapon is both beautiful and deadly, with a balance that makes it perfect for swift, precise strikes",
"Mycenaean falchata with a curved blade of dark iron, its hilt wrapped in black leather and studded with onyx. The weapon has a menacing appearance, and its weight gives it a powerful swing",
"Scythian javelin with a shaft of lightweight mithril, tipped with a barbed steel head. The javelin is designed for throwing, with a sleek, aerodynamic shape that allows it to fly true",
"Sumerian sling made from braided dragonhide, with a pouch of reinforced leather. The sling is simple yet effective, capable of hurling stones with great force",
"Celtic halberd with a shaft of sturdy ash wood, tipped with a broad steel axe head and a pointed spike. The weapon is versatile, capable of both slashing and thrusting attacks",
"Germanic lucerne hammer with a haft of reinforced oak, topped with a heavy iron head. The hammer is designed for crushing armor, with a spiked back for piercing",
"Hittite gladius with a blade of tempered steel, its hilt wrapped in brown leather and capped with a bronze pommel. The weapon is short and sturdy, perfect for close combat",
"Mycenaean dagger with a blade of gleaming silver, its hilt wrapped in blue silk and adorned with a sapphire. The dagger is lightweight and easy to conceal, making it ideal for stealthy attacks",
"Scythian war axe with a head of blackened iron, its haft wrapped in brown leather and studded with iron rivets. The weapon is heavy and powerful, capable of cleaving through armor",
"Sumerian mace with a head of spiked steel, its haft wrapped in black leather. The weapon is designed for crushing blows, with a weight that makes it devastating in combat",
"Celtic longsword with a blade of enchanted silver, its hilt wrapped in white leather and inlaid with gold. The weapon is elegant and powerful, with a keen edge that can cut through armor",
"Germanic rapier with a blade of tempered steel, its hilt wrapped in black leather and adorned with a diamond. The weapon is lightweight and designed for thrusting attacks, with a narrow blade that can pierce armor",
"Hittite battleaxe with a head of gleaming steel, its haft wrapped in brown leather and studded with iron rivets. The weapon is heavy and powerful, capable of cleaving through armor",
"Mycenaean morningstar with a head of spiked iron, its haft wrapped in black leather. The weapon is designed for crushing blows, with a weight that makes it devastating in combat",
"Scythian crossbow with a stock of polished oak, its limbs made of reinforced steel. The weapon is designed for ranged attacks, with a mechanism that allows it to fire bolts with great force",
"Sumerian bow made from yew wood, its string of braided silk. The weapon is designed for ranged attacks, with a sleek, aerodynamic shape that allows it to fire arrows with great accuracy",
"Celtic staff made from enchanted oak, its head adorned with a crystal orb. The weapon is designed for casting spells, with a length that makes it perfect for channeling magical energy",
"Germanic wand made from dragonbone, its length adorned with runes. The weapon is designed for casting spells, with a lightweight design that makes it easy to wield",
"Hittite scepter made from gold, its head adorned with a large ruby. The weapon is designed for casting spells, with a length that makes it perfect for channeling magical energy",
"Mycenaean trident with a shaft of reinforced steel, its head adorned with barbed prongs. The weapon is designed for thrusting attacks, with a length that makes it perfect for underwater combat",
"Scythian flail with a head of spiked iron, its chain made of reinforced steel. The weapon is designed for crushing blows, with a weight that makes it devastating in combat",
"Sumerian warhammer with a head of blackened iron, its haft wrapped in brown leather and studded with iron rivets. The weapon is heavy and powerful, capable of crushing armor",
"Celtic claymore with a blade of tempered steel, its hilt wrapped in black leather and adorned with a diamond. The weapon is designed for slashing attacks, with a length that makes it perfect for two-handed combat",
"Germanic broadsword with a blade of polished steel, its hilt wrapped in red leather and adorned with a ruby. The weapon is designed for slashing attacks, with a broad blade that makes it perfect for powerful swings",
"Hittite scythe with a blade of dark iron, its haft wrapped in black leather. The weapon is designed for slashing attacks, with a curved blade that makes it perfect for reaping",
"Mycenaean pike with a shaft of reinforced oak, its head adorned with a barbed steel point. The weapon is designed for thrusting attacks, with a length that makes it perfect for keeping enemies at bay",
"Scythian glaive with a blade of polished steel, its haft wrapped in brown leather. The weapon is designed for slashing attacks, with a length that makes it perfect for sweeping strikes"]

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