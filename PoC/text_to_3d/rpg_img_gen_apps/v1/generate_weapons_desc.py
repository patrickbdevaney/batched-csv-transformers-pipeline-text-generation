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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 50 words of intricate varied weapons from various ancient bronze age iron age europe middle east far east mythological fantasy RPG settings different from {subject}. each example very different alternating weapon type staves bows crossbows all melee weapon types, colorful material and shape description and style/influence (e.g. fantasy elf, demon, god, atlantean, dragon, dwarf). The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=180, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["bronze Celtic longsword with a hilt wrapped in leather and decorated with knotwork", "Germanic iron axe with a double-bladed head, engraved with runes", "Mycenaean bronze spear with a leaf-shaped blade and intricate inlay", "Hittite war hammer with a bronze head shaped like a lion's maw", "Babylonian bronze short sword inlaid with lapis lazuli and celestial symbols", "Japanese katana with a folded steel blade and a dragon-themed tsuba", "elven silver bow adorned with ivy leaves and glowing runes", "dwarven warhammer with a square head and runic engravings", "demon’s curved scimitar, its blade wreathed in dark flames", "godly golden spear tipped with a radiant sunburst", "Celtic iron dagger with a triskele hilt and spiral patterns", "Germanic war club studded with iron spikes and wrapped in fur", "Mycenaean bronze double axe with a central sun motif", "Hittite bronze sickle sword with a crescent-shaped blade", "Babylonian bronze mace topped with a bull’s head and inlaid with gold", "Japanese yari spear with a straight, razor-sharp blade and lacquered shaft", "elven crystal dagger with a blade that glows with a soft inner light", "dwarven battle axe with a massive, double-edged iron head", "dragonbone bow with a string of woven dragon sinew and glowing runes", "demon’s whip made of fiery, molten chains", "divine sword with a blade made of pure light, glowing brilliantly", "Celtic spear with a broad bronze head and spiral engravings", "Germanic iron sword with a boar-shaped pommel and a notched blade", "Mycenaean bronze sword with a blade etched with swirling ocean patterns", "Hittite bronze spear with a winged sun disc at the base of the blade", "Babylonian sickle sword with a curved bronze blade and lapis lazuli handle", "Japanese tanto dagger with a carved bone handle and a polished blade", "elven longbow made of whitewood and adorned with silver leaf", "dwarven throwing axe with a broad, crescent-shaped iron blade", "dragonfire sword with a blade that glows with internal heat", "demon’s jagged dagger dripping with a dark, venomous liquid", "divine spear with a tip that glows with celestial energy", "Celtic war hammer with a stone head bound with iron bands", "Germanic throwing spear with a long, barbed iron tip", "Mycenaean double-edged sword with a bronze blade and gold inlay", "Hittite war axe with a broad, crescent-shaped bronze blade", "Babylonian bronze sword with a hilt shaped like a winged lion", "Japanese naginata with a curved blade and a shaft wrapped in silk", "elven rapier with a thin, silver blade and an intricately carved guard", "dwarven crossbow with a steel prod and a heavy iron bolt", "dragonfang dagger with a curved, serrated blade", "demon’s infernal flail with spiked, glowing orbs", "divine bow that shoots arrows of pure light", "Celtic bronze axe with a crescent-shaped blade and spiral designs", "Germanic iron spear with a boar-shaped tip and etched runes", "Mycenaean bronze dagger with a blade engraved with warrior scenes", "Hittite bronze mace with a spiked head and a decorated handle", "Babylonian war club with an obsidian head and gold inlay", "Japanese kanabo club studded with iron spikes", "elven glaive with a crystal blade that shimmers in the light", "dwarven hammer with a square head engraved with mountain motifs", "dragonbone sword with a blade that glows faintly in the dark", "demon’s scythe with a black, serrated blade and a smoky aura", "divine staff topped with a glowing orb of pure energy", "Celtic iron sword with a hilt adorned with animal motifs", "Germanic war axe with a broad, iron blade and a wolf-head pommel", "Mycenaean bronze spear with a blade shaped like a flame", "Hittite war chariot scythe blades of sharp, polished bronze", "Babylonian sickle with a curved bronze blade and a golden handle", "Japanese shuriken with sharpened edges and a star-shaped design", "elven shortsword with a delicate, curved blade and a leaf-shaped guard", "dwarven war pick with a heavy, spiked iron head", "dragonclaw mace with a head shaped like a dragon's talon", "demon’s cursed sword with a blade that glows with an eerie, red light", "divine mace with a head that radiates a soft, holy light", "Celtic bronze spear with an engraved, leaf-shaped blade", "Germanic war hammer with a round, iron head and fur-wrapped handle", "Mycenaean bronze sword with a tapered blade and gold-accented hilt", "Hittite bronze sickle sword with a crescent blade and sunburst design", "Babylonian bronze dagger with a lion-shaped pommel and lapis inlay", "Japanese nodachi sword with a long, curved blade and a dragon-carved hilt", "elven warhammer with a head shaped like a crystalline star", "dwarven battleaxe with a double-bladed iron head and runic carvings", "dragon's breath bow that ignites its arrows upon release", "demon’s twin daggers with blades that emit a dark, smoky aura", "divine spear that glows with the radiance of the sun", "Celtic bronze axe with a leaf-shaped blade and intricate knotwork", "Germanic sword with a broad, double-edged iron blade", "Mycenaean bronze dagger with a blade etched with waves", "Hittite war hammer with a lion-shaped bronze head", "Babylonian sickle sword with a curved blade and gold inlays", "Japanese tachi sword with a long, curved blade and intricate tsuba", "elven bow of whitewood with a string of woven silver thread", "dwarven war pick with a pointed iron head and engraved runes", "dragonfang sword with a serrated edge and a glowing hilt", "demon's cursed mace with a spiked head that drips with venom", "divine staff that channels radiant energy into powerful blasts", "Celtic iron sword with a hilt adorned with spiral patterns", "Germanic spear with a barbed iron tip and runic inscriptions", "Mycenaean bronze axe with a broad, double-edged blade", "Hittite war chariot blades with sharp, polished bronze edges", "Babylonian bronze mace with a lion-headed top and lapis inlay", "Japanese yari spear with a straight, razor-sharp blade", "elven rapier with a thin, silver blade and a leaf-shaped guard", "dwarven crossbow with a steel prod and a heavy iron bolt", "dragonfang dagger with a curved, serrated blade"]

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