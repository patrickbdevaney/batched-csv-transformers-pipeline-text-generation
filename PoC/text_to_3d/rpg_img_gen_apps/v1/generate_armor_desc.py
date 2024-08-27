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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 50 words of intricate varied armor pieces from various ancient bronze age iron age europe middle east far east mythological fantasy RPG settings different from {subject}. each example very different alternating armor piece, colorful material and shape description and style/influence (for fantasy elf, demon, god, atlantean, dragon, dwarf,). The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=180, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["bronze Celtic helmet adorned with spiraled ram horns and knotwork patterns", "Germanic leather bracers embossed with runes and reinforced with iron studs", "Mycenaean bronze cuirass etched with scenes of sea monsters and warriors", "Hittite bronze scale armor with overlapping plates resembling dragon scales", "Babylonian bronze breastplate inlaid with lapis lazuli and celestial symbols", "Japanese kabuto helmet featuring a dragon crest and intricate gold rivets", "elven silver breastplate shaped like overlapping leaves, shimmering in light", "dwarven iron gauntlets with interlocking gears and engraved runes", "demon’s spiked pauldrons glowing with infernal red light", "godly golden cuirass adorned with radiant sunburst designs", "Celtic torc collar of twisted bronze, terminating in snarling wolf heads", "Germanic iron helmet with an engraved boar crest and fur lining", "Mycenaean greaves decorated with spiraling patterns and gold accents", "Hittite war mask featuring a fearsome lion visage in burnished bronze", "Babylonian ceremonial helm topped with a golden bull's head", "Japanese lacquered chestplate with a phoenix motif in gold and red", "elven chainmail shirt woven with fine mithril links, glimmering softly", "dwarven plate boots with intricate engravings of mountains and hammers", "dragon-scaled armor with scales that shift colors in the light", "demonic horned helmet that emits a dark, ominous aura", "golden bracers inscribed with divine symbols that glow faintly", "Celtic bronze shield with a triskele pattern and iron boss", "Germanic wolfskin cloak fastened with a bronze brooch", "Mycenaean bronze helmet with a horsehair plume and cheek guards", "Hittite chestplate with engraved winged sun discs and protective scales", "Babylonian scale mail made of bronze, interwoven with gold wire", "Japanese samurai kote arm guards with intricate dragonfly patterns", "elven leather boots embossed with delicate leaf designs", "dwarven iron chestplate with thick rivets and angular designs", "dragonbone gauntlets with claws that extend over the fingers", "demon's chestplate covered in jagged spikes and glowing sigils", "divine armor made of light itself, shifting in color and intensity", "Celtic iron chainmail, with polished links woven into intricate patterns", "Germanic wooden shield reinforced with iron bands and a central boss", "Mycenaean bronze greaves adorned with swirling oceanic motifs", "Hittite conical helmet decorated with eagle feathers and bronze studs", "Babylonian ornate cuirass with lion reliefs and golden accents", "Japanese war mask depicting a fierce tengu with a long nose", "elven circlet of silver vines, adorned with tiny glowing crystals", "dwarven horned helmet with runes carved into the steel", "dragon scale cloak that shimmers in the light, with a metallic sheen", "demon’s gauntlets with clawed fingers and smoldering red veins", "celestial shield made of stardust, glowing with a soft luminescence", "Celtic bronze breastplate engraved with knotwork and spirals", "Germanic iron bracers engraved with bear claw motifs", "Mycenaean bronze cuirass with embossed scenes of chariot warfare", "Hittite scale armor with a bronze lion’s head on the chest", "Babylonian winged helm inlaid with lapis lazuli and gold", "Japanese sode shoulder guards shaped like samurai crests", "elven light armor made of intertwined vines and enchanted wood", "dwarven greaves carved from thick steel, featuring hammer motifs", "dragonhide vest with scales that harden upon impact", "demon’s helm with glowing red eyes and curved horns", "divine gauntlets that radiate warmth and emit a soft golden light", "Celtic war mask made of bronze, depicting a snarling beast", "Germanic fur-lined boots with iron reinforcements at the toes", "Mycenaean bronze vambraces adorned with spiraling designs", "Hittite war boots with bronze plating and leather straps", "Babylonian gold-inlaid bracers depicting ancient celestial maps", "Japanese mempo face mask shaped like a snarling demon", "elven mithril bracers engraved with flowing elvish script", "dwarven belt with a buckle shaped like an anvil, heavily ornamented", "dragonbone helmet adorned with curling horns and sharp spikes", "demonic chestplate that appears to pulse with malevolent energy", "armor made of woven starlight, shifting and ethereal", "Celtic iron gauntlets with engraved knotwork and spiral designs", "Germanic bronze helmet with a boar crest and cheek guards", "Mycenaean bronze shield with a central gorgon face in relief", "Hittite ceremonial cuirass adorned with wings and sun discs", "Babylonian war crown with an eagle motif and lapis lazuli inlays", "Japanese lacquered armor with gold leafing and intricate patterns", "elven shoulder guards shaped like crescent moons, glowing softly", "dwarven iron shield with a raised hammer emblem and thick rivets", "dragon scale bracers that harden and crackle with energy", "demon's spiked boots that emit a faint sulfuric smoke", "divine helmet crowned with radiant beams of light", "Celtic bronze belt with interlocking spiral designs", "Germanic iron chainmail with fur-trimmed edges and wolf head clasps", "Mycenaean greaves with detailed scenes of warriors and chariots", "Hittite war crown adorned with bronze feathers and gemstones", "Babylonian cuirass inlaid with gold and depicting mythical beasts", "Japanese kabuto with antler-like horns and intricate carvings", "elven circlet made of twisted silver vines and glowing gems", "dwarven iron pauldrons shaped like mountain peaks", "dragonbone chestplate that shifts in hue from green to gold", "demon's gauntlets with clawed fingers and pulsing red veins", "divine cuirass with glowing runes and a radiant aura", "Celtic war helmet with a horsehair crest and spiral engravings", "Germanic bronze greaves with embossed animal motifs", "Mycenaean bronze helmet with a crest of horsehair and embossed designs", "Hittite scale armor adorned with intricate wing patterns", "Babylonian war mask with a snarling lion’s face and gold accents", "Japanese armor plated with lacquered wood and gold leaf", "elven silver breastplate shaped like a tree with spreading branches", "dwarven gauntlets with heavy iron plates and glowing runes", "dragonhide boots with talon-shaped toes and a metallic sheen", "demon’s helmet with a grimacing face and curved, black horns", "divine shield radiating light and engraved with celestial patterns"]

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