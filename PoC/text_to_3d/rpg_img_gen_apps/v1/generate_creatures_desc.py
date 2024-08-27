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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 50 words of varied creatures like animals ie bear mountian lion, mythical creatures, demons, angels, undead like lich, mummy, zombie, vampire, sea serpent, metallic warriors from various ancient bronze age europe middle east far east mythological fantasy RPG settings different from {subject}. each example very different style/influence (e.g. fantasy elf, demon, god, atlantean, dragon, dwarf). The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["massive bronze bear with fur that glows in the sunlight", "savage mountain lion with golden eyes and retractable claws", "majestic ram with curling horns and silver-tipped fleece", "ancient sea serpent with scales like polished emeralds and eyes like sapphires", "shadowy cryptid with elongated limbs and eyes that pierce the darkness", "elven deer with silver antlers and glowing hooves", "golden gilded metallic elf with a body that gleams like polished armor", "fearsome minotaur with bull horns and a towering muscular frame", "sinuous dragon with iridescent scales and fiery breath", "phantom wolf with glowing red eyes and a ghostly howl", "glittering phoenix with fiery plumage that regenerates when burned", "serpentine hydra with multiple heads and venomous fangs", "ancient mummy wrapped in decaying bandages and cursed with immortality", "spectral wraith that drifts through walls and drains the life from its victims", "armored skeletal warrior with rusted weapons and hollow, burning eyes", "vampire lord with pale skin, elongated fangs, and a thirst for blood", "hulking ogre with thick, stone-like skin and a bone club", "angelic being with glowing wings and a halo of pure light", "noble griffin with the body of a lion and the wings of an eagle", "shadowy lich with glowing blue eyes and a staff of dark magic", "fire-breathing chimera with the heads of a lion, goat, and dragon", "ancient kraken with tentacles the size of ships and eyes like deep abysses", "ethereal nymph with flowing hair and a body that shimmers like water", "undead zombie with rotting flesh and a relentless hunger for the living", "colossal cyclops with a single eye and the strength of a hundred men", "golden pegasus with metallic wings that sparkle in the sunlight", "fiery salamander that lives in molten lava and breathes flames", "metallic dragonfly with iridescent wings and a crystalline body", "heavenly cherub with tiny wings and a golden bow", "majestic unicorn with a spiraled horn of pure ivory", "ferocious werewolf with sharp claws and fur that bristles with each breath", "metallic centaur with a gleaming bronze body and a powerful bow", "shadowy banshee with a haunting wail that freezes the blood", "golden-armored angel with a flaming sword and radiant wings", "mighty thunderbird with wings that crackle with lightning", "ancient basilisk with a gaze that turns creatures to stone", "slithering naga with the body of a serpent and the upper torso of a human", "vengeful wraith with tattered robes and a cold, deathly aura", "massive direwolf with fur as black as midnight and eyes that glow red", "serene water sprite with translucent skin and hair like flowing water", "golden metallic lion with a mane that gleams like the sun", "fearsome gorgon with writhing snakes for hair and a petrifying gaze", "glowing will-o’-the-wisp that dances above the marshes, luring travelers to their doom", "ancient stone golem with runes carved into its massive body", "shimmering fairy with delicate wings and a mischievous smile", "gruesome ghoul with decayed flesh and a ravenous appetite for corpses", "shadowy djinn with swirling smoke for a lower body and glowing eyes", "armored dragon turtle with a shell like a fortress and a fiery breath", "metallic serpent with golden scales that shimmer in the light", "glowing celestial dragon with scales like the night sky", "monstrous sea leviathan with tentacles that stretch for miles", "bronze-armored mummy with eyes that glow with cursed fire", "fire-breathing wyvern with leathery wings and a venomous sting", "ancient sphinx with the body of a lion and the head of a wise human", "glowing spirit bear with ethereal fur and piercing blue eyes", "vengeful specter with a tattered cloak and chains that rattle in the night", "majestic hippocampus with the body of a horse and the tail of a fish", "gilded skeletal warrior with bones that shine like gold", "ancient earth elemental with a body made of rock and soil", "fierce harpy with sharp talons and wings that beat like thunder", "golden-armored golem with joints that creak and grind", "enormous roc with wings that block out the sun and talons like swords", "savage dire bear with a thick, iron-like hide and a roar that shakes the mountains", "ethereal angel of death with wings of shadow and a scythe of pure darkness", "burning efreet with skin like molten lava and a heart of fire", "shadowy wendigo with elongated limbs and a hunger for flesh", "glistening mermaid with scales that shimmer like pearls and a voice that enchants sailors", "metallic gryphon with talons of steel and wings that sparkle in the sun", "ancient tree spirit with bark-like skin and roots that stretch deep into the earth", "golden ghost with a spectral form that glows with a faint, eerie light", "fiery phoenix with feathers that burn with the intensity of a thousand suns", "shadowy vampire bat with leathery wings and fangs that drip with blood", "enormous sandworm with a segmented body and a mouth lined with rows of sharp teeth", "golden-armored sphinx with wings that glimmer in the sun", "shadowy demon with bat-like wings and eyes that burn with an inner fire", "glowing celestial lion with a mane of starlight and a roar that echoes across the heavens", "bronze griffin with a beak like a sword and claws like iron", "monstrous behemoth with a body like a mountain and a roar that shakes the earth", "fiery salamander with scales that glow like embers and a tongue of flame", "shadowy reaper with a scythe of darkness and a cloak of shadows", "golden metallic centaur with a bow of pure light and arrows that shine like stars", "ethereal sylph with wings of mist and a body that floats on the breeze", "undead lich with a skeletal frame and eyes that burn with dark magic", "gleaming golden dragon with scales that shine like the sun", "ancient kraken with tentacles that stretch to the horizon and eyes like the abyss", "glowing moonlit hare with fur that shimmers like silver in the night", "fiery hellhound with red-hot claws and eyes that burn with infernal fire", "shadowy banshee with a face twisted in eternal agony and a voice that chills the soul", "golden golem with joints that creak and eyes that glow with a soft light", "fearsome werebear with fur like steel and claws like knives", "celestial stag with antlers that glow like the moon and hooves that leave trails of light", "shadowy hellhound with glowing red eyes and a snarl that echoes in the darkness", "glittering metallic mermaid with a tail that shimmers like gold and a voice that mesmerizes sailors"]

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