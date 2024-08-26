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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 100 words of intricate varied creatures from various ancient bronze age europe mythological fantasy RPG settings different from {subject}. each example very different alternating between fae undead elves sprites gods ghosts demons and others, colorful description. The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=230, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["A majestic unicorn with a shimmering white coat and a spiraling golden horn. Its mane flows like liquid silver, and its eyes sparkle with ancient wisdom.",
"A fierce Celtic dragon with emerald scales and fiery breath. Its wings are vast and leathery, and its roar can be heard for miles.",
"An erudite elf with long, flowing hair and piercing blue eyes. Clad in elegant robes, they exude an aura of wisdom and authority.",
"A graceful faerie with delicate wings that shimmer in the moonlight. Their laughter is like the tinkling of bells, and they leave a trail of sparkling dust in their wake.",
"A powerful Etruscan chimera with the body of a lion, the head of a goat, and the tail of a serpent. Its breath is poisonous, and its roar is terrifying.",
"A wise Sumerian sphinx with the body of a lion and the head of a human. It guards ancient secrets and poses riddles to those who seek its knowledge.",
"A noble centaur with the upper body of a human and the lower body of a horse. They are skilled archers and fierce warriors.",
"A mischievous leprechaun with a green coat and a tall hat. They are known for their trickery and their hidden pots of gold.",
"A majestic griffin with the body of a lion and the wings and head of an eagle. Its talons are sharp, and its beak is powerful.",
"An imperious elf queen with a crown of silver and a scepter of crystal. She rules her kingdom with grace and wisdom.",
"A fearsome banshee with long, flowing hair and a mournful wail. Her appearance foretells death and disaster.",
"A cunning kelpie with the ability to transform into a beautiful horse. It lures travelers to their doom in the depths of the water.",
"A wise druid with a staff of oak and a cloak of leaves. They have the ability to communicate with animals and control the forces of nature.",
"A majestic phoenix with fiery plumage and the ability to be reborn from its ashes. Its song is hauntingly beautiful.",
"A powerful minotaur with the body of a man and the head of a bull. It wields a massive axe and guards labyrinthine passages.",
"A graceful dryad with skin like bark and hair like leaves. They are the guardians of the forests and can blend seamlessly with the trees.",
"A fierce werewolf with sharp claws and glowing eyes. They transform under the full moon and are feared by all.",
"A noble elf prince with a bow of silver and arrows of light. He is a skilled hunter and a wise leader.",
"A mischievous sprite with wings like a dragonfly and a playful demeanor. They are known for their pranks and their love of music.",
"A powerful titan with the strength of a hundred men and the ability to control the elements. They are ancient beings of immense power.",
"A wise oracle with the ability to see the future and communicate with the gods. They are sought after for their guidance and wisdom.",
"A majestic pegasus with wings of pure white and the ability to fly. Its hooves leave a trail of stardust in the sky.",
"A fierce manticore with the body of a lion, the wings of a bat, and the tail of a scorpion. Its venom is deadly, and its roar is terrifying.",
"An erudite elf scholar with a library of ancient tomes and a mind full of knowledge. They are respected for their wisdom and their dedication to learning.",
"A graceful nymph with the ability to control water and communicate with aquatic creatures. They are the guardians of rivers and lakes.",
"A powerful giant with the strength to move mountains and the ability to control the weather. They are feared and respected by all.",
"A wise sage with a long beard and a staff of crystal. They have the ability to see into the past and the future.",
"A majestic dragon with scales of gold and the ability to breathe fire. Its eyes are like molten lava, and its roar shakes the earth.",
"A fierce harpy with the body of a bird and the head of a woman. Its talons are sharp, and its screech is deafening.",
"An imperious elf king with a crown of gold and a sword of light. He rules his kingdom with strength and wisdom.",
"A mischievous brownie with a love of mischief and a talent for hiding. They are known for their pranks and their love of sweets.",
"A powerful hydra with multiple heads and the ability to regenerate. Its breath is poisonous, and its bite is deadly.",
"A graceful sylph with wings like a butterfly and the ability to control the wind. They are the guardians of the air and can create gentle breezes or powerful storms.",
"A fierce basilisk with the ability to turn creatures to stone with its gaze. Its scales are like armor, and its bite is venomous.",
"A wise seer with the ability to communicate with the spirits and see into the future. They are sought after for their guidance and their ability to commune with the divine.",
"A majestic wyvern with wings of leather and the ability to breathe fire. Its tail is barbed, and its roar is terrifying.",
"A fierce gorgon with snakes for hair and the ability to turn creatures to stone with her gaze. Her appearance is fearsome, and her power is deadly.",
"An erudite elf mage with a staff of crystal and the ability to control the elements. They are respected for their wisdom and their mastery of magic.",
"A graceful mermaid with a tail of shimmering scales and the ability to control the tides. They are the guardians of the ocean and can communicate with sea creatures.",
"A powerful cyclops with a single eye and the strength of a hundred men. They are feared for their power and their ability to crush their enemies.",
"A wise druidess with the ability to communicate with animals and control the forces of nature. She is respected for her wisdom and her connection to the natural world.",
"A majestic roc with wings of gold and the ability to carry off elephants. Its talons are sharp, and its beak is powerful.",
"A fierce chimera with the body of a lion, the head of a goat, and the tail of a serpent. Its breath is poisonous, and its roar is terrifying.",
"An imperious elf lord with a crown of silver and a scepter of crystal. He rules his kingdom with grace and wisdom.",
"A mischievous pixie with wings like a dragonfly and a playful demeanor. They are known for their pranks and their love of music.",
"A powerful titaness with the strength of a hundred men and the ability to control the elements. She is an ancient being of immense power.",
"A wise oracle with the ability to see the future and communicate with the gods. She is sought after for her guidance and wisdom.",
"A majestic hippogriff with the body of a horse and the wings and head of an eagle. Its talons are sharp, and its beak is powerful.",
"A fierce manticore with the body of a lion, the wings of a bat, and the tail of a scorpion. Its venom is deadly, and its roar is terrifying.",
"An erudite elf sage with a library of ancient tomes and a mind full of knowledge. They are respected for their wisdom and their dedication to learning.",
"A graceful naiad with the ability to control water and communicate with aquatic creatures. They are the guardians of rivers and lakes.",
"A powerful giantess with the strength to move mountains and the ability to control the weather. She is feared and respected by all.",
"A wise sage with a long beard and a staff of crystal. He has the ability to see into the past and the future.",
"A majestic dragon with scales of silver and the ability to breathe fire. Its eyes are like molten lava, and its roar shakes the earth.",
"A fierce harpy with the body of a bird and the head of a woman. Its talons are sharp, and its screech is deafening.",
"An imperious elf queen with a crown of gold and a sword of light. She rules her kingdom with strength and wisdom.",
"A mischievous brownie with a love of mischief and a talent for hiding. They are known for their pranks and their love of sweets.",
"A powerful hydra with multiple heads and the ability to regenerate. Its breath is poisonous, and its bite is deadly.",
"A graceful sylph with wings like a butterfly and the ability to control the wind. They are the guardians of the air and can create gentle breezes or powerful storms.",
"A fierce basilisk with the ability to turn creatures to stone with its gaze. Its scales are like armor, and its bite is venomous.",
"A wise seer with the ability to communicate with the spirits and see into the future. They are sought after for their guidance and their ability to commune with the divine.",
"A majestic wyvern with wings of leather and the ability to breathe fire. Its tail is barbed, and its roar is terrifying.",
"A fierce gorgon with snakes for hair and the ability to turn creatures to stone with her gaze. Her appearance is fearsome, and her power is deadly.",
"An erudite elf mage with a staff of crystal and the ability to control the elements. They are respected for their wisdom and their mastery of magic.",
"A graceful mermaid with a tail of shimmering scales and the ability to control the tides. They are the guardians of the ocean and can communicate with sea creatures.",
"A powerful cyclops with a single eye and the strength of a hundred men. They are feared for their power and their ability to crush their enemies.",
"A wise druidess with the ability to communicate with animals and control the forces of nature. She is respected for her wisdom and her connection to the natural world."
]

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