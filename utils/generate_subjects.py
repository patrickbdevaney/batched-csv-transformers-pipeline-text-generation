import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
import random

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")

# Seed words pool
seed_words = ["Elf", "Dragon", "Undead", "Lich", "Mage", "Knight", "Siren", "Dwarf", "God", "Demon", "Mummy", "Tomb", "Tavern", "Wizard", "Orc", "Paladin", "Necromancer", "Golem", "Goblin", "Vampire", "Ranger", "Thief", "Sorcerer", "Witch", "Barbarian", "Priest", "Cleric", "Warrior", "Ghoul", "Banshee", "Phoenix", "Fairy", "Troll", "Minotaur", "Bard", "Warlock", "Giant", "Werewolf", "Warlord", "Enchanter", "Druid", "Harpy", "Assassin", "Shaman", "Zombie", "Chimera", "Gryphon", "Djinn", "Sphinx", "Titan", "Gorgon", "Sprite", "Succubus", "Imp", "Nymph", "Dryad", "Lamia", "Naga", "Mermaid", "Elemental", "Yeti", "Centaur", "Cyclops", "Hobbit", "Hag", "Revenant", "Behemoth", "Gargoyle", "Wyvern", "Kraken", "Djinni", "Wraith", "Manticore", "Bugbear", "Kobold", "Ogre", "Satyr", "Valkyrie", "Shade", "Homunculus", "Gremlin", "Roc", "Kelpie", "Salamander", "Wyrm", "Basilisk", "Hydra", "Undine", "Efreet", "Seraph", "Tarrasque", "Rakshasa", "Incubus", "Chimera", "Sylph", "Drake", "Ifrit", "Jotun"]
used_words = set()

# Function to generate a word related to the subject
def generate_word(subject):
    prompt = f"List a word related to {subject} and do not repeat words."
    generated_text = text_generator(prompt, max_length=5, num_return_sequences=1)[0]['generated_text']
    generated_word = generated_text.split()[-1]  # Get the last word from the generated text
    return generated_word

# Function to generate text and write to CSV
def generate_and_write_to_csv(output_csv, num_iterations=10):
    dialogues = []
    for _ in range(num_iterations):
        # Select a subject that has not been used
        available_subjects = [word for word in seed_words if word not in used_words]
        if not available_subjects:
            print("No more available subjects to use.")
            break

        subject = random.choice(available_subjects)
        generated_word = generate_word(subject)
        dialogues.append({'subject': generated_word})

        # Update used words and seed words
        used_words.add(subject)
        used_words.add(generated_word)  # Ensure the generated word is not reused
        seed_words.append(generated_word)

    # Convert the result to a pandas DataFrame and save to CSV
    result_df = pd.DataFrame(dialogues)
    result_df.to_csv(output_csv, index=False)
    print(f"All generated words written to {output_csv}")

    # Save the generated texts to a .txt file with utf-8 encoding
    with open(output_csv.replace('.csv', '.txt'), 'w', encoding='utf-8') as txt_file:
        for dialogue in dialogues:
            txt_file.write(dialogue['subject'] + '\\n')
    print(f"All generated words written to {output_csv.replace('.csv', '.txt')}")

# Run the function
generate_and_write_to_csv('subjects.csv')

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared.")
