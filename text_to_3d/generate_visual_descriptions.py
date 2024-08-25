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
    prompt = f"Generate a vivid visual description (less than 35 words) of a single object, landform, building, creature, plant, or possession in an ancient celt, roman, germanic, babylon, egypt, and fantasy RPG setting different from {subject}. Be unique and make each example exotically different from the previous, alternating between object, animal, building, plant, mythical creature, and others. Enclose each description in brackets like [ <description> ]. The described item should be a distinct object easy for an AI to 3D model from an image of it. No comments. always finish description"
    try:
        generated_text = text_generator(prompt, max_length=190, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["glimmering stone pillar carved with intricate runes, bathed in soft moonlight", "ancient oak tree with massive roots coiled around a weathered silver sword", "large, golden dragon's egg resting on sharp, black obsidian shards", "shallow crystal-clear pool with a mirror-like surface reflecting a starry night sky", "brooch shaped like a phoenix, with detailed feathers and a glowing amber core", "round talisman made of polished jade, softly pulsing with green light", "stone idol of a god with glowing eyes, worn and cracked by time", "ornate silver harp, its strings made of delicate threads that shimmer", "tall crystal staff topped with a jagged shard of ice that glows faintly", "shadowy nymph with delicate features, darting between dense, dark trees", "round shield made of dark metal, covered in glowing runes and etched patterns", "large sunken temple with broken columns, surrounded by ghostly, glowing fish", "serpent made of gold coiled tightly around a large, shiny apple", "floating island with jagged edges, perpetually shrouded in twilight mist", "smooth, round pearl orb with swirling, cloudy patterns inside", "elegant silver chalice filled with a liquid that sparkles and shifts like starlight", "vortex of churning water suspended in the air, constantly rotating", "tall standing stone with lightning-like cracks glowing from within", "ornate golden helmet adorned with large, colorful feathers", "floating, glowing candle that casts no shadow and never melts", "emerald-encrusted amulet with a realistic dragon's eye at its center", "vibrant coral garden, glowing softly in a dark underwater cavern", "thick, ancient tome bound in rough dragonhide, pages glowing with a faint blue light", "celestial sphere made of metal, rotating slowly with engraved star maps", "crescent moonstone pendant that emits a faint, cold light", "ivy-covered ruins of an ancient city with crumbling walls and tall spires", "bronze gate with a glowing inscription, sealed tightly by magic", "grove of tall, ancient trees where time appears frozen, leaves unmoving", "polished mirror with a smooth, reflective surface that shows shifting images", "obsidian dagger with a sharp, curved blade, glowing with red energy", "elegant quill made of gold, writing on its own in mid-air with glowing ink", "flat stone altar surrounded by flickering blue ghostly flames", "crystal skull with hollow eyes, a faint light flickering within", "ancient manuscript with tattered edges, soaked and decayed, yet glowing faintly", "hollow tree with a dark entrance leading to a maze of roots below", "arched bridge made of delicate, translucent moonbeams", "jeweled goblet with intricate engravings, glowing faintly as it fills with liquid", "golden crown adorned with large, glowing rubies, resting on a velvet cushion", "thin silver ring with an inscription that glows under moonlight", "frozen waterfall with ice that shimmers and emits a soft, musical tone", "ornate lantern with a flickering, eternal flame inside", "small silver key with ornate designs, glowing before it vanishes", "thin, shimmering veil suspended in the air, separating two different realms", "secluded forest grove with leaves that rustle with whispered voices", "stone archway with glowing runes, standing alone in a clearing", "ember of an eternal fire, glowing brightly and never cooling", "waterfall that defies gravity, flowing upwards into the sky", "ancient tree with metallic golden leaves that softly shimmer", "giant mushroom with a broad cap, glowing faintly with an eerie light", "ghostly ship with tattered sails, floating on a sea of dense mist", "treasure chest covered in barnacles, guarded by a large, coiled sea serpent", "comet with a fiery red tail, streaking across a dark sky", "stone altar under the moonlight, dedicated to an ancient, forgotten goddess", "large bronze bell, tarnished and cracked, tolling on its own", "glowing map with intricate details, charting the stars and celestial paths", "moss-covered stone face with water trickling continuously from its mouth", "hourglass with shimmering sand, flowing endlessly without running out", "ring of tall standing stones, each one glowing with a different color", "silver arrow with a sharp, pointed tip, glowing with a faint blue light", "crystal ball on a pedestal, clouded with swirling mist inside", "ancient tree with twisted branches, glowing with the light of trapped souls", "stone portal with a glowing center, hidden deep within a cave", "floating crystal, slowly rotating and humming with a soft vibration", "gilded cage holding a small, flickering flame", "large, ice-covered heart that pulses slowly with light", "shimmering fabric stretched across a frame, resembling the night sky", "golden apple with a smooth surface, glowing faintly in the dark", "large anvil made of dark metal, surrounded by the glow of indestructible weapons", "flat stone tablet with glowing, engraved words", "golden harp with fine strings, glowing softly as it plays itself", "forest of petrified trees, each one frozen in place and glowing faintly", "ancient horn made of bone, glowing faintly as it echoes", "shifting sands that glimmer under the sunlight, hiding the ruins of lost cities", "dark figure cloaked in mist, barely visible within an enchanted forest", "glowing scales of a water dragon, shimmering beneath a tranquil lake", "jeweled scarab with a detailed carapace, glowing with ancient magic", "quiver made of dark leather, glowing as it never runs out of arrows", "feather of a phoenix, glowing with eternal fire and warmth", "sapphire pendant with a glowing center, pulsing like ocean waves", "bronze shield with a raised emblem, glowing faintly with ancient power", "floating rune, glowing softly and hovering just above the ground", "curved blade covered in frost, emitting a cold, sharp glow", "heavy stone door with intricate carvings, glowing with mystical energy", "water sprite with delicate wings, dancing above the surface of a pond", "orb of pure, crackling energy, glowing and floating in mid-air", "sword made of light, glowing brightly and cutting through darkness", "clear spring with water that glows, showing visions of the future", "ethereal butterfly with glowing wings, guiding travelers through darkness", "garden with flowers that glow softly, singing in the wind", "dark grove filled with glowing orbs, where spirits of the forest dwell", "large stone throne carved from bone, glowing with the essence of a leviathan", "glowing sigil engraved on a stone wall, marking the entrance to a lost city", "dark red cloak that glows faintly and renders the wearer invisible in shadows", "shimmering waterfall made of liquid light, glowing as it flows", "ornate crown with glowing runes, bestowing wisdom to the wearer", "constellation of stars frozen in the sky, glowing brightly against the darkness"
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