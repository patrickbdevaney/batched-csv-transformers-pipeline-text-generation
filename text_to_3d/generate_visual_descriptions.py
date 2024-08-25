import pandas as pd
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import gc
import re
import random
import os

# Initialize the text generation pipeline with 16-bit precision
print("Initializing the text generation pipeline with 16-bit precision...")
model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
print("Text generation pipeline initialized with 16-bit precision.")

# Seed words pool
seed_words = ["A weathered iron brooch shaped like a raven, its wings etched with intricate knotwork",
"A towering oak tree, its roots twisted around ancient stone idols buried deep in the earth",
"A bronze shield engraved with the image of a coiled serpent, its eyes set with garnets",
"A horned helm forged from dark iron, adorned with a single hawk feather",
"A moss-covered stone circle, each monolith carved with mysterious spiral patterns",
"A silver torc inlaid with emeralds, worn by the chieftain of a powerful clan",
"A massive war hammer, its head shaped like a boar’s skull, with runes of strength along the handle",
"A spectral hound with eyes that glow like burning coals, its fur black as night",
"A pair of leather boots, reinforced with bronze plates, crafted for the swiftest warrior",
"A cauldron of shimmering bronze, said to brew potions that grant visions of the future",
"A deer with antlers made of living wood, leaves sprouting from its branches",
"A longbow carved from yew wood, the bowstring glowing faintly with enchantment",
"A stone altar stained with the blood of ancient sacrifices, surrounded by withered flowers",
"A cloak of wolf fur, fastened with a silver brooch shaped like a wolf’s head",
"A tall figure clad in green, their face obscured by a hood, holding a staff entwined with ivy",
"A sword with a hilt wrapped in leather, the blade engraved with runes that glow in moonlight",
"A gold coin minted with the image of a long-forgotten king, found buried in an ancient barrow",
"A ring made of intertwined gold and silver bands, said to bind the wearer’s fate with another’s",
"A spectral warrior clad in ancient armor, forever bound to guard the entrance to a sacred grove",
"A deer antler carved into a hunting horn, used to summon allies from the spirit world",
"A silver chalice, its rim encrusted with tiny rubies, used in druidic rituals",
"A massive stone fortress perched on a hill, surrounded by a mist that never lifts",
"A crow with feathers as black as night, often seen as a harbinger of doom",
"A bronze spear tipped with a jagged flint blade, used in ritual hunts",
"A tree hollow filled with glowing mushrooms that emit a soft, otherworldly light",
"A ghostly figure dressed in tattered robes, holding an ancient staff of twisted wood",
"A stag with golden antlers, revered as a symbol of the gods’ favor",
"A wolf pelt cloak, lined with the fur of a dire wolf, worn by the clan’s greatest hunter",
"A harp crafted from a single piece of oak, its strings humming with ancient melodies",
"A stone cairn marked with runes that tell the tale of a hero’s final battle",
"A golden torc, heavy and ornately decorated, signifying the wearer’s high status",
"A glowing crystal embedded in the forehead of a druidic staff, pulsing with arcane power",
"A blackened iron cauldron, used to brew potions that grant strength in battle",
"A raven with eyes that glow a piercing blue, messenger of the otherworld",
"A stone dagger, its blade chipped and worn, yet still sharp and deadly",
"A large bear with fur as white as snow, a guardian spirit of the northern tribes",
"A cloak woven from the leaves of an ancient oak, shimmering with protective magic",
"A helm adorned with the horns of a stag, worn by the leader of the hunt",
"A stone amulet carved with protective runes, worn by warriors into battle",
"A spear tipped with a blade made of pure obsidian, glinting with a deadly sharpness",
"A silver flute, its notes said to charm the animals of the forest",
"A glowing wisp of light that leads travelers to hidden places of power",
"A druidic circle where the very air hums with ancient power, stones aligned with the stars",
"A worn leather-bound book filled with ancient Celtic spells and incantations",
"A wolf with fur that blends perfectly with the shadows, stalking the forests at night",
"A sword forged in the heart of a volcano, its blade forever burning with an inner fire",
"A crown of oak leaves and mistletoe, worn by the high druid during sacred ceremonies",
"A ghostly stag that appears only under the light of a full moon, leading lost souls to the afterlife",
"A stone cairn at the top of a windswept hill, marking the grave of a forgotten hero",
"A stone tablet inscribed with runes that tell the history of an ancient tribe",
"A cloak of raven feathers, granting the wearer the ability to disappear into the night",
"A spear with a shaft made from ash wood, its tip engraved with protective runes",
"A bronze mirror that shows not only the reflection but the true nature of the person gazing into it",
"A massive black bear with eyes that burn with the fire of an ancient spirit",
"A golden goblet that never empties, always filled with the finest mead",
"A wolf’s head carved into a stone pillar, symbolizing the power of the clan it protects",
"A long, twisted staff of rowan wood, said to protect against evil spirits",
"A ghostly hand reaching out from a pool of still water, beckoning travelers to their doom",
"A stag’s head mounted on a wall, its antlers hung with trophies from countless hunts",
"A silver dagger that hums with magic, used in rituals to bind and protect",
"A stone circle on a hilltop, each stone engraved with runes of protection",
"A bow crafted from the yew tree, its string made from the hair of a maiden sacrificed to the gods",
"A raven perched on the shoulder of a druid, its eyes watching and knowing all",
"A horn made from the tusk of a great boar, used to call warriors to battle",
"A silver bracelet shaped like a serpent, its eyes set with tiny emeralds",
"A dark forest where the trees whisper in an ancient tongue, and shadows move on their own",
"A stag with fur that shines like silver in the moonlight, leading a herd of spectral deer",
"A cauldron bubbling with a potion that grants visions of the past and future",
"A golden torc that tightens around the wearer’s neck, binding them to a powerful curse",
"A stone throne carved with symbols of the gods, set deep within a sacred grove",
"A ghostly raven that appears on the battlefield, signaling the arrival of death",
"A pair of boots made from the skin of a wolf, said to give the wearer the speed of the beast",
"A tree with bark as white as bone, its roots said to reach into the underworld",
"A silver chalice that glows faintly, used in rituals to commune with the gods",
"A druidic staff made from a single branch of an ancient oak, its tip crowned with mistletoe",
"A ring made of intertwined branches, enchanted to protect the wearer from harm",
"A stone circle where the spirits of the dead gather to dance under the full moon",
"A spear that hums with power, its tip forged from the heart of a fallen star",
"A wolf with fur the color of ash, its eyes glowing with an eerie light",
"A sword with a blade that gleams like silver, etched with runes of protection",
"A raven’s feather dipped in ink, used to write spells of binding and control",
"A golden belt adorned with symbols of the sun, worn by the high priestess",
"A stone basin filled with water that reflects the future, used in druidic rituals",
"A cloak made from the hide of a great bear, granting the wearer strength and protection",
"A spear with a shaft made of rowan wood, tipped with a blade forged in dragon fire",
"A ghostly figure that walks the forest paths at night, guiding lost travelers to safety",
"A ring of stones on a hilltop, where the ancient gods are said to have walked",
"A blacksmith’s hammer that glows with heat, used to forge weapons of great power",
"A silver necklace with a pendant shaped like a crescent moon, symbolizing the goddess",
"A stone tower overlooking the sea, where the winds whisper secrets of the deep",
"A raven with a single white feather, known as a messenger of the gods",
"A cauldron that never empties, always bubbling with a potion of healing",
"A spear tipped with a blade made of pure crystal, glowing with an inner light"]
used_words = set()

# Function to generate a detailed visual description prompt
def generate_description_prompt(subject):
    prompt = f"Generate a detailed visual description of something in a fantasy RPG setting different from {subject}."
    generated_text = text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
    return generated_description

# Function to generate text and write to CSV
def generate_and_write_to_csv(output_csv):
    descriptions = []

    # Check if the CSV file already exists
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        descriptions = existing_df.to_dict('records')

    # Select a subject that has not been used
    available_subjects = [word for word in seed_words if word not in used_words]
    if not available_subjects:
        print("No more available subjects to use.")
        return

    subject = random.choice(available_subjects)
    generated_description = generate_description_prompt(subject)
    descriptions.append({'subject': subject, 'description': generated_description})

    # Update used words and seed words
    used_words.add(subject)
    seed_words.append(generated_description)  # Add the generated description to the seed bank array

    # Convert the result to a pandas DataFrame and save to CSV
    result_df = pd.DataFrame(descriptions)
    result_df.to_csv(output_csv, index=False)
    print(f"All generated descriptions written to {output_csv}")

    # Save the generated texts to a .txt file with utf-8 encoding
    with open(output_csv.replace('.csv', '.txt'), 'w', encoding='utf-8') as txt_file:
        for description in descriptions:
            txt_file.write(description['description'] + '\n')
    print(f"All generated descriptions written to {output_csv.replace('.csv', '.txt')}")

# Run the function
generate_and_write_to_csv('descriptions.csv')

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
print("GPU memory cleared.")