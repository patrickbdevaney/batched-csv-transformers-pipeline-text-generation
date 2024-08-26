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
    prompt = f"write concise vivid visual description enclosed in brackets like [ <description> ] less than 50 words of varied buildings and architectural elements like column capital, archway, atrium, roundhouse, arcade, stoa, portcullis, castle ancient bronze age europe middle east far east mythological fantasy RPG settings different from {subject}. each example very different style/influence. The description is physically distinct easy for an AI to 3D model from an image of it. No comments, always finish the description."
    try:
        generated_text = text_generator(prompt, max_length=200, num_return_sequences=1, truncation=True)[0]['generated_text']
        generated_description = re.sub(rf'{re.escape(prompt)}\s*', '', generated_text).strip()  # Remove the prompt from the generated text
        return generated_description if generated_description else None
    except Exception as e:
        print(f"Error generating description for subject '{subject}': {e}")
        return None


# Seed words pool
seed_words = ["ornate elven archway with intricate leaf motifs and glowing runes", "towering Celtic round tower with a conical roof and stone walls", "massive stone ziggurat with tiered levels and a wide, ceremonial staircase", "ancient Roman stoa with tall columns and a marble floor", "Minoan palace with vibrant frescoes and labyrinthine corridors", "Mycenaean palace with massive stone walls and a central megaron", "Egyptian mausoleum with a towering obelisk and golden hieroglyphs", "crusader castle with high battlements and a deep, dry moat", "post and lintel Irish passage grave with a narrow entrance and carved stones", "Lions Gate of Hattusa with twin lion statues flanking the entrance", "grand Venetian portcullis with iron bars and ornate carvings", "Norman castle in Ireland with round towers and a stone curtain wall", "towering gothic cathedral with flying buttresses and stained glass windows", "massive Egyptian pyramid with smooth, golden limestone blocks", "elaborate Byzantine basilica with a domed roof and mosaic-covered walls", "ancient Sumerian temple with stepped platforms and ziggurat-like structure", "stone Dolmen with massive capstones and standing stones", "Minoan column with a tapering shaft and a vibrant red color", "fortified Viking longhouse with thick wooden walls and a thatched roof", "ornate Japanese torii gate with curved roofs and red-lacquered wood", "grand Elven palace with silver spires and crystal windows", "Babylonian Ishtar Gate with blue-glazed bricks and golden animal reliefs", "Gothic rose window with intricate tracery and vibrant stained glass", "ancient Greek amphitheater with tiered stone seats and a central stage", "massive Sumerian ziggurat with terraced levels and a temple at the summit", "Roman aqueduct with arched spans and towering piers", "ornate Celtic cross with interwoven knotwork and standing on a stone base", "crumbling Norman keep with thick walls and arrow slits", "grand Persian palace with tall columns and colorful tilework", "fortified medieval gatehouse with a heavy wooden door and portcullis", "ancient Egyptian temple with towering pylons and massive statues", "elaborate Venetian bridge with arched stone spans and intricate carvings", "Byzantine Hagia Sophia with a massive dome and golden mosaics", "ancient Greek temple with fluted Doric columns and a sculpted pediment", "grand Roman colosseum with tiered seating and towering arches", "ornate Gothic spire with intricate stone carvings and flying buttresses", "elaborate Celtic hillfort with earthen ramparts and timber palisades", "massive Mycenaean tholos tomb with a beehive-shaped interior and stone lintel", "stonehenge-like megalithic circle with massive standing stones", "ancient Minoan labyrinth with twisting corridors and hidden chambers", "towering Babylonian ziggurat with stepped terraces and a temple at the summit", "elaborate Persian garden with geometric patterns and flowing water channels", "ornate Egyptian pylon with massive statues and carved reliefs", "grand Byzantine cathedral with a domed roof and vibrant mosaics", "fortified medieval tower with crenellated battlements and arrow slits", "ancient Roman triumphal arch with sculpted reliefs and towering pillars", "massive Hittite city gate with carved stone lions and a broad archway", "ornate Japanese pagoda with tiered roofs and wooden eaves", "crumbling Roman aqueduct with arched spans and moss-covered stones", "grand Elven citadel with soaring towers and delicate bridges", "stone Celtic broch with thick walls and a spiral staircase", "fortified Crusader fortress with thick walls and a commanding view", "ancient Egyptian obelisk with hieroglyphs and a pointed tip", "elaborate Persian apadana with tall columns and a roof of cedar wood", "grand Gothic cathedral with pointed arches and soaring spires", "ornate Venetian palazzo with arched windows and a central courtyard", "massive Viking longhall with wooden beams and a thatched roof", "ancient Greek acropolis with temples and statues on a rocky outcrop", "towering Norman castle keep with thick stone walls and narrow windows", "ornate Celtic roundhouse with wattle and daub walls and a thatched roof", "grand Babylonian palace with glazed brick walls and hanging gardens", "ancient Roman basilica with tall columns and a coffered ceiling", "elaborate Egyptian hypostyle hall with massive columns and a stone roof", "fortified medieval manor with a moat and a stone curtain wall", "grand Persian caravanserai with a central courtyard and arched galleries", "ornate Byzantine monastery with a domed church and cloistered gardens", "massive Mycenaean fortress with cyclopean walls and a megaron", "ancient Sumerian city walls with mudbrick ramparts and fortified gates", "elaborate Minoan throne room with frescoed walls and a stepped platform", "towering Gothic bell tower with a pointed spire and intricate stonework", "grand Roman forum with temples, statues, and a central plaza", "ornate Venetian campanile with a tall, slender tower and arched openings", "massive Egyptian temple complex with pylons, obelisks, and sacred lakes", "ancient Celtic burial mound with a stone-lined passage and a corbelled roof", "elaborate Hittite palace with columned halls and intricately carved reliefs", "fortified Norman abbey with a high stone wall and a central cloister", "grand Byzantine aqueduct with arched spans and stone piers", "ornate Elven treehouse with wooden platforms and hanging bridges", "massive Viking ship burial mound with a stone chamber and earthen covering", "ancient Greek gymnasium with columned porticos and a central courtyard", "towering Babylonian ziggurat with a spiraling staircase and a temple at the top", "elaborate Persian fire temple with a domed roof and a central altar", "grand Gothic monastery with cloistered gardens and arched corridors", "ornate Celtic crannog with wooden platforms and thatched huts on stilts", "massive Roman fortress with stone walls, watchtowers, and a central courtyard", "ancient Egyptian sun temple with obelisks, statues, and open courtyards", "elaborate Minoan lustral basin with stone steps and frescoed walls", "fortified medieval bridge with stone arches and a central tower", "grand Venetian palace with a marble façade and arched windows", "ornate Byzantine church with a domed roof and mosaic-covered walls"]

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