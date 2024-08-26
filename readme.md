# CSV Segmentation and Inference Pipeline for Dialogue Generation

This approach involves taking a CSV file named `subject` with a single column containing *k* rows and splitting it into segmented CSV files of 1000 rows each. These segmented CSV files are then serialized. The generation script traverses the directory, identifies the CSV files, and generates Python scripts to run an inference pipeline. Each script processes the rows one by one, passing each subject to a predefined prompt designed to generate questions and answers about the subject,  in this case assuming the context of playing a character in a fantasy role-playing game world.

Using models like **Mistral-7b** or  **LLaMA 3.1 8B Instruct**, you can reliably generate diverse and creative responses for all rows in the CSV column, even when repeating a word from the `subject` column.

## Customizing the Prompt

You can modify the prompt to fit your generation task and the semantic scope of the CSV subject dataset to suit your specific domain. 

### Example 1: Electrical Engineering Concepts

Consider a `subject` column in your input CSV with related electrical engineering concepts:

subject:
Ohm's Law
Kirchhoff's Current Law
Kirchhoff's Voltage Law
AC Current
DC Current
Quantum Computing


You might modify the prompt as follows:

```python
def generate_dialogue(subject):
    prompt = f"Explain {subject} in electrical engineering and the best hands-on ways to learn and master it"
    dialogue = text_generator(prompt, max_length=135, truncation=True)[0]['generated_text']
    return dialogue
```

With sufficient context length, good prompts, and subject datasets, you might yield pseudo-agent-like abilities.

### Example 2: Web Development Concepts

Consider a `subject` column with web development concepts:

subject:
index
app
server
routes
middleware
controller
model
view
kubernetes


You might modify the prompt as follows:

```python
def generate_dialogue(subject):
    prompt = f"Generate a source code file for {subject} that builds a full stack website. Make it work in the context of previous responses if any"
    dialogue = text_generator(prompt, max_length=135, truncation=True)[0]['generated_text']
    return dialogue
```

## Advantages of Segmenting CSV Files and Serializing Scripts

1. **Efficient Output Writing**: 
   - Write the output once every 1000 rows, approximately every 45 minutes with a 16-bit pipeline.
   - Check results periodically to see if the output CSV formatting is to your liking.

2. **System Resilience**: 
   - If you lose power or have a system crash, the output of inference is already written to disk in output files, so you don’t run the system for 30 hours with nothing to show for it.
   - You can pick up where you left off with the generation process.

3. **Reduced Execution Time**:
   - Orchestrating separate Python files with a shell script reduces execution time from 35-40 hours to ~26 hours (assuming 160 max token length and my csv having >36000 rows), as the loops for a single file completely Python-based implementation introduce overhead to the inference pipeline.

## Next Steps: Subject Generator Script

The next feature to implement is a script that uses the inference pipeline and a prompt enforcing no repetition to try to make a subject generator. This way, you can automatically generate hundreds of subjects that semantically branch out, with a mostly automated dialogue generation system.

### Process Overview

1. Generate 0-k subjects for some arbitrary desired number.
2. Iterate over all of them.
3. Done in as little as three command line arguments.

---

### How to use
1. $ git clone this repo* Have the csv with subjects     #subject generator coming
2. $ pip install xformers pytorch 2.1.2 cu121            #to avoid flas attention not compiled on windows
3. $ pip install transformers pandas tdqm
4. $ python split_csv.py
5. $ python generate_scripts.py
6. $ ./run_all_checkpointed.sh

 *or download the split_csv.py, generate_scripts.py, and run_all_checkpointed.sh individually or by zip.

## Note
One issue is that execution time is mainly influenced by max token length. The ideal thing is to set max token length
low and just above the sum of tokens of the prompt and the length of the response you specify. Instruct language models aren't 100% consistent or predictable with prompt adherence. I have had outputs where it adds commentary or notes on characterizing the response which I would discard later. In this case I set max token length to 256 despite an expected 130 token length, since cut off output that lack the delimiting symbols for question and answer is no good. This however more than doubles the execution time. The current solution is to enforce response length in prompt have a concise prompt and leave room in max token legnth for extra tokens. Doin this allowed for 160 max token length and 45 minute 1000 row batches. Execution time for inference scales about linear with max token length, on my device I noticed 16 bit is six times faster than 8 bit, and inference time is a function of hardware as well.

## Idea
Modify the subject_generation.py python script with a prompt "Generate a detailed visual description of an object, character, or animal in a fantasy rpg setting different from {subject}"
This should create a large csv of visual description prompts.

Here is an example usage of the flux inference pipeline from https://github.com/black-forest-labs/flux using diffusers:

import torch
from diffusers import FluxPipeline

model_id = "black-forest-labs/FLUX.1-schnell" #you can also use `black-forest-labs/FLUX.1-dev`

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
seed = 42
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4, #use a larger number if you are using [dev]
    generator=torch.Generator("cpu").manual_seed(seed)
).images[0]
image.save("flux-schnell.png")

We could modify this code for it to serially run inference on hundreds of visual description prompts generated by the subject generation python script.

When we have a large directory of generated images, we could create a shell script to iterate the following command over all images in the directory with instant mesh:

https://github.com/TencentARC/InstantMesh
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video

This creates a workflow on a single GPU to make hundreds of 3d model assets for a video game, metaverse, and other simulative use cases.

## Update 
Image description generator has been successfully implemented with quality control. It can indefinitely generate thousands of prompts for stable diffusion pipelines, flux, auraflow, and others. The subject generator and image generator programs are essentially self perpetuating for text generation, while keeping within a certain theme. 

I wrote scripts for bat because you cannot make virtual environments in Windows with git bash' require command prompt or powershell. For the the text generation component of dialogues and description generation, standard versions of python packages like transformers are acceptable. In the case of flux and other specific frameworks its better to use venvs/conda as appropriate.  

## Image Generation - Getting Started

1. git clone https://github.com/patrickbdevaney/batched-csv-transformers-pipeline-text-generation

2. edit one of the image gen scripts e.g. generate_arch_desc_v2.py. Change the contents of the string array seed bank to descriptions of objects related to what you want to generate.

3. change the prompt to guide the theme and nature of the descriptions you want to generate. This will affect the form and content images generated. 

4. run this script until you get about 1000 descriptions.

5. run description_parser.py to normalize the contents.

6. change file name of parsed descriptions for clarity and to not overwrite with other image description lists parsed.

7. either run split_img_desc_csv.py, scriptgen-flux-schnell-4.py, and gen_maestro.bat for checkpointed image generation runs, or merge.py to merge parsed description lists and oneshot_v2.py to iterate over all descriptions at once. 

## Notes on Image Gen

Checkpointed is better for very large csvs or runs in which you generate images with flux dev with 50 steps which is time consumptive. The purpose of checkpointing is to allow a seamless way to finish generation, shut down and quit mid way through while not having to start from the beginning. One shot is better for schnell 4 step image generation.

Given the same prompt and seed, flux is very consistent with the appearance of the output. You can get an idea of what it will generate by using flux schnell four step. You can then check the image number, its corresponding row in descriptions.csv will be <number>+1. 

You can then rerun the same prompt with more steps or modify the wrapper prompt in oneshot.py or scriptgen-flux-schnell-4.py.

My recommendation is to iterate with schnell over a large number of descriptions, then pick out prompts you nike form descriptions.csv and rerun with more steps and desired modifications to the wrapper prompt of oneshot.py

