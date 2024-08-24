# CSV Segmentation and Inference Pipeline for Dialogue Generation

This approach involves taking a CSV file named `subject` with a single column containing *k* rows and splitting it into segmented CSV files of 1000 rows each. These segmented CSV files are then serialized. The generation script traverses the directory, identifies the CSV files, and generates Python scripts to run an inference pipeline. Each script processes the rows one by one, passing each subject to a predefined prompt designed to generate questions and answers about the subject, assuming the context of playing a character in a fantasy role-playing game world.

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

