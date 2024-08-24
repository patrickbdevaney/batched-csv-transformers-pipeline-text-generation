##Merge
The merge utility is to merge all of the serrialized output csvs to one csv. I used The following prompt

 prompt = f"Role: character from ancient fantasy RPG. in character Q&A about {subject}. Q: less than 15 words. A: less than 35 words. Enclose Q in *1* <question> *1* and A in *2* <answer> *2*. Do not include any commentary or notes on output."

##Parse
 *1* and *2* are delimiters to unambiguously extract questions and answers from your output. 

 You could use the same method to have whatever data you want to extract to a column for classification, llm training etc.

##Subject Generation
 Of particular note is the subject generator. Its purpose is to generate subjects for later use in the pipeline. It passes words from an initial bank into the prompt and has the llm generate a related word. Then it wrties this word to that bank. It stipulates that a word cannot be reused when it draws a word again from the bank to feed the llm.

 The implication is that you can use this to generate an expanding semantic tree and potentially a large list of subjects related to your initial subject.

 It could be used that way for idea discovery and brianstorming for creativity, to come up with a list of words you didn't think of. 

 Then you can split the yielded csv with csc_split.py, generate the serialized python programs with generate_scripts.py, and run inference to have it explain many new concepts or output the task you specify in the prompt.

 You could prompt it as explained before to compose starter code to populate a complex repository with, but this has its pitfalls.
