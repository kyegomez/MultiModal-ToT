[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MultiModal Tree of Thoughts
Multi Modal tree of thoughts that leverages the GPT-4 language model and the
Stable Diffusion model to generate a multimodal output and evaluate the
output based a metric from 0.0 to 1.0 and then run a search algorithm using DFS and BFS and return the best output.
    
    
task: Generate an image of a swarm of bees -> Image generator -> GPT4V evaluates the img from 0.0 to 1.0 -> DFS/BFS -> return the best output


- GPT4Vision will evaluate the image from 0.0 to 1.0 based on how likely it accomplishes the task
- DFS/BFS will search for the best output based on the evaluation from GPT4Vision
- The output will be a multimodal output that is a combination of the image and the text
- The output will be evaluated by GPT4Vision
- The prompt to the image generator will be optimized from the output of GPT4Vision and the search

# Usage
`streamlit run app.py`

# License
MIT

