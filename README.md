[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# MultiModal Tree of Thoughts
Multi-Modal Foundation Model -> Deepfloyd iF or stable diffusion -> mmllm -> if

The objective is to implement DALLE-3 where given an input task to generate an image => its fed and enriched by the multimodal llm => passed into image generation



# Appreciation
* Lucidrains
* Agorians


# Install
`pip install mm-tot`

# Usage
```python
from tot.main import MMTot

mmtot = MMTot(
    num_thoughts=3, 
    max_steps=5, 
    value_threshold=0.7, 
    initial_prompt="Generate an image of a city skyline at night."
)

solution = mmtot.solve()
print(f"Solution: {solution}")
```


# Architecture

# Todo


# License
MIT

