from tot.main import MMTot

mmtot = MMTot(
    num_thoughts=3, 
    max_steps=5, 
    value_threshold=0.7, 
    initial_prompt="Generate an image of a city skyline at night."
)

solution = mmtot.solve()
print(f"Solution: {solution}")