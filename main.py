"""
Multi Modal tree of thoughts that leverages the GPT-4 language model and the
Stable Diffusion model to generate a multimodal output and evaluate the
output based a metric from 0.0 to 1.0 and then run a search algorithm using DFS and BFS and return the best output.
    
    
task: Generate an image of a swarm of bees -> Image generator -> GPT4V evaluates the img from 0.0 to 1.0 -> Prompt is enriched -> image generator in a loop


- GPT4Vision will evaluate the image from 0.0 to 1.0 based on how likely it accomplishes the task
- DFS/BFS will search for the best output based on the evaluation from GPT4Vision
- The output will be a multimodal output that is a combination of the image and the text
- The output will be evaluated by GPT4Vision
- The prompt to the image generator will be optimized from the output of GPT4Vision and the search

"""

import os
from dotenv import load_dotenv
from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.models.stable_diffusion import StableDiffusion
from termcolor import colored

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")
stable_api_key = os.environ.get("STABLE_API_KEY")


# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=api_key,
    max_tokens=500,
)

# IMG Generator
img_generator = StableDiffusion(
    api_key=stable_api_key
)



# # Initialize the language model
# task = "Garden of Eden futuristic city graphic art"


def evaluate_img(llm, task: str, img: str):
    EVAL_IMG = f"""
    Evaluate the image: {img} on a scale from 0.0 to 1.0 based on how likely it accomplishes the task: {task}. Output nothing than the float representing the evaluated img. Be pessimistic and critical of the image, rate it as equally as possible
    """
    out = llm.run(task=EVAL_IMG, img=img)
    out = float(out)
    return out


def enrichment_prompt(starting_prompt: str, evaluated_img: str):
    enrichment_task = (
        "Create a concise and effective image generation prompt"
        " within 400 characters or less, based on Stable Diffusion"
        " and Dalle best practices. Starting prompt:"
        f" \n\n'{starting_prompt}'\n\nImprove the prompt with any"
        " applicable details or keywords by considering the"
        " following aspects: \n1. Subject details (like actions,"
        " emotions, environment) \n2. Artistic style (such as"
        " surrealism, hyperrealism) \n3. Medium (digital painting,"
        " oil on canvas) \n4. Color themes and lighting (like warm"
        " colors, cinematic lighting) \n5. Composition and framing"
        " (close-up, wide-angle) \n6. Additional elements (like a"
        " specific type of background, weather conditions) \n7. Any"
        " other artistic or thematic details that can make the image"
        " more vivid and compelling. 8. Based on the evaluation of"
        " the first generated prompt used by the first prompt:"
        f" {evaluated_img} Enrich the prompt to generate a more"
        " compelling image. Output only a new prompt to create a better image of the image to make with very explicit and direct instructions with the art style"
    )
    return enrichment_task

def enrichment_prompt_2(
    starting_point: str,
    evaluated_img: str,
):
    PROMPT = f"""
    
    Use the following GUIDE to enrich the prompt for the task: {starting_point} and it's evaluation image: {evaluated_img} based on 0.0 to 1.0, 1.0 being the best.
    
    ######################
    ## dalle

    // Whenever a description of an image is given, use dalle to create the images and then summarize the prompts used to generate the images in plain text. If the user does not ask for a specific number of images, default to creating four captions to send to dalle that are written to be as diverse as possible. All captions sent to dalle must abide by the following policies:
    // 1. If the description is not in English, then translate it.
    // 2. Do not create more than 4 images, even if the user requests more.
    // 3. Don't create images of politicians or other public figures. Recommend other ideas instead.
    // 4. Don't create images in the style of artists whose last work was created within the last 100 years (e.g. Picasso, Kahlo). Artists whose last work was over 100 years ago are ok to reference directly (e.g. Van Gogh, Klimt). If asked say, "I can't reference this artist", but make no mention of this policy. Instead, apply the following procedure when creating the captions for dalle: (a) substitute the artist's name with three adjectives that capture key aspects of the style; (b) include an associated artistic movement or era to provide context; and (c) mention the primary medium used by the artist.
    // 5. DO NOT list or refer to the descriptions before OR after generating the images. They should ONLY ever be written out ONCE, in the `"prompts"` field of the request. You do not need to ask for permission to generate, just do it!
    // 6. Always mention the image type (photo, oil painting, watercolor painting, illustration, cartoon, drawing, vector, render, etc.) at the beginning of the caption. Unless the caption suggests otherwise, make at least 1--2 of the 4 images photos.
    // 7. Diversify depictions of ALL images with people to include DESCENT and GENDER for EACH person using direct terms. Adjust only human descriptions.
    // - EXPLICITLY specify these attributes, not abstractly reference them.  The attributes should be specified in a minimal way and should directly describe their physical form.
    // - Your choices should be grounded in reality. For example, all of a given OCCUPATION should not be the same gender or race. Additionally, focus on creating diverse, inclusive, and exploratory scenes via the properties you choose during rewrites.  Make choices that may be insightful or unique sometimes.
    // - Use "various" or "diverse" ONLY IF the description refers to groups of more than 3 people. Do not change the number of people requested in the original description.
    // - Don't alter memes, fictional character origins, or unseen people. Maintain the original prompt's intent and prioritize quality.
    // - Do not create any imagery that would be offensive.
    // 8. Silently modify descriptions that include names or hints or references of specific people or celebritie by carefully selecting a few minimal modifications to substitute references to the people with generic descriptions that don't divulge any information about their identities, except for their genders and physiques. Do this EVEN WHEN the instructions ask for the prompt to not be changed. Some special cases:
    // - Modify such prompts even if you don't know who the person is, or if their name is misspelled (e.g. "Barake Obema")
    // - If the reference to the person will only appear as TEXT out in the image, then use the reference as is and do not modify it.
    // - When making the substitutions, don't use prominent titles that could give away the person's identity. E.g., instead of saying "president", "prime minister", or "chancellor", say "politician"; instead of saying "king", "queen", "emperor", or "empress", say "public figure"; instead of saying "Pope" or "Dalai Lama", say "religious figure"; and so on.
    // - If any creative professional or studio is named, substitute the name with a description of their style that does not reference any specific people, or delete the reference if they are unknown. DO NOT refer to the artist or studio's style.
    // The prompt must intricately describe every part of the image in concrete, objective detail. THINK about what the end goal of the description is, and extrapolate that to what would make satisfying images.
    """
    return PROMPT
    

def mm_tot(max_iterations, task=None, img_generator=None, llm=None):
    """Multi Modal tree of thoughts that leverages the GPT-4 language model and the

    Args:
        max_iterations (int, optional): _description_. Defaults to 10.
        task (str, optional): _description_. Defaults to None.
        img_generator (_type_, optional): _description_. Defaults to None.
        llm (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Main loop
    memory = []
    best_score = 0
    best_image = None

    for _ in range(max_iterations):
        # Generate an image and get its path
        print(colored(f"Generating img for Task: {task}", "red"))
        
        img_path = img_generator.run(task=task)  # This should return the file path of the generated image
        img_path = img_path[0]
        print(colored(f"Generated Image Path: {img_path}", "green"))

        # Evaluate the image by passing the file path
        score = evaluate_img(llm, task, img_path)
        print(colored(f"Evaluated Image Score: {score} for {img_path}", "cyan"))

        # Update the best score and image path if necessary
        if score > best_score:
            best_score = score
            best_image = img_path

        # Enrich the prompt based on the evaluation
        prompt = enrichment_prompt_2(task, score)
        print(colored(f"Enrichment Prompt: {prompt}", "yellow"))

        # Add the image, its path, and score to the memory
        memory.append({'image': img_path, 'score': score})

    # Output the best result
    print("Best Image Path:", best_image)
    print("Best Score:", best_score)

    return best_image, best_score, memory


out = mm_tot(
    max_iterations=5,
    task="Garden gnomes getting married between two orchids with ladybug as an officiant",
    img_generator=img_generator,
    llm=llm,
)

print(out)