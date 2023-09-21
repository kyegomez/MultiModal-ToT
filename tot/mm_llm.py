import logger
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

class MultiModalInference:
    """

    A class for multimodal inference using pre-trained models from the Hugging Face Hub.

    Attributes
    ----------
    device : str
        The device to use for inference.
    checkpoint : str, optional
        The name of the pre-trained model checkpoint (default is "HuggingFaceM4/idefics-9b-instruct").
    processor : transformers.PreTrainedProcessor
        The pre-trained processor.
    max_length : int
        The maximum length of the generated text.
    chat_history : list
        The chat history.

    Methods
    -------
    infer(prompts, batched_mode=True)
        Generates text based on the provided prompts.
    chat(user_input)
        Engages in a continuous bidirectional conversation based on the user input.
    set_checkpoint(checkpoint)
        Changes the model checkpoint.
    set_device(device)
        Changes the device used for inference.
    set_max_length(max_length)
        Changes the maximum length of the generated text.
    clear_chat_history()
        Clears the chat history.


    # Usage
    ```
    from exa import MultiModalInference
    mmi = MultiModalInference()

    user_input = "User: What is in this image? https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"
    response = mmi.chat(user_input)
    print(response)

    user_input = "User: And who is that? https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
    response = mmi.chat(user_input)
    print(response)

    mmi.set_checkpoint("new_checkpoint")
    mmi.set_device("cpu")
    mmi.set_max_length(200)
    mmi.clear_chat_history()
    ```

    """
    def __init__(
        self,
        checkpoint="HuggingFaceM4/idefics-9b-instruct",
        device=None,
        torch_dtype=torch.bfloat16,
        max_length=3000
    ):
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch_dtype,   
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(checkpoint)

        self.max_length = max_length

        self.chat_history = []
    
    def run(
        self,
        prompts,
        batched_mode=True
    ):
        """
        Generates text based on the provided prompts.

        Parameters
        ----------
            prompts : list
                A list of prompts. Each prompt is a list of text strings and images.
            batched_mode : bool, optional
                Whether to process the prompts in batched mode. If True, all prompts are processed together. If False, only the first prompt is processed (default is True).

        Returns
        -------
            list
                A list of generated text strings.
        """
        inputs = self.processor(
            prompts,
            add_end_of_utterance_token=False,
            return_tensors="pt"
        ).to(self.device) if batched_mode else self.processor(
            prompts[0],
            return_tensors="pt"
        ).to(self.device)
        

        exit_condition = self.processor.tokenizer(
            "<end_of_utterance>",
            add_special_tokens=False
        ).input_ids

        bad_words_ids = self.processor.tokenizer(
            [
                "<image>",
                "<fake_token_around_image"
            ],

            add_special_tokens=False
        ).input_ids

        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=self.max_length,
        )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return generated_text
    
    def chat(self, user_input):
        """
        Engages in a continuous bidirectional conversation based on the user input.

        Parameters
        ----------
            user_input : str
                The user input.

        Returns
        -------
            str
                The model's response.
        """
        self.chat_history.append(user_input)
        
        prompts = [self.chat_history]
        
        response = self.run(prompts)[0]

        self.chat_history.append(response)

        return response
    
    def set_checkpoint(self, checkpoint):
        """
        Changes the model checkpoint.

        Parameters
        ----------
            checkpoint : str
                The name of the new pre-trained model checkpoint.
        """
        self.model = IdeficsForVisionText2Text.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint)
    
    def set_device(self, device):
        """
        Changes the device used for inference.

        Parameters
        ----------
            device : str
                The new device to use for inference.
        """
        self.device = device
        self.model.to(self.device)
    
    def set_max_length(self, max_length):
        self.max_length = max_length
    
    def clear_chat_history(self):
        self.chat_history = []
    
    def generate_thoughts(
            self, 
            state, 
            k, 
            initial_prompt, 
            rejected_solutions=None
        ):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("New state generating thought:", state, "\n\n")
        prompt = f"""
        Accomplish the task below by decomposing it as many very explicit subtasks as possible, be very explicit and thorough denoted by 
        a search process, highlighted by markers ‘1’,..., ‘3’ as “first operations” guiding subtree exploration for the OBJECTIVE, 
        focus on the third subtree exploration. Produce prospective search steps (e.g., the subtree exploration ‘5. 11 + 1’) 
        and evaluates potential subsequent steps to either progress
        towards a solution or retrace to another viable subtree then be very thorough 
        and think atomically then provide solutions for those subtasks, 
        then return the definitive end result and then summarize it


        ########## OBJECTIVE
        {initial_prompt}
        """
        thoughts = self.run(prompt)
        # print(f"Generated thoughts: {thoughts}")
        return thoughts

        
    def generate_solution(
            self, 
            initial_prompt, 
            state, 
            rejected_solutions=None
        ):

        try:

            if isinstance(state, list):
                state_text = '\n'.join(state)
            else:
                state_text = state
            
            prompt = f"""
            Generate a series of solutions to comply with the user's instructions, 
            you must generate solutions on the basis of determining the most reliable solution in the shortest amount of time, 
            while taking rejected solutions into account and learning from them. 
            Considering the reasoning provided:\n\n
            ###'{state_text}'\n\n###
            Devise the best possible solution for the task: {initial_prompt}, Here are evaluated solutions that were rejected: 
            ###{rejected_solutions}###, 
            complete the {initial_prompt} without making the same mistakes you did with the evaluated rejected solutions. Be simple. Be direct. Provide intuitive solutions as soon as you think of them.
            """

            answer = self.run(prompt)

            print(f'Generated Solution Summary {answer}')

            return answer
        except Exception as e:
            logger.error(f"Error in generate_solutions: {e}")
            return None

    def evaluate_states(self, states, initial_prompt):
        if not states:
            return {}

        if self.evaluation_strategy == 'value':
            state_values = {}

            for state in states:
                if (type(state) == str):
                    state_text = state
                else:
                    state_text = '\n'.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                prompt = f""" 
                    To achieve the following goal: '{initial_prompt}', pessimistically value the context of the past solutions and more importantly the latest generated solution you had AS A FLOAT BETWEEN 0 AND 1\n
                    Past solutions:\n\n
                    {state_text}\n       
                    If the solutions is not making fast progress in achieving the goal, give it a lower score.
                    Evaluate all solutions AS A FLOAT BETWEEN 0 and 1:\n,  DO NOT RETURN ANYTHING ELSE
                """
                response = self.run(prompt)

                try:
                    value_text = self.run(response)
                    # print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"Evaluated Thought Value: {value}")
                except ValueError:
                    value = 0  
                state_values[state] = value
            return state_values

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
    