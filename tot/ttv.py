import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


class TextToVideo:
    def __init__(
        self,
        model_id,
        output_path
    ):
        self.model_id = model_id
        self.output_path = output_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run(self, prompt):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.to("cuda")

        image = pipe(prompt).images[0]

        image.save(self.output_path)