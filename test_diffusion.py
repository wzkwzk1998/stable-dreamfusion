from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id = 'stabilityai/stable-diffusion-2-1'

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "A high detailed pig, back view"
image = pipe(prompt, height=64, width=64).images
print(len(image))
image = image[0]
    
image.save("lego_man_test.png")