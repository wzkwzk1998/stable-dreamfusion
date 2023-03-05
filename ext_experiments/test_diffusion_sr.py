import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline, StableDiffusionPipeline
import torch
import numpy as np


# load model and scheduler
# model_id = "stabilityai/stable-diffusion-2-1"
# pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipeline = pipeline.to("cuda")
model_id = 'stabilityai/stable-diffusion-x4-upscaler'
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id)

low_res_img = Image.open('./test_imgs/llff_flower.png').convert('RGB')
prompt = ''

img = pipeline(prompt=prompt, image=low_res_img).images[0]
img.save("./test_imgs/sr_diffusion_test_flower.png")

# let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))
# img_path = './test_imgs/blender_hotdog.png'
# # low_res_img = Image.open(img_path).convert('RGB').resize((128, 128))
# low_res_img = Image.open(img_path).convert("RGB")
# prompt = "a panoramic style photo of a ocean"

# img = pipeline(prompt=prompt).images[0]
# upscaled_image.save("./test_imgs/blender_l2h_img.png")
# low_res_img.save('./test_imgs/blender_l2img.png')
# upsampled_img = low_res_img.resize((512, 512))