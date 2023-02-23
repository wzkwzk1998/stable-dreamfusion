import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline, LDMSuperResolutionPipeline
import torch
import numpy as np


# load model and scheduler
model_id = "YiYiXu/latent-upscaler"
pipeline = LDMSuperResolutionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipeline = pipeline.to("cuda")

# let's download an  image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((128, 128))
img_path = './test_imgs/blender_hotdog.png'
# low_res_img = Image.open(img_path).convert('RGB').resize((128, 128))
low_res_img = Image.open(img_path).convert("RGB")

upscaled_image = pipeline(image=low_res_img).images[0]
upscaled_image.save("./test_imgs/blender_l2h_img.png")
low_res_img.save('./test_imgs/blender_l2img.png')
upsampled_img = low_res_img.resize((512, 512))
upsampled_img.save("./test_imgs/blender_l2up_img.png")