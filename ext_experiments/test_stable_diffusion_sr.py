import os
import sys
import random
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm, trange
from guidance.stable_diffusion_sr import StableDiffusionForSR
from guidance.stable_diffusion import StableDiffusion
import numpy as np
from easydict import EasyDict

img_path = './test_imgs/llff_flower.png'
iters = 50000
save_path = './test_imgs/check_interval'
save_iters = 1 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = torch.tensor(np.array(cond_img) / 255.0).to(device=device)

    sd = StableDiffusionForSR.from_pretrained('stabilityai/stable-diffusion-x4-upscaler').to(device)

    prompt = ''
    imgs, interval = sd.img_sr(prompt, image=cond_img, num_inference_steps=50, save_interval=True)
    x0_save_dir = os.path.join(save_path, f'x0')
    latentimg_save_dir = os.path.join(save_path, f'latent')
    os.makedirs(x0_save_dir, exist_ok=True)
    os.makedirs(latentimg_save_dir, exist_ok=True)
    print('img len : {}'.format(len(interval['x0'])))
    for i in range(len(interval['x0'])):
        x0img = interval['x0'][i]
        latentimg = interval['latents'][i]
        t = interval['t'][i]
        x0_save_path = os.path.join(x0_save_dir, f'{t}.png')
        latentimg_save_path = os.path.join(latentimg_save_dir, f'{t}.png')
        x0img.save(x0_save_path)
        latentimg.save(latentimg_save_path)


        