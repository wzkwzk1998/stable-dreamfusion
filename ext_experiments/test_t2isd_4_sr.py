import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm, trange
from guidance.stable_diffusion_sr import StableDiffusionForSR
from guidance.stable_diffusion import StableDiffusion
import numpy as np
from easydict import EasyDict


img_path = './test_imgs/llff_flower_upsampling.png'
save_path = './test_imgs/llff_flower_sr_using_upsampling.png'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    # load img from img_path and convert to tensor
    img = Image.open(img_path).convert('RGB')
    img = torch.tensor(np.array(img) / 255.0)
    # permute 
    img = img.permute(2, 0, 1).unsqueeze(0).contiguous().to(dtype=torch.float32).to(device=device)

    # load stable diffusion model
    sd = StableDiffusion.from_pretrained('stabilityai/stable-diffusion-2-1-base').to(device)

    img = sd.img_t2i_from_interval('a rose', init_image=img, from_step=100, num_inference_steps=100)
    img = img[0]
    
    img.save(save_path)
    

