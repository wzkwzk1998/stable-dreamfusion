"""
In this experiment, we will try to optimize the latents instead of the image directly, which is inspired by the paper: Latent-nerf and SJC.
"""
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
save_path = './test_imgs/llff_flower_sr_z0fromlatents.png'
save_iters = 10 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = torch.tensor(np.array(cond_img) / 255.0).to(device=device)

    # pred_rgb = Image.open('./test_imgs/llff_flower_upsampling.png').convert('RGB')
    # pred_rgb = torch.tensor(np.array(pred_rgb) / 255.0).to(device=device)
    # pred_rgb = pred_rgb[None].permute(0, 3, 1, 2).to(dtype=torch.float32, device=device)
    # create pred_rgb 0.5, tensor, on device
    pred_rgb = (torch.ones((1, 3, 512, 512)) / 2.0).to(device)
    
    guidance = StableDiffusionForSR.from_pretrained('stabilityai/stable-diffusion-x4-upscaler').to(device)
    grad_list = []
    with torch.no_grad():
        pred_rgb_latents = guidance.encode_imgs(pred_rgb).to(device)
    pred_rgb_latents.requires_grad_(True)
    text_embedded = guidance.get_text_embeds([''], ['']).to(device)        # empty text and negative text
    optimizer = torch.optim.Adam([pred_rgb_latents], lr=1e-3)
    
    for i in tqdm(range(iters)):
        optimizer.zero_grad()
        step = random.randint(20, 980)
        loss, x0 = guidance.img_sr_x0_fromlatents(text_embeddings=text_embedded, image=cond_img, latents=pred_rgb_latents, from_step=step, output_type='tensor')
        loss.backward()
        # grad = pred_rgb - x0
        # pred_rgb.backward(grad)
        # print(pred_rgb_latents.grad)
        optimizer.step()
        grad_list.append(torch.mean(loss.detach().cpu()).item())

        if i % save_iters == 0:
            with torch.no_grad():
                pred_rgb_detach = guidance.decode_latents(pred_rgb_latents)
            pred_rgb_detach = pred_rgb_detach.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            pred_rgb_detach = guidance.numpy_to_pil(pred_rgb_detach)[0]
            pred_rgb_detach.save(save_path)
            x0 = x0.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            x0 = guidance.numpy_to_pil(x0)[0]
            x0.save('./test_imgs/x0_sr_latents.png')
        if i % 1000 == 0 and i != 0:
            print(f'loss : {sum(grad_list) / len(grad_list)}')
            grad_list.clear()
    