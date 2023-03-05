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
save_path = './test_imgs/llff_flower_x0_fromnoise.png'
save_iters = 1 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = torch.tensor(np.array(cond_img) / 255.0).to(device=device)

    # pred_rgb = cond_img.detach().clone()
    # pred_rgb = F.interpolate(pred_rgb[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # pred_rgb = (torch.ones((1, 3, 512, 512)) / 2.0).to(device=device).requires_grad_(True)
    pred_rgb = Image.open('./test_imgs/llff_flower_x0_fromnoise.png').convert('RGB')
    pred_rgb = torch.tensor(np.array(pred_rgb) / 255.0)
    pred_rgb = pred_rgb.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).requires_grad_(True)

    guidance = StableDiffusionForSR.from_pretrained('stabilityai/stable-diffusion-x4-upscaler').to(device)
    optimizer = torch.optim.Adam([pred_rgb], lr=1e-3)
    opt = {
        'text': '',
        'negative': '',
        'dir_text': False,
        'h_guidance': 512,
        'w_guidance': 512,
    }
    opt = EasyDict(opt)
    data = {
        'H' : 512,
        'W' : 512,
        'condition_image': cond_img,
    }
    condition_dict = guidance.prepare_guidance_condition(opt)
    grad_list = []
    prompt = ''
    
    for i in tqdm(range(iters)):
        optimizer.zero_grad()
        x0 = guidance.img_sr_x0(prompts='', image=cond_img, init_image=pred_rgb, start_from_step=200, output_type='tensor')
        grad = pred_rgb - x0
        pred_rgb.backward(gradient=grad)
        optimizer.step()
        grad_list.append(torch.mean(grad.detach().cpu() ** 2).item())

        if i % save_iters == 0:
            pred_rgb_detach = pred_rgb.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            pred_rgb_detach = guidance.numpy_to_pil(pred_rgb_detach)[0]
            pred_rgb_detach.save(save_path)
            x0 = x0.detach().cpu().permute(0, 2, 3, 1).float().numpy()
            x0 = guidance.numpy_to_pil(x0)[0]
            x0.save('./test_imgs/x0_sr.png')
        if i % 1000 == 0 and i != 0:
            print(f'loss : {sum(grad_list) / len(grad_list)}')
            grad_list.clear()
    