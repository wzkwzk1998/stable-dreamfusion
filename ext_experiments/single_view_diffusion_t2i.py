import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm, trange
from guidance.stable_diffusion import StableDiffusion
import numpy as np
from easydict import EasyDict

img_path = './test_imgs/llff_flower.png'
iters = 50000 
save_path = './test_imgs/llff_l2img_t2i.png'
save_iters = 100 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    # scale to [-1, 1]
    # pred_rgb = ((torch.randn((1, 512, 512, 3)).clamp(-1.0, 1.0) + 1) / 2).to(device).requires_grad_(True)
    # create pred_rgb 0.5, tensor, on device
    pred_rgb = (torch.ones((1, 3, 512, 512)) / 2.0).to(device=device).requires_grad_(True)
    guidance = StableDiffusion.from_pretrained('stabilityai/stable-diffusion-2-1-base').to(device)
    optimizer = torch.optim.Adam([pred_rgb], lr=1e-3)
    opt = {
        'text': 'a rose',
        'negative': '',
        'dir_text': False,
        'h_guidance': 512,
        'w_guidance': 512,
    }
    opt = EasyDict(opt)
    data = {
        'H' : 512,
        'W' : 512,
    }
    condition_dict = guidance.prepare_guidance_condition(opt)
    grad_list = []
    
    pbar = trange(int(iters))
    for i in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            pred_rgb.clamp_(0.0, 1.0)
        loss, grad = guidance.train_step(opt, data, condition_dict, pred_rgb, guidance_scale=100)
        loss.backward()
        optimizer.step()
        if i % save_iters == 0:
            # save pred_rgb 
            rgb_np = pred_rgb.detach().cpu().numpy()
            rgb_np = np.transpose(rgb_np, (0, 2, 3, 1))
            rgb_img = Image.fromarray((rgb_np[0] * 255).astype(np.uint8))
            rgb_img.save(save_path)

        if i % 1000 == 0 and i != 0:
            print(f'loss : {sum(grad_list) / len(grad_list)}')
            grad_list.clear()
            
        grad_list.append(grad)
        
        # pbar.set_description(f'loss : {sum(grad_list) / len(grad_list)}')
