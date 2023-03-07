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

img_path = './test_imgs/llff_flower.png'
iters = 50000
save_path = './test_imgs/llff_l2img_sds_sr.png'
save_iters = 100 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = torch.tensor(np.array(cond_img) / 255.0).to(device=device)
    pred_rgb = cond_img.detach().clone()
    pred_rgb = F.interpolate(pred_rgb[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True)
    pred_rgb = pred_rgb.permute(0, 2, 3, 1).contiguous().to(dtype=torch.float32)
    # rgb_temp = np.round((pred_rgb.permute(0, 2, 3, 1).contiguous().squeeze(0).detach().clone().cpu().numpy()).clip(0.0, 1.0) * 255).astype('uint8')
    # Image.fromarray(rgb_temp).save('./test_imgs/rgb_temp.png')
    pred_rgb = ((torch.randn((1, 512, 512, 3)).clamp(-1.0, 1.0) + 1) / 2).to(device).requires_grad_(True)
    pred_rgb.requires_grad_(True)
    guidance = StableDiffusionForSR.from_pretrained('stabilityai/stable-diffusion-x4-upscaler').to(device)
    optimizer = torch.optim.Adam([pred_rgb], lr=1e-5)
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
    
    pbar = trange(int(iters))
    for i in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            pred_rgb -= pred_rgb.min()
            pred_rbg /= (pred_rgb.max() - pred_rgb.min())
            # pred_rgb.clamp_(0.0, 1.0)

        pred_rgb.requires_grad_(True)
        loss, grad = guidance.train_step(opt, data, condition_dict, pred_rgb, guidance_scale=0)
        loss.backward()
        optimizer.step()
        if i % save_iters == 0:
            rgb_np = np.round((pred_rgb.detach().clone().squeeze(0).cpu().numpy()).clip(0.0, 1.0) * 255).astype('uint8')
            Image.fromarray(rgb_np).save(save_path)
        if i % 1000 == 0 and i != 0:
            print(f'loss : {sum(grad_list) / len(grad_list)}')
            grad_list.clear()
            
        grad_list.append(grad)
        # pbar.set_description(f'loss : {sum(grad_list) / len(grad_list)}')