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
from guidance.stable_diffusion_inpainting import StableDiffusionForInpainting
import numpy as np
from easydict import EasyDict

img_path = './test_imgs/sr_diffusion_test_flower.png'
iters = 50000
save_path = './test_imgs/llff_flower_inpainting_x0_fromnoise.png'
save_iters = 1 
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = torch.tensor(np.array(cond_img) / 255.0).to(device=device).unsqueeze(0).permute(0, 3, 1, 2).contiguous().to(device)
    cond_img = cond_img * 2 - 1 
    
    mask = torch.zeros((1, 512, 512)).to(device)
    # pred_rgb = cond_img.detach().clone()
    # pred_rgb = F.interpolate(pred_rgb[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # rgb_temp = np.round((pred_rgb.permute(0, 2, 3, 1).contiguous().squeeze(0).detach().clone().cpu().numpy()).clip(0.0, 1.0) * 255).astype('uint8')
    # Image.fromarray(rgb_temp).save('./test_imgs/rgb_temp.png')
    pred_rgb = ((torch.randn((1, 3, 512, 512)).clamp(-1.0, 1.0) + 1) / 2).to(device).requires_grad_(True)
    pred_rgb.requires_grad_(True)
    guidance = StableDiffusionForInpainting.from_pretrained('stabilityai/stable-diffusion-2-inpainting').to(device)
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
        t = random.randint(2, 400) 
        x0 = guidance.img_inpainting_x0(prompts='', image=cond_img, init_image=pred_rgb, mask_image=mask, 
                                        height=512, width=512, start_from_step=200, output_type='tensor')
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
            x0.save('./test_imgs/x0.png')
        if i % 1000 == 0 and i != 0:
            print(f'loss : {sum(grad_list) / len(grad_list)}')
            grad_list.clear()
    
    # pbar = trange(int(iters))
    # for i in pbar:
    #     optimizer.zero_grad()
    #     with torch.no_grad():
    #         pred_rgb.clamp_(0.0, 1.0)

    #     pred_rgb.requires_grad_(True)
    #     loss, grad = guidance.train_step(opt, data, condition_dict, pred_rgb, guidance_scale=0)
    #     loss.backward()
    #     optimizer.step()
    #     if i % save_iters == 0:
    #         rgb_np = np.round((pred_rgb.detach().clone().squeeze(0).cpu().numpy()).clip(0.0, 1.0) * 255).astype('uint8')
    #         Image.fromarray(rgb_np).save(save_path)
    #     if i % 1000 == 0 and i != 0:
    #         print(f'loss : {sum(grad_list) / len(grad_list)}')
    #         grad_list.clear()
            
    #     grad_list.append(grad)
        # pbar.set_description(f'loss : {sum(grad_list) / len(grad_list)}')