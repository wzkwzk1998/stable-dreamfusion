import os
import sys
sys.path.append(os.getcwd())
import torch
import PIL.Image as Image
from tqdm import tqdm, trange
from guidance.stable_diffusion_sr import StableDiffusionForSR
import numpy as np
from easydict import EasyDict

img_path = './test_imgs/llff.png'
iters = 20000
save_path = './llff_l2img_sds.png'
save_iters = 1000
device = torch.device('cuda')


if __name__ == '__main__':
    # process cond img
    cond_img = Image.open(img_path).convert('RGB')
    cond_img = np.array(cond_img) / 255.0
    cond_img = torch.tensor(cond_img, dtype=torch.float32, device=device).unsqueeze(0).to(device)
    pred_rgb = ((torch.randn((1, 512, 512, 3)).clamp(-1.0, 1.0) + 1) / 2).to(device).requires_grad_(True)
    guidance = StableDiffusionForSR(device).to(device)
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
    condition_dict_batch = guidance.prepare_guidance_condition_batch(opt, data)
    condition_dict.update(condition_dict_batch)

    for i in trange(int(iters)):
        optimizer.zero_grad()
        loss = guidance.train_step(opt, data, condition_dict, pred_rgb, guidance_scale=7.5)
        loss.backward()
        import pdb
        pdb.set_trace()
        optimizer.step()
        if i % save_iters == 0:
            rgb_np = (pred_rgb.detach().cpu().squeeze(0) * 255).numpy().astype('uint8')
            Image.fromarray(rgb_np).save(save_path)
            print(f'save image at epoch : {i}')

            
        

        


