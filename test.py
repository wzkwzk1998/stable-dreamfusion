import torch
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    # # load ./test_imgs/llff_flower.png
    # img = Image.open('./test_imgs/llff_flower.png')
    # # transform to tensor
    # img = torch.tensor(np.array(img) / 255.0)
    # # interpolate to 512x512
    # img = torch.nn.functional.interpolate(img[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # # save img
    # Image.fromarray((img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('./test_imgs/llff_flower_upsampling.png')

    
    # load ./test_imgs/llff_flower.png, transform to tensor and interpolate to 512x512 and save it
    img = Image.open('./test_imgs/llff_flower.png')
    img = torch.tensor(np.array(img) / 255.0)
    img = torch.nn.functional.interpolate(img[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # save img
    Image.fromarray((img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('./test_imgs/llff_flower_upsampling.png') 


