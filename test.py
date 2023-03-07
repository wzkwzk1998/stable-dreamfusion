import torch
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    # # load ./test_imgs/llff_flower.png
    img = Image.open('./test_imgs/llff_fern.png')
    # transform to tensor
    img = torch.tensor(np.array(img) / 255.0)
    # interpolate to 512x512
    img = torch.nn.functional.interpolate(img[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # save img
    Image.fromarray((img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('./test_imgs/llff_fern_sampling.png')

    
    # NOTE: create upsamlping image from llff flower
    # load ./test_imgs/llff_flower.png, transform to tensor and interpolate to 512x512 and save it
    # img = Image.open('./test_imgs/llff_flower_upsampling.png')
    # img = torch.tensor(np.array(img) / 255.0)
    # img = torch.nn.functional.interpolate(img[None].permute(0, 3, 1, 2), size=(512, 512), mode='bilinear', align_corners=True).to(dtype=torch.float32)
    # # save img
    # Image.fromarray((img[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save('./test_imgs/llff_flower_upsampling.png') 

    # NOTE: align mean between llff flower and llff_flower+x0_fromnoise
    # load llff_flower_upsampling.png and transform to tensor
    # llff_flower = Image.open('./test_imgs/llff_flower_upsampling.png')
    # llff_flower = torch.tensor(np.array(llff_flower) / 255.0)
    # llff_flower_x0_fromnoise = Image.open('./test_imgs/llff_flower_x0_fromnoise_copy.png')
    # llff_flower_x0_fromnoise = torch.tensor(np.array(llff_flower_x0_fromnoise) / 255.0)
    # align_mean = torch.zeros_like(llff_flower_x0_fromnoise)
    # for i in range(3):
    #     llff_flower_mean = llff_flower[:, :, i].mean()
    #     noise_mean = llff_flower_x0_fromnoise[:, :, i].mean()
    #     align_mean[:, :, i] = llff_flower_x0_fromnoise[:, :, i] - (noise_mean - llff_flower_mean)

    # # save llff_flower_x0_fromnoise.png
    # Image.fromarray((align_mean.clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)).save('./test_imgs/align_mean.png')

    # NOTE: check the pixel value of align_mean.png
    # pixel = (155, 278)
    # align_mean = Image.open('./test_imgs/align_mean.png')
    # align_mean = torch.tensor(np.array(align_mean) / 255.0)
    # print(align_mean[pixel[0], pixel[1], :])


