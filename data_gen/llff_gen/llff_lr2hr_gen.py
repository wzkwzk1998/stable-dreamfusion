import numpy as np
import os
import glob
import shutil
from PIL import Image
import copy
import imageio
from diffusers import StableDiffusionUpscalePipeline
import torch

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to('cuda')

def process(img):
    img = pipeline('', image=img).images[0]
    return img


def llff_process(llff_path, llff_process_path):
    '''
    generate low resolution data for nerf_synthetic
    '''
    files = os.listdir(llff_path)
    for f in files:
        origin_path = os.path.join(llff_path, f)
        process_path = os.path.join(llff_process_path, f)
        if os.path.isdir(origin_path):
            llff_process(origin_path, process_path)
        else:
            if process_path.endswith('.png') or process_path.endswith('.jpg') or process_path.endswith('.jpeg') \
                or process_path.endswith('.PNG') or process_path.endswith('.JPG') or process_path.endswith('.JPEG'):
                # img = cv2.imread(origin_path, flags=cv2.IMREAD_UNCHANGED)
                img = Image.open(origin_path).convert('RGB')
                img = process(img)
                os.makedirs(os.path.dirname(process_path), exist_ok=True)
                img.save(process_path)
            else:
                os.makedirs(os.path.dirname(process_path), exist_ok=True)
                shutil.copy(origin_path, process_path)

    return

            
if __name__ == '__main__':
    
    llff_dir = '/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_lr'
    llff_lowres_dir = '/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data_lr2hr'
    llff_process(llff_dir, llff_lowres_dir)