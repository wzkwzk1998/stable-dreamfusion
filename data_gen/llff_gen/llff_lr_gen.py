import numpy as np
import os
import glob
import shutil
from PIL import Image
import cv2
import copy
import imageio


def llff_centre_cropping(img):
    H = 360
    W = 504
    print(img.shape)
    img_cropping = img[img.shape[0] // 2 - H // 2 : img.shape[0] // 2 + H // 2,
                        img.shape[1] // 2 - W // 2 : img.shape[1] // 2 + W // 2]

    print(img_cropping.shape)
    
    return img_cropping


def llff_256x256_centre_cropping(img):

    res =  min(img.shape[0], img.shape[1])
    img_cropping = img[img.shape[0] // 2 - res // 2 : img.shape[0] // 2 + res // 2,
                (img.shape[1] // 2 - res // 2) : (img.shape[1] // 2 + res // 2)]
    print(img_cropping.shape)
    img = cv2.resize(img, (256, 256), cv2.INTER_AREA)

    return img_cropping
        

def llff_inpainting(llff_path, llff_inpainting_path):
    '''
    generate low resolution data for nerf_synthetic
    '''
    files = os.listdir(llff_path)
    for f in files:
        origin_path = os.path.join(llff_path, f)
        inpainting_path = os.path.join(llff_inpainting_path, f)
        if os.path.isdir(origin_path):
            llff_inpainting(origin_path, inpainting_path)
        else:
            if inpainting_path.endswith('.png') or inpainting_path.endswith('.jpg') or inpainting_path.endswith('.jpeg') \
                or inpainting_path.endswith('.PNG') or inpainting_path.endswith('.JPG') or inpainting_path.endswith('.JPEG'):
                img = cv2.imread(origin_path, flags=cv2.IMREAD_UNCHANGED)
                img = llff_centre_cropping(img)
                os.makedirs(os.path.dirname(inpainting_path), exist_ok=True)
                cv2.imwrite(inpainting_path, img)
            else:
                os.makedirs(os.path.dirname(inpainting_path), exist_ok=True)
                shutil.copy(origin_path, inpainting_path)

    return

            
if __name__ == '__main__':
    

    
    llff_dir = 'data/llff/nerf_llff_data'
    llff_inpainting_dir = 'data/llff/nerf_llff_data_504x360_centre_cropping'
    llff_inpainting(llff_dir, llff_inpainting_dir)