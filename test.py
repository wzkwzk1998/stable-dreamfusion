import torch
import torch.nn.functional as F
import PIL.Image as Image

if __name__ == '__main__':
    # img_path = '/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/room/images_8/DJI_20200226_143850_006.png'
    # img_path = '/data5/wuzhongkai/proj/stable-dreamfusion/logs/dreamfusion_bunny_fullres_3/validation/df_ep0200_0005_rgb.png'
    img_path = '/data5/wuzhongkai/data/dreamfusion_data/llff/nerf_llff_data/fern/images_8/image000.png'
    img = Image.open(img_path).convert('RGB')
    img = img.crop((100, 100, 228, 228))
    # img.save('test_img.png')
    # img = img.resize((1024, 1024))
    # img = img.resize((128, 128))
    
    img.save('./test_imgs/llff.png')