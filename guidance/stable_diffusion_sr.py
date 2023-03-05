import os
import sys
sys.path.append(os.getcwd())
from typing import Callable, List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, \
    DDPMScheduler, DDIMScheduler, LMSDiscreteScheduler
from diffusers import DiffusionPipeline
import numpy as np

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import PIL.Image as Image
from tqdm import tqdm, trange
import time
from guidance.gradient_utils import SpecifyGradient


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class StableDiffusionForSR(DiffusionPipeline):
    def __init__(self,
                vae: AutoencoderKL,
                text_encoder: CLIPTextModel,
                tokenizer: CLIPTokenizer,
                unet: UNet2DConditionModel,
                low_res_scheduler: DDPMScheduler,
                scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
                max_noise_level: int = 350):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
        )
        self.register_to_config(max_noise_level=max_noise_level)

        # try:
        #     with open('./TOKEN', 'r') as f:
        #         self.token = f.read().replace('\n', '') # remove the last \n!
        #         print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        # except FileNotFoundError as e:
        #     self.token = True
        #     print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        # self.device = device
        # self.num_train_timesteps = 1000
        # self.min_step = int(self.num_train_timesteps)
        # self.max_step = int(self.num_train_timesteps)

        # self.sd_version = sd_version
        # self.hf_key = hf_key
        # if hf_key is not None:
        #     print(f'[INFO] using hugging face custom model key: {hf_key}')
        #     model_key = hf_key
        # elif self.sd_version == '2.1':
        #     model_key = "stabilityai/stable-diffusion-x4-upscaler"
        # else:
        #     raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # print(f'[INFO] loading stable diffusion : {model_key}')

        # # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)

        # # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)

        # # 3. The UNet model for generating the latents.
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        # # 4. Create a scheduler for inference
        # # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        # self.low_res_scheduler = DDPMScheduler.from_pretrained(model_key, subfolder='low_res_scheduler')
        

        # self.vae.requires_grad_(False)
        # self.text_encoder.requires_grad_(False)
        # self.unet.requires_grad_(False)

        print(f'[INFO] loaded stable diffusion!')
    

    def prepare_guidance_condition(self, opt):
        text_z = self.prepare_text_embeddings(opt)
        condition_dict = {
            'text_z': text_z,
        }

        return condition_dict

        
    def prepare_text_embeddings(self, opt):

        if opt.text is None:
            print(f"[WARN] text prompt is not provided.")
            text_z = None

        if not opt.dir_text:
            text_z = self.get_text_embeds([opt.text], [opt.negative])
        else:
            text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{opt.text}, {d} view"

                negative_text = f"{opt.negative}"

                # explicit negative dir-encoded text
                if opt.suppress_face:
                    if negative_text != '': negative_text += ', '

                    if d == 'back': negative_text += "face"
                    # elif d == 'front': negative_text += ""
                    elif d == 'side': negative_text += "face"
                    elif d == 'overhead': negative_text += "face"
                    elif d == 'bottom': negative_text += "face"
                
                text_z_single = self.get_text_embeds([text], [negative_text])
                text_z.append(text_z_single)
        
        return text_z


    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings


    def decode_latents(self, latents):
        latents = 1 / 0.08333 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        # map imgs [0, 1] to [-1, 1]
        imgs = 2 * imgs - 1 
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.08333
        return latents


    def prepare_image(self, image):
        if isinstance(image, Image.Image):
            print('PIL.Image')
            width, height = image.size
            width = width - width % 64
            height = height - height % 64
            image = image.resize((width, height))
            image = np.array(image.convert("RGB"))
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image[None].permute(0, 3, 1, 2).contiguous()
            elif image.ndim == 4:
                image = image.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError('Invalid image shape')
            # image = image.to(dtype=torch.float32) / 127.5 - 1.0         # scale to [-1, 1]
            image = image.to(dtype=torch.float32) * 2 - 1.0

        return image
        
            
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height, width)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def train_step(self, 
                    opt,
                    data,
                    condition_dict,
                    pred_rgb:torch.Tensor, 
                    guidance_scale:float = 7.5,
                    noise_level: int = 20):
        '''
        pred_rgb: [0, 1]
        image: [0, 1]
        mask: 0 or 1
        '''        
        # 1. prepare text embeddigns and pred_rgb data
        text_embeddings = condition_dict['text_z']
        if opt.dir_text:
            dirs = data['dir'] # [B,]
            text_embeddings = text_embeddings[dirs]
        else:
            text_embeddings = text_embeddings
        h_image = data['H']
        w_image = data['W']
        height = opt.h_guidance
        width = opt.w_guidance
        pred_rgb = pred_rgb.reshape((1, h_image, w_image, 3)).permute(0, 3, 1, 2).contiguous()
        pred_rgb = F.interpolate(pred_rgb, (height, width), mode='bilinear', align_corners=False)
        
        # 2. prepare condition image
        # NOTE: the conditional image is from [-1 ~ 1]
        image = data['condition_image']
        image = self.prepare_image(image)
        image = image.to(dtype=text_embeddings.dtype, device=self.device)
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = torch.randn(image.shape, device=self.device, dtype=text_embeddings.dtype)
        # image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * 2)

        # encode image into latents with vae, requires grad!
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(pred_rgb)

        # Check that sizes of mask, masked image and latents match
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # predict the noise residual with unet, NO grad!
        # NOTE: pay attention that it is with out grad
        # TODO: the time step in here may need to adjust
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # in PNDM schedular, this function will return the input directly
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # add the inpainting    
            latent_model_input = torch.cat([latent_model_input, image], dim=1)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=noise_level
            ).sample
        
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # optimize for image
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = ((1 - self.alphas[t]) ** 0.5) / (self.alphas[t] ** 0.5)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        loss = SpecifyGradient.apply(latents, grad)
        
        # inpainting mse loss
        return loss, torch.mean(grad.detach().clone() ** 2).item()


    def forward(self, 
                opt,
                data,
                condition_dict,
                pred_rgb:torch.Tensor, 
                guidance_scale:float = 7.5,
                noise_level: int = 20):
        
        # 1. prepare text embeddigns and pred_rgb data
        text_embeddings = condition_dict['text_z']
        if opt.dir_text:
            dirs = data['dir'] # [B,]
            text_embeddings = text_embeddings[dirs]
        else:
            text_embeddings = text_embeddings
        h_image = data['H']
        w_image = data['W']
        height = opt.h_guidance
        width = opt.w_guidance
        pred_rgb = pred_rgb.reshape((1, h_image, w_image, 3)).permute(0, 3, 1, 2).contiguous()
        pred_rgb = F.interpolate(pred_rgb, (height, width), mode='bilinear', align_corners=False)
        
        # 2. prepare condition image
        # NOTE: the conditional image is from [-1 ~ 1]
        image = condition_dict['condition_image']
        image = self.prepare_image(image) 
        image = image.to(dtype=text_embeddings.dtype, device=self.device)
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = torch.randn(image.shape, device=self.device, dtype=text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        # draw condition image
        import PIL.Image as Image
        image_detach = ((image / 2) + 0.5).clamp(0, 1)
        image_detach = (image * 255)[0].permute(1, 2, 0).contiguous().detach().cpu().numpy().astype('uint8')
        Image.fromarray(image_detach).save('./test_imgs/noisy_image.png')
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * 2)

        # encode image into latents with vae, requires grad!
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(pred_rgb)

        # Check that sizes of mask, masked image and latents match
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # predict the noise residual with unet, NO grad!
        # TODO: the time step in here may need to adjust
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        # with torch.no_grad():
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        # in PNDM schedular, this function will return the input directly
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # add the inpainting    
        latent_model_input = torch.cat([latent_model_input, image], dim=1)

        noise_pred = self.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=noise_level
        ).sample
        
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        loss = torch.mean((noise_pred - noise) ** 2)
        
        # inpainting mse loss
        return loss
    
    @torch.no_grad()
    def img_sr_x0(
            self, 
            prompts, 
            image,
            init_image: torch.Tensor,
            start_from_step: int,
            num_inference_steps=50, 
            guidance_scale=9.0, 
            negative_prompts='', 
            noise_level=20, 
            output_type='pil'):


        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        # 2. prepare image
        image = self.prepare_image(image)
        image = image.to(dtype=text_embeddings.dtype, device=self.device)
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = torch.randn(image.shape, device=self.device, dtype=text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * 2)

        # 3. create latents (random noise)
        height, width = init_image.shape[2] // 4, init_image.shape[3] // 4
        assert height == image.shape[2]
        assert width == image.shape[3]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(init_image)
        noise = torch.randn_like(latents)
        start_from_step = torch.tensor([start_from_step], dtype=torch.long, device=self.device)
        latents = self.scheduler.add_noise(latents, noise, start_from_step)

        # 3. check input
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        

        latent_model_input = torch.cat([latents] * 2)
        
        # in PNDM schedular, is function will return the input directly
        latent_model_input = torch.cat([latent_model_input, image], dim=1)
        noise_pred = self.unet(
            latent_model_input, start_from_step, encoder_hidden_states=text_embeddings, class_labels=noise_level
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        alphas = self.scheduler.alphas_cumprod
        x0_latents = (latents - ((1 - alphas[start_from_step]) ** 0.5) * noise_pred) * (1 / alphas[start_from_step])

        # Text embeds -> img latents
        # self.scheduler.set_timesteps(num_inference_steps)
        # for i, t in enumerate(tqdm(self.scheduler.timesteps)):
        #     print(f'time step is : {t}')
        #     latent_model_input = torch.cat([latents] * 2)
            
        #     # in PNDM schedular, is function will return the input directly
        #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        #     latent_model_input = torch.cat([latent_model_input, image], dim=1)

        #     # predict the noise residual
        #     noise_pred = self.unet(
        #         latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=noise_level
        #     ).sample

        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 

        #     latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.vae.to(dtype=torch.float32)
        image = self.decode_latents(x0_latents)
        if output_type == 'pil':
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        return image


    @torch.no_grad()
    def img_sr_x0(
            self, 
            prompts, 
            image,
            init_image: torch.Tensor,
            start_from_step: int,
            num_inference_steps=50, 
            guidance_scale=9.0, 
            negative_prompts='', 
            noise_level=20, 
            output_type='pil'):


        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        # 2. prepare image
        image = self.prepare_image(image)
        image = image.to(dtype=text_embeddings.dtype, device=self.device)
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = torch.randn(image.shape, device=self.device, dtype=text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * 2)

        # 3. create latents (random noise)
        height, width = init_image.shape[2] // 4, init_image.shape[3] // 4
        assert height == image.shape[2]
        assert width == image.shape[3]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(init_image)
        noise = torch.randn_like(latents)
        start_from_step = torch.tensor([start_from_step], dtype=torch.long, device=self.device)
        latents = self.scheduler.add_noise(latents, noise, start_from_step)

        # 3. check input
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )
        

        latent_model_input = torch.cat([latents] * 2)
        
        # in PNDM schedular, is function will return the input directly
        latent_model_input = torch.cat([latent_model_input, image], dim=1)
        noise_pred = self.unet(
            latent_model_input, start_from_step, encoder_hidden_states=text_embeddings, class_labels=noise_level
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        alphas = self.scheduler.alphas_cumprod
        x0_latents = (latents - ((1 - alphas[start_from_step]) ** 0.5) * noise_pred) * (1 / alphas[start_from_step])

        # Text embeds -> img latents
        # self.scheduler.set_timesteps(num_inference_steps)
        # for i, t in enumerate(tqdm(self.scheduler.timesteps)):
        #     print(f'time step is : {t}')
        #     latent_model_input = torch.cat([latents] * 2)
            
        #     # in PNDM schedular, is function will return the input directly
        #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        #     latent_model_input = torch.cat([latent_model_input, image], dim=1)

        #     # predict the noise residual
        #     noise_pred = self.unet(
        #         latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=noise_level
        #     ).sample

        #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 

        #     latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.vae.to(dtype=torch.float32)
        image = self.decode_latents(x0_latents)
        if output_type == 'pil':
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        return image
        
    
    @torch.no_grad()
    def img_sr(self, 
            prompts, 
            image, 
            num_inference_steps=50, 
            guidance_scale=9.0, 
            negative_prompts='', 
            noise_level=20, 
            output_type='pil',
            save_interval=False):

        # 1. prepare text embeddings
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        # 2. prepare image
        image = self.prepare_image(image)
        image = image.to(dtype=text_embeddings.dtype, device=self.device)
        noise_level = torch.tensor([noise_level], dtype=torch.long, device=self.device)
        noise = torch.randn(image.shape, device=self.device, dtype=text_embeddings.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        image = torch.cat([image] * 2)
        noise_level = torch.cat([noise_level] * 2)

        # 3. create latents (random noise)
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
                    text_embeddings.shape[0] // 2,
                    num_channels_latents,
                    height,
                    width,
                    text_embeddings.dtype,
                    self.device,
                    generator = None,
                    latents = None, 
                    )

        # 3. check input
        num_channels_image = image.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        interval_dict = {
            'latents': [],
            'x0': [],
            't': []
        }
        alphas = self.scheduler.alphas_cumprod
        # Text embeds -> img latents
        self.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            print(f'time step is : {t}')
            latent_model_input = torch.cat([latents] * 2)
            
            # in PNDM schedular, is function will return the input directly
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = torch.cat([latent_model_input, image], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=noise_level
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            if save_interval:
                latents2img_interval = self.decode_latents(latents)
                x0_latents = (latents - ((1 - alphas[t]) ** 0.5) * noise_pred) * (1 / alphas[t])
                x02img_interval = self.decode_latents(x0_latents)
                latents2img_interval = latents2img_interval.cpu().permute(0, 2, 3, 1).float().numpy()
                latents2img_interval = self.numpy_to_pil(latents2img_interval)[0]
                x02img_interval = x02img_interval.cpu().permute(0, 2, 3, 1).float().numpy()
                x02img_interval = self.numpy_to_pil(x02img_interval)[0]

                interval_dict['latents'].append(latents2img_interval)
                interval_dict['x0'].append(x02img_interval)
                interval_dict['t'].append(t)

        self.vae.to(dtype=torch.float32)
        image = self.decode_latents(latents)
        if output_type == 'pil':
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)

        if save_interval:
            return image, interval_dict

        return image


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    import requests
    import PIL
    from io import BytesIO

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()
    
    seed_everything(opt.seed)

    device = torch.device('cuda:0')
    sd = StableDiffusionForSR.from_pretrained('stabilityai/stable-diffusion-x4-upscaler').to(device)

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    image_path = './test_imgs/llff_flower.png'
    low_res_image = PIL.Image.open(image_path).convert("RGB")
    init_image = np.array(low_res_image) / 255.0
    init_image = torch.tensor(init_image, dtype=torch.float32, device=device)
    init_image = init_image[None].permute(0, 3, 1, 2).contiguous()
    init_image = F.interpolate(init_image, (init_image.shape[2] * 4, init_image.shape[3] * 4), mode='bilinear')
    # low_res_image = torch.tensor(np.array(low_res_image) / 255.0)
    prompt = ''
    # imgs = sd.img_sr_x0(prompt, image=low_res_image, init_image=init_image, start_from_step=300, num_inference_steps=2)
    imgs = sd.img_sr(prompt, image=low_res_image, num_inference_steps=50)
    imgs = imgs[0]
    # PIL.Image.fromarray(((imgs.permute(1, 2, 0).contiguous().cpu().numpy()) * 255).round().astype('uint8')).save('./test_imgs/sr_diffusion_test_flower.png')
    imgs.save('./test_imgs/sr_diffusion_test_flower.png')





