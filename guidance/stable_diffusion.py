from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, DDPMScheduler
from diffusers import StableDiffusionPipeline, DiffusionPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import time
import PIL.Image as Image


from torch.cuda.amp import custom_bwd, custom_fwd 

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad) 
        
        # dummy loss value
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler,
        # safety_checker: StableDiffusionSafetyChecker,
        feature_extractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        # self.device = device
        # self.sd_version = sd_version

        # print(f'[INFO] loading stable diffusion...')
        
        # if hf_key is not None:
        #     print(f'[INFO] using hugging face custom model key: {hf_key}')
        #     model_key = hf_key
        # elif self.sd_version == '2.1':
        #     model_key = "stabilityai/stable-diffusion-2-1-base"
        # elif self.sd_version == '2.0':
        #     model_key = "stabilityai/stable-diffusion-2-base"
        # elif self.sd_version == '1.5':
        #     model_key = "runwayml/stable-diffusion-v1-5"
        # else:
        #     raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # # Create model
        # self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        # self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        # self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")
        scheduler = DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder="scheduler")
        

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            # safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loaded stable diffusion!')
    
    def prepare_guidance_condition(self, opt):
        text_z = self.prepare_text_embeddings(opt)
        condition_dict = {
            'text_z': text_z,
        }

        return condition_dict
    
    def prepare_guidance_condition_batch(self, opt, data):
        return {}
        
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


    def train_step(self, opt, data, condition_dict,  pred_rgb, guidance_scale=100):

        text_z = condition_dict['text_z']
        if opt.dir_text:
            dirs = data['dir'] # [B,]
            text_z = text_z[dirs]
        else:
            text_z = text_z
        h_image = data['H']
        w_image = data['W']

        # _t = time.time()
        pred_rgb = pred_rgb.reshape((1, h_image, w_image, 3)).permute(0, 3, 1, 2).contiguous()
        pred_rgb_512 = F.interpolate(pred_rgb, (opt.h_guidance, opt.w_guidance), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = torch.tensor([200], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_z).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        alphas = self.scheduler.alphas_cumprod

        # w(t), sigma_t^2
        w = (1 - alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        # _t = time.time()
        loss = SpecifyGradient.apply(latents, grad) 
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return loss, torch.mean(grad ** 2).item() 


    @torch.no_grad()
    def img_t2i_x0(
            self, 
            prompts, 
            init_image: torch.Tensor,
            height: int=512,
            width: int=512,
            start_from_step: int=200,
            num_inference_steps=50, 
            guidance_scale=100, 
            negative_prompts='', 
            noise_level=20, 
            output_type='pil'):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        # encode image into latents with vae, requires grad!
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(init_image)
        noise = torch.randn_like(latents)
        start_from_step = torch.tensor([start_from_step], dtype=torch.long, device=self.device)
        latents = self.scheduler.add_noise(latents, noise, start_from_step)

        if num_channels_latents != self.unet.config.in_channels:
            raise ValueError(
            )


        latent_model_input = torch.cat([latents] * 2)
        noise_pred = self.unet(
            latent_model_input, start_from_step, encoder_hidden_states=text_embeddings
        ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        alphas = self.scheduler.alphas_cumprod
        x0_latents = (latents - ((1 - alphas[start_from_step]) ** 0.5) * noise_pred) * (1 / alphas[start_from_step])

        self.vae.to(dtype=torch.float32)
        image = self.decode_latents(x0_latents)
        if output_type == 'pil':
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        return image


    def img_t2i_from_interval(self, 
            prompts, 
            init_image: torch.Tensor,
            height: int=512,
            width: int=512,
            from_step: int=200,
            num_inference_steps=50, 
            guidance_scale=7.5, 
            negative_prompts='', 
            output_type='pil'):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        
        num_channels_latents = self.vae.config.latent_channels
        latents = self.encode_imgs(init_image)
        noise = torch.randn_like(latents)
        from_step = torch.tensor([from_step], dtype=torch.long, device=self.device)
        latents = self.scheduler.add_noise(latents, noise, from_step)

        # decode latents and get image and save
        image_noisy = self.decode_latents(latents)
        image_noisy = image_noisy.cpu().permute(0, 2, 3, 1).float().numpy() 
        image_noisy = self.numpy_to_pil(image_noisy)[0]
        image_noisy.save(f'./test_imgs/noisy.png')
        assert num_channels_latents == self.unet.config.in_channels

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            if t > from_step:
                continue
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        
        self.vae.to(dtype=torch.float32)
        image = self.decode_latents(latents)

        if output_type == 'pil':
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            image = self.numpy_to_pil(image)
        
        return image


    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images



    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()



