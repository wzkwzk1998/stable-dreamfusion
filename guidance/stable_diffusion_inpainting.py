from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
import numpy as np

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from tqdm import tqdm
import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusionForInpainting(nn.Module):
    def __init__(self, device):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = device
        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)

        print(f'[INFO] loading stable diffusion...')
                
        # 1. Load the autoencoder model which will be used to decode the latents into image space. 
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="vae", use_auth_token=self.token).to(self.device)

        # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="tokenizer", use_auth_token=self.token)
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="text_encoder", use_auth_token=self.token).to(self.device)

        # 3. The UNet model for generating the latents.
        self.unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="unet", use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference
        # self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps)
        self.scheduler = PNDMScheduler.from_pretrained("stabilityai/stable-diffusion-2-inpainting", subfolder="scheduler", use_auth_token=self.token)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        print(f'[INFO] loaded stable diffusion!')


    def prepare_guidance_condition(self, opt):
        text_z = self.prepare_text_embeddings(opt)
        condition_dict = {
            'text_z': text_z,
        }

        return condition_dict
    
    def prepare_guidance_condition_batch(self, opt, data):
        condition_image = data['rgb_gt']
        assert condition_image.shape[1] == opt.h_guidance and condition_image.shape[2] == opt.w_guidance
        # TODO: mask will be provide in data in the future
        mask = torch.zeros((opt.h_guidance, opt.w_guidance))
        mask[opt.h_guidance // 2 - opt.h_guidance // 4 : opt.h_guidance // 2 + opt.h_guidance // 4,
            opt.w_guidance // 2 - opt.w_guidance // 4 : opt.w_guidance // 2 + opt.w_guidance // 4] = 1
        condition_dict_batch = {
            'condition_image': condition_image.to(self.device),
            'mask': mask.to(self.device),
        }

        return condition_dict_batch

        
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
                
                text_z_single = self.guidance.get_text_embeds([text], [negative_text])
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

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        # map imgs [0, 1] to [-1, 1]
        imgs = 2 * imgs - 1 

        # allow grad!
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents


    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
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


    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance=True
    ):
        mask = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        with torch.no_grad():
            masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = 0.18215 * masked_image_latents

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        mask = mask.repeat(batch_size, 1, 1, 1)
        masked_image_latents = masked_image_latents.repeat(batch_size, 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents


    def prepare_mask_and_masked_image(self, image, mask):
        """
        Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
        converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
        ``image`` and ``1`` for the ``mask``.

        The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
        binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

        Args:
            image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
                It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
                ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
            mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
                It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
                ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


        Raises:
            ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
            should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
            TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
                (ot the other way around).

        Returns:
            tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
                dimensions: ``batch x channels x height x width``.
        """
        if isinstance(image, torch.Tensor):
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

            # Batch single image
            if image.ndim == 3:
                assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
                image = image.unsqueeze(0)

            # Batch and add channel dim for single mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

            # Batch single mask or add channel dim
            if mask.ndim == 3:
                # Single batched mask, no channel dim or single mask not batched but channel dim
                if mask.shape[0] == 1:
                    mask = mask.unsqueeze(0)

                # Batched masks no channel dim
                else:
                    mask = mask.unsqueeze(1)

            assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
            assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
            assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

            # Check image is in [-1, 1]
            if image.min() < -1 or image.max() > 1:
                raise ValueError("Image should be in [-1, 1] range")

            # Check mask is in [0, 1]
            if mask.min() < 0 or mask.max() > 1:
                raise ValueError("Mask should be in [0, 1] range")

            # Binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

            # Image as float32
            image = image.to(dtype=torch.float32)
        elif isinstance(mask, torch.Tensor):
            raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
        else:
            if isinstance(image, PIL.Image.Image):
                image = np.array(image.convert("RGB"))
            image = image[None].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
            if isinstance(mask, PIL.Image.Image):
                mask = np.array(mask.convert("L"))
                mask = mask.astype(np.float32) / 255.0
            mask = mask[None, None]
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        # masked_image: [-1, 1]
        return mask, masked_image


    def train_step(self, 
                    opt,
                    data,
                    condition_dict,
                    pred_rgb:torch.Tensor, 
                    guidance_scale:float=100,
                    lambda_il:float=1.0,
                    target:torch.Tensor=None,
                    lambda_diffusion:float=1e-5):
        '''
        pred_rgb: [0, 1]
        image: [0, 1]
        mask: 0 or 1
        '''        
        # interp to 512x512 to be fed into vae.
        # pred_rgb = F.interpolate(pred_rgb , (height, width), mode='bilinear', align_corners=False)
        
        # assert isinstance(text_embeddings, torch.Tensor)
        # assert pred_rgb.shape[-2] == height and pred_rgb.shape[-1] == width
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
        
        # import pdb
        # pdb.set_trace()
        
        image = condition_dict['condition_image']
        image = image * 2 - 1.0
        image = image.permute(0, 3, 1, 2).contiguous()
        mask = condition_dict['mask']
        # import PIL.Image as Image
        # image_log = image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # print(image_log.shape)
        # Image.fromarray(((image_log + 1) * 127.5).astype('uint8')).save('image_masked.png')
        # Image.fromarray((mask.cpu().numpy() * 255.0).astype('uint8')).save('mask.png')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        num_channels_latents = self.vae.config.latent_channels
        # pred_rgb_copy = pred_rgb.clone().detach()
        # pred_rgb_copy.requires_grad_(True)
        # pred_rgb_il = pred_rgb.clone().detach()
        # pred_rgb_il.requires_grad_(True)
        latents = self.encode_imgs(pred_rgb)

        # encode mask and masked_image to latent
        mask_fr = mask.clone().detach()
        mask, masked_image = self.prepare_mask_and_masked_image(image, mask)
        # print('masked_image [{}, {}]'.format(masked_image.min(), masked_image.max()))
        mask, masked_image_latents = self.prepare_mask_latents(
            mask, 
            masked_image,
            text_embeddings.shape[0] // 2,
            height, 
            width,
            text_embeddings.dtype,
            self.device,
            None
        )

        # Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # predict the noise residual with unet, NO grad!
        # NOTE: pay attention that it is with out grad
        with torch.no_grad():
        # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # TODO: check the function here
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # add the inpainting    
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # optimize for text
        # loss = ((noise_pred - noise) ** 2).mean()
        # loss.backward()

        # optimize for image
        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = (self.alphas[t]) ** 0.5
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)
        # grad = torch.nan_to_num(grad)

        # clip grad for stable training?
        # grad = grad.clamp(-1, 1)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        latents.backward(gradient=grad, retain_graph=True)
        # grad_rgb = pred_rgb_copy.grad
        # grad_rgb = grad_rgb * mask_fr
        # # inpainting loss 
        # # img_loss = inpainting_loss(pred_rgb_il, target, mask_fr)
        # img2mse = lambda x, y : torch.mean((x - y) ** 2)
        # mse_loss = img2mse(pred_rgb_il, target)
        # mse_loss.backward()
        # grad_il = pred_rgb_il.grad
        # grad_il = grad_il * (1 - mask_fr)
        # grad_all = lambda_il * grad_il + lambda_diffusion * grad_rgb
        # pred_rgb.backward(gradient=grad_all, retain_graph=True)
        
        # inpainting mse loss
        return torch.tensor(0, dtype=torch.float32).requires_grad_(True)

    
    def train_step_with_diffusion_grad(self, 
                    text_embeddings, 
                    pred_rgb:torch.Tensor, 
                    image:torch.Tensor, 
                    mask:torch.Tensor, 
                    height=512, 
                    width=512, 
                    guidance_scale:float=7.5,
                    lambda_il:float=1.0,
                    target:torch.Tensor=None,
                    lambda_diffusion:float=1e-5,
                    use_mse_loss:bool=True):
        '''
        This function is backward with diffusion grad
        Args:
            pred_rgb: [0, 1]
            image: [0, 1]
            mask: 0 or 1
            lambda_il: the coefficient of the inpainting loss, use when use_mse_loss is True
            lambda_diffusion: the coefficient of the diffusion loss, use when use_mse_loss is True
            use_mse_loss: set True to use mse loss
        '''        
        # interp to 512x512 to be fed into vae.
        # pred_rgb = F.interpolate(pred_rgb , (height, width), mode='bilinear', align_corners=False)
        assert isinstance(text_embeddings, torch.Tensor)
        assert pred_rgb.shape[-2] == height and pred_rgb.shape[-1] == width

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        num_channels_latents = self.vae.config.latent_channels
        pred_rgb_il = pred_rgb.clone().detach()
        pred_rgb_il.requires_grad_(True)
        latents = self.encode_imgs(pred_rgb)

        # encode mask and masked_image to latent
        mask_fr = mask.clone().detach()
        mask, masked_image = self.prepare_mask_and_masked_image(image, mask)
        # print('masked_image [{}, {}]'.format(masked_image.min(), masked_image.max()))
        mask, masked_image_latents = self.prepare_mask_latents(
            mask, 
            masked_image,
            text_embeddings.shape[0] // 2,
            height, 
            width,
            text_embeddings.dtype,
            self.device,
            None
        )

        # Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )

        # NOTE: in this function we do not ignore the diffusion model grad
        # add noise
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        # pred noise
        latent_model_input = torch.cat([latents_noisy] * 2)
        # TODO: check the function here
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        # add the inpainting    
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        # noise_pred loss backward to get grad
        noise_mse_loss = torch.mean((noise_pred - noise) ** 2)
        
        # not use mse loss
        if not use_mse_loss:
            return noise_mse_loss
        
        # use mse loss for no mask part
        grad_rgb = pred_rgb.grad
        grad_rgb = grad_rgb * mask_fr
        # inpainting mse loss 
        # img_loss = inpainting_loss(pred_rgb_il, target, mask_fr)
        img2mse = lambda x, y : torch.mean((x - y) ** 2)
        mse_loss = img2mse(pred_rgb_il, target)
        mse_loss.backward()
        grad_il = pred_rgb_il.grad
        grad_il = grad_il * (1 - mask_fr)
        # print(grad_il[..., 0, 0])
        # print(grad_rgb[..., 180, 252])
        grad_all = lambda_il * grad_il + lambda_diffusion * grad_rgb
        pred_rgb.backward(gradient=grad_all, retain_graph=True)
        
        return  torch.tensor(0, dtype=torch.float32).requires_grad_()


    @torch.no_grad()
    def img_inpainting(self, prompts, image, mask_image, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, negative_prompts=''):

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]
        
        # Prompts -> text embeds
        text_embeddings = self.get_text_embeds(prompt=prompts, negative_prompt=negative_prompts)

        # Text embeds -> img latents
        self.scheduler.set_timesteps(num_inference_steps)

        # create latents (random noise)
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
                    text_embeddings.shape[0] // 2,
                    num_channels_latents,
                    height,
                    width,
                    text_embeddings.dtype,
                    self.device,
                    None
                    )

        # prepare mask latent variables
        mask, masked_image = self.prepare_mask_and_masked_image(image, mask_image)
        mask, masked_image_latents = self.prepare_mask_latents(
            mask, 
            masked_image,
            text_embeddings.shape[0] // 2,
            height, 
            width,
            text_embeddings.dtype,
            self.device,
            None
        )

        # Check that sizes of mask, masked image and latents match
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    

        print(f'[INFO] sampling!')
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            latent_model_input = torch.cat([latents] * 2)
            
            # in PNDM schedular, is function will return the input directly
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # print(latent_model_input.shape)
            # print(mask.shape)
            # print(masked_image_latents.shape)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        image = self.decode_latents(latents)
        
        return image


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt
    import requests
    import PIL
    from io import BytesIO

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--prompt', type=str)
    # parser.add_argument('--negative', default='', type=str)
    # parser.add_argument('-H', type=int, default=512)
    # parser.add_argument('-W', type=int, default=512)
    # parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--steps', type=int, default=50)
    # opt = parser.parse_args()
    
    seed_everything(opt.seed)

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")


    init_image_path = '/data5/wuzhongkai/proj/dreamfusion_repl/data/llff/nerf_llff_data/flower/images_8/image000.png'
    init_image = PIL.Image.open(init_image_path).convert("RGB").resize((504, 360))
    mask_image = PIL.Image.open('./mask_504x360.png').convert("L")

    init_image = np.array(init_image) / 127.5 - 1.0
    mask_image = np.array(mask_image) / 255
    init_image = torch.Tensor([init_image]).permute(0, 3, 1, 2) 
    mask_image = torch.Tensor([mask_image]).unsqueeze(0)


    device = torch.device('cuda')

    sd = StableDiffusionForInpainting(device)

    prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    imgs = sd.img_inpainting(prompt, image=init_image, mask_image=mask_image, height=360, width=504)
    imgs = imgs[0]


    # visualize image
    PIL.Image.fromarray(((imgs.permute(1, 2, 0).cpu().numpy()) * 255).round().astype('uint8')).save('./yellow_cat_inpainting.png')

    imgs = sd.img_inpainting("", image=init_image, mask_image=mask_image, height=360, width=504)
    imgs = imgs[0]


    # visualize image
    PIL.Image.fromarray(((imgs.permute(1, 2, 0).cpu().numpy()) * 255).round().astype('uint8')).save('./yellow_cat_inpainting_no_prompt.png')



