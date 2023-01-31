import torch
import numpy as np
import torch.nn as nn

class ImageReconstruction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_fn = lambda pred_rgb, rgb_gt: torch.mean((pred_rgb - rgb_gt) ** 2)
    
    def train_step(self, 
                    opt,
                    data,
                    condition_dict,
                    pred_rgb:torch.Tensor):

        rgb_gt = data['rgb_gt']
        loss = self.loss_fn(pred_rgb, rgb_gt)
        
        return loss
        
    def prepare_guidance_condition(self, opt):
        return {}
    
    def prepare_guidance_condition_batch(self, opt, data):
        return {}
        