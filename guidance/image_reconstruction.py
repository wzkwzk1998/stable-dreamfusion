import torch


class ImageReconstruction:
    def __init__(self, device) -> None:
        self.device = device
    
    
    def mse_loss(self, target:torch.Tensor, rgb:torch.Tensor) -> torch.Tensor:
        return torch.mean((target - rgb) ** 2)


    def train_step(self, target:torch.Tensor, rgb:torch.Tensor) -> torch.Tensor:
        return self.mse_loss(target, rgb)