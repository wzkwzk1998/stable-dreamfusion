import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

# TODO: not sure about the details..., the details of the 
class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU(inplace=True)

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def forward(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)
        
    
    def forward(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x

class SigmaNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.skips = [4]
        self.pts_mlp = nn.ModuleList(
            [nn.Linear(self.dim_in, self.dim_hidden)] + \
            [nn.Linear(self.dim_hidden, self.dim_hidden) if i not in self.skips else nn.Linear(self.dim_in + self.dim_hidden, self.dim_hidden) \
                for i in range(num_layers - 1)]
        )
        self.sigma_linear = nn.Linear(self.dim_hidden, self.dim_out)
        
    
    def forward(self, x):
        """
        this function only return sigma and correspoding feature

        Args:
            x (torch.Tensor): pts encode feature.

        Returns:
            torch.Tensor: sigma
            torch.Tensor: feature  
        """
        h = x
        for i, l in enumerate(self.pts_mlp):
            h = self.pts_mlp[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)

        sigma = self.sigma_linear(h)
        return sigma, h

class RgbNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.feature_mlp = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.rgb_mlp = nn.Linear(self.dim_hidden // 2, self.dim_out)
        self.views_mlp = nn.ModuleList([nn.Linear(self.dim_in + dim_hidden, dim_hidden // 2)])


    def forward(self, h, d):
        """
        this function only return sigma and correspoding feature

        Args:
            x (torch.Tensor): pts encode feature.

        Returns:
            torch.Tensor: sigma
            torch.Tensor: feature  
        """
        feature = self.feature_mlp(h)
        h = torch.cat([feature, d], -1)

        for i, l in enumerate(self.views_mlp):
            h = self.views_mlp[i](h)
            h = F.relu(h)
        
        rgb =  self.rgb_mlp(h)
        return rgb


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=8, # 5 in paper
                 hidden_dim=256, # 128 in paper
                 num_layers_bg=3, # 3 in paper
                 hidden_dim_bg=64, # 64 in paper
                 encoding='frequency_torch', # pure pytorch
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=10)
        self.view_encoder, self.view_in_dim = get_encoder(encoding, input_dim=3, multires=4)
        self.sigma_net = SigmaNet(self.in_dim, 1, hidden_dim, num_layers, bias=True, block=ResBlock)
        self.rgb_net = RgbNet(self.view_in_dim, 3, hidden_dim, num_layers, bias=True, block=ResBlock)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def density_blob(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        # g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))
        g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        enc = self.encoder(x, bound=self.bound)

        h = self.sigma_net(enc)

        sigma = F.softplus(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = torch.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):
    
        with torch.enable_grad():
            x.requires_grad_(True)
            sigma, albedo = self.common_forward(x)
            # query gradient
            normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        
        # normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        # normal = torch.nan_to_num(normal)

        return normal
        
    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        '''
        render forward to produce sigma and color, where we invoke common_forward method
        Args:
            x: torch.Tensor, shape [N, 3], in [-bound, bound]
            d: torch.Tensor, shape [N, 3], view direction, nomalized in [-1, 1]
            l: torch.Tensor, shape [3], plane light direction, nomalized in [-1, 1] 
            ratio: scalar, ambient ratio
        '''
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            # sigma, color = self.common_forward(x, d)
            # normal = None
            x_feature = self.encoder(x, bound=self.bound)
            sigma, h = self.sigma_net(x_feature)
            # sigma = trunc_exp(sigma + self.gaussian(x))
            
            d_feature = self.view_encoder(d, bound=self.bound)
            color = self.rgb_net(h, d_feature)
            color = torch.sigmoid(color)
            normal = None

            return sigma, color, normal
        else:
            # query normal

            # sigma, albedo = self.common_forward(x)
            # normal = self.normal(x)
        
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma, albedo = self.common_forward(x)
                # query gradient
                normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
            normal = safe_normalize(normal)
            # normal = torch.nan_to_num(normal)
            # normal = normal.detach()

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        # sigma = self.common_forward(x)
        x_feature = self.encoder(x, bound=self.bound)
        sigma, h = self.sigma_net(x_feature)
        
        return {
            'sigma': sigma,
            # 'albedo': albedo,
        }


    def background(self, x, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params

# class NeRFNetwork(NeRFRenderer):
#     def __init__(self, 
#                  opt,
#                  num_layers=4, # 5 in paper
#                  hidden_dim=96, # 128 in paper
#                  num_layers_bg=2, # 3 in paper
#                  hidden_dim_bg=64, # 64 in paper
#                  encoding='frequency_torch', # pure pytorch
#                  ):
        
#         super().__init__(opt)

#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
#         self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=6)
#         self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True, block=ResBlock)

#         # background network
#         if self.bg_radius > 0:
#             self.num_layers_bg = num_layers_bg   
#             self.hidden_dim_bg = hidden_dim_bg
#             self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
#             self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
#         else:
#             self.bg_net = None

#     def gaussian(self, x):
#         # x: [B, N, 3]
        
#         d = (x ** 2).sum(-1)
#         g = self.opt.blob_density * torch.exp(- d / (self.opt.blob_radius ** 2))

#         return g

#     def common_forward(self, x):
#         # x: [N, 3], in [-bound, bound]

#         # sigma
#         h = self.encoder(x, bound=self.bound)

#         h = self.sigma_net(h)


#         sigma = trunc_exp(h[..., 0] + self.gaussian(x))
#         albedo = torch.sigmoid(h[..., 1:])

#         return sigma, albedo
    
#     # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
#     def finite_difference_normal(self, x, epsilon=1e-2):
#         # x: [N, 3]
#         dx_pos, _ = self.common_forward((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
#         dx_neg, _ = self.common_forward((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
#         dy_pos, _ = self.common_forward((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
#         dy_neg, _ = self.common_forward((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
#         dz_pos, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
#         dz_neg, _ = self.common_forward((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
#         normal = torch.stack([
#             0.5 * (dx_pos - dx_neg) / epsilon, 
#             0.5 * (dy_pos - dy_neg) / epsilon, 
#             0.5 * (dz_pos - dz_neg) / epsilon
#         ], dim=-1)

#         return -normal
    
#     def normal(self, x):
    
#         with torch.enable_grad():
#             x.requires_grad_(True)
#             sigma, albedo = self.common_forward(x)
#             # query gradient
#             normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        
#         # normal = self.finite_difference_normal(x)
#         normal = safe_normalize(normal)
#         # normal = torch.nan_to_num(normal)

#         return normal
        
#     def forward(self, x, d, l=None, ratio=1, shading='albedo'):
#         '''
#         render forward to produce sigma and color, where we invoke common_forward method
#         Args:
#             x: torch.Tensor, shape [N, 3], in [-bound, bound]
#             d: torch.Tensor, shape [N, 3], view direction, nomalized in [-1, 1]
#             l: torch.Tensor, shape [3], plane light direction, nomalized in [-1, 1] 
#             ratio: scalar, ambient ratio
#         '''
#         # x: [N, 3], in [-bound, bound]
#         # d: [N, 3], view direction, nomalized in [-1, 1]
#         # l: [3], plane light direction, nomalized in [-1, 1]
#         # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

#         if shading == 'albedo':
#             # no need to query normal
#             sigma, color = self.common_forward(x)
#             normal = None
        
#         else:
#             # query normal

#             # sigma, albedo = self.common_forward(x)
#             # normal = self.normal(x)
        
#             with torch.enable_grad():
#                 x.requires_grad_(True)
#                 sigma, albedo = self.common_forward(x)
#                 # query gradient
#                 normal = - torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
#             normal = safe_normalize(normal)
#             # normal = torch.nan_to_num(normal)
#             # normal = normal.detach()

#             # lambertian shading
#             lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

#             if shading == 'textureless':
#                 color = lambertian.unsqueeze(-1).repeat(1, 3)
#             elif shading == 'normal':
#                 color = (normal + 1) / 2
#             else: # 'lambertian'
#                 color = albedo * lambertian.unsqueeze(-1)
            
#         return sigma, color, normal

      
#     def density(self, x):
#         # x: [N, 3], in [-bound, bound]
        
#         sigma, albedo = self.common_forward(x)
        
#         return {
#             'sigma': sigma,
#             'albedo': albedo,
#         }


#     def background(self, d):

#         h = self.encoder_bg(d) # [N, C]
        
#         h = self.bg_net(h)

#         # sigmoid activation for rgb
#         rgbs = torch.sigmoid(h)

#         return rgbs

#     # optimizer utils
#     def get_params(self, lr):

#         params = [
#             # {'params': self.encoder.parameters(), 'lr': lr * 10},
#             {'params': self.sigma_net.parameters(), 'lr': lr},
#         ]        

#         if self.bg_radius > 0:
#             # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
#             params.append({'params': self.bg_net.parameters(), 'lr': lr})

#         return params