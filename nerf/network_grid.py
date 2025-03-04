import torch
import torch.nn as nn
import torch.nn.functional as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=1,
                 hidden_dim=32,
                 num_layers_bg=2,
                 hidden_dim_bg=16,
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_encoder('hashgrid', input_dim=3, log2_hashmap_size=19, desired_resolution=2048 * self.bound, interpolation='smoothstep', level_dim=4)

        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True)
        self.normal_net = MLP(self.in_dim, 3, hidden_dim, num_layers, bias=True)

        self.density_activation = trunc_exp if self.opt.density_activation == 'exp' else F.softplus

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            
            # use a very simple network to avoid it learning the prompt...
            self.encoder_bg, self.in_dim_bg = get_encoder('frequency', input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    # add a density blob to the scene center
    def density_blob(self, x):
        # x: [B, N, 3]
        # NOTE: Dreamfusion just use blob_iters in the first few epoch, but we use it all the time for better performance
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * (1 - torch.sqrt(d) / self.opt.blob_radius)

        return g

    def common_forward(self, x):

        # if not torch.any(x <= self.bound) or not torch.any(x >= -self.bound):
        #     # raise ValueError('coordinate must be within bound')
        #     import pdb
        #     pdb.set_trace()
        # sigma
        enc = self.encoder(x, bound=self.bound)

        h = self.sigma_net(enc)

        sigma = self.density_activation(h[..., 0] + self.density_blob(x))
        albedo = torch.sigmoid(h[..., 1:])

        return sigma, albedo, enc
    
    def forward(self, x, d, l=None, ratio=1, shading='albedo', soft_light_ratio=0):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        sigma, albedo, enc = self.common_forward(x)

        if shading == 'albedo':
            normal = None
            color = albedo
        
        else: # lambertian shading

            normal = self.normal_net(enc)
            normal = safe_normalize(normal)
            normal = torch.nan_to_num(normal)

            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            # if shading == 'textureless':
            #     color = lambertian.unsqueeze(-1).repeat(1, 3)
            if shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                # NOTE: soft light aug
                # color = soft_light_ratio * albedo * lambertian.unsqueeze(-1)
                color = (soft_light_ratio + (1 - soft_light_ratio) * albedo) * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo, _ = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
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
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.normal_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr / 10})

        return params