import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import scalar_quantize_pytorch.entropy_models as entropy_models
from scalar_quantize_pytorch.entropy_models import FullyFactorizedEntropyModel

class ScalarQuantize(nn.Module):
    def __init__(
        self,
        dim,
        gain=1,
        proj_dim=None,
        learnable_gain=False,
        training_q_method='add_noise',
        inference_q_method='round',
        entropy_model_config=None,
        entropy_loss_ratio=1.0,
        num_rate_option=1,
        cont_rate_train=False,
        is_skip_q=False,
        info_skip_dither=False,
        q_dropout=None,
        **kwargs,
        ):
        super().__init__()
        
        # Projection
        self.dim = dim
        if proj_dim is not None and dim != proj_dim:
            self.proj_dim = proj_dim
            self.is_proj = True
            self.in_proj = weight_norm(nn.Linear(dim, proj_dim))
            self.out_proj = weight_norm(nn.Linear(proj_dim, dim))
        else:
            self.proj_dim = self.dim
            self.is_proj = False
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        
        # Biterate scalable
        self.num_rate_option = num_rate_option
        self.is_skip_q = is_skip_q
        if self.is_skip_q:
            self.q_dropout = 1 / (1 + self.num_rate_option) if q_dropout is None else q_dropout

        # Gain
        if num_rate_option == 1:
            gain_shape = (self.proj_dim,)
            gain_vec = torch.full(gain_shape, gain)
        elif num_rate_option > 1:
            gain_shape = (num_rate_option, self.proj_dim)
            gain_vec = torch.full(gain_shape, gain)

        self.learnable_gain = learnable_gain
        if learnable_gain:
            self.gain = nn.Parameter(torch.log(gain_vec), requires_grad=True)
        else:
            self.gain = nn.Parameter(gain_vec, requires_grad=False)
        
        # Uniform noise generator
        self.training_q_method = training_q_method    # 'add_noise', 'ste', 'univ', 'noq'
        self.inference_q_method = inference_q_method  # 'round', 'univ', 'noq'
        self.noise_sampler = torch.distributions.uniform.Uniform(-0.5, 0.5)
        
        # Entropy model
        if entropy_model_config is None:
            self.entropy_model = getattr(entropy_models, "NoEM")()
        else:
            entropy_model_config['dim'] = self.proj_dim
            entropy_model_type = entropy_model_config.pop('type')
            self.entropy_model = getattr(entropy_models, entropy_model_type)(**entropy_model_config)
        
        # Entropy loss
        if num_rate_option == 1: 
            self.entropy_loss_ratio = entropy_loss_ratio
        elif num_rate_option > 1: 
            assert len(entropy_loss_ratio) == num_rate_option
            self.register_buffer("entropy_loss_ratio", torch.Tensor(entropy_loss_ratio))
        self.cont_rate_train = cont_rate_train        
        self.info_skip_dither = info_skip_dither
    
    def init_gain(self,
        new_gain,
        **kwargs
    ):
        if self.learnable_gain:
                new_gain = torch.log(new_gain)
                
        if self.num_rate_option == 1:
            assert new_gain.dim() == 1
            self.gain.data = new_gain

        elif self.num_rate_option > 1:
            if new_gain.dim() == 1:
                new_gain = new_gain.unsqueeze(0)
                self.gain.data = torch.cat([new_gain for _ in range(self.num_rate_option)], dim=0)  # [#op, D]
            elif new_gain.dim() == 2:
                self.gain.data = new_gain


    def get_gain(
        self,
        x=None,
        rate_option=None,   # int or float (inference), [B] (train)
        **kwargs
        ):
        if rate_option is not None:
            if isinstance(rate_option, int) or (isinstance(rate_option, float) and rate_option >= self.num_rate_option - 1):
                gain = self.gain[int(rate_option)]               # [D]
            elif isinstance(rate_option, float):
                rate_opt_int = int(rate_option)
                rate_opt_rmd = rate_option - rate_opt_int
                gain = torch.pow(self.gain[rate_opt_int], 1 - rate_opt_rmd) * torch.pow(self.gain[rate_opt_int + 1], rate_opt_rmd)
            else: # Rate options in torch.Tensor
                if self.cont_rate_train:
                    rate_opt_int = torch.floor(rate_option).long()  # This will NOT be "num_rate_option-1" (maximum), due to torch.rand()!
                    rate_opt_rmd = rate_option - rate_opt_int
                    gain = torch.pow(self.gain[rate_opt_int], 1 - rate_opt_rmd) * torch.pow(self.gain[rate_opt_int + 1], rate_opt_rmd)
                else: # All entries are int
                    gain = self.gain[rate_option].unsqueeze(1)  # [B, 1, D]
        else:
            gain = self.gain                                # [D]

        if self.learnable_gain:
            gain = torch.exp(gain)
            
        if self.learnable_gain:
            inv_gain = torch.clamp(1 / gain, min=1e-24)
        else:
            inv_gain = 1 / gain
        
        return gain, inv_gain
    
    def quantize(
        self,
        x,
        gain,
        **kwargs
        ):
        
        aux_data_dict = {}
        
        # Apply gain
        x = x * gain

        # Quantization
        if self.training:
            if self.training_q_method == 'add_noise':
                noise = self.noise_sampler.sample(sample_shape=x.size()).to(x.device)
                x_q = x + noise
            elif self.training_q_method == 'ste':   # Rounding with straight-through estimator
                x_q = torch.round(x)
                x_q = x + (x_q - x).detach()    
            elif self.training_q_method == 'univ':  # Universial quantization
                noise = self.noise_sampler.sample(sample_shape=x.shape[:-1]).to(x.device)       # [B, ...]
                noise_shift = torch.stack([noise for d in range(x.shape[-1])], dim=noise.dim()) # [B, ..., D]
                x_q = torch.round(x + noise_shift)
                # x_q = x + (x_q - x).detach()    # STE (TODO: wrong position?)
                aux_data_dict['x_before_round'] = x
                aux_data_dict['noise_shift'] = noise_shift
            elif self.training_q_method == 'noq':
                x_q = x
        else:
            if self.inference_q_method == 'round':
                x_q = torch.round(x)
            elif self.inference_q_method == 'univ':  # Universial quantization
                noise = self.noise_sampler.sample(sample_shape=x.shape[:-1]).to(x.device)       # [B, ...]
                noise_shift = torch.stack([noise for d in range(x.shape[-1])], dim=noise.dim()) # [B, ..., D]
                x_q = torch.round(x + noise_shift)
                aux_data_dict['noise_shift'] = noise_shift
            elif self.inference_q_method == 'noq':
                x_q = x

        return x_q, aux_data_dict

    def inv_quantize(
        self, 
        x_q, 
        inv_gain,
        aux_data_dict=None,
        skip_dither=False,
        **kwargs
        ):      
        if self.training and self.training_q_method == 'univ':
            if not skip_dither:
                x_q = x_q - aux_data_dict['noise_shift']
            x_before_round = aux_data_dict['x_before_round']
            x_q = x_before_round + (x_q - x_before_round).detach()  # STE as if univ_q without scaling is not exist!

        if not self.training and self.inference_q_method == 'univ' and not skip_dither:
            x_q = x_q - aux_data_dict['noise_shift']
        
        # Apply inverse gain
        x_q_norm = x_q * inv_gain
           
        return x_q_norm
    
    def forward(
        self,
        x,
        return_info=False,
        rate_option=None,   # int or [B]
        info_skip_dither=None,
        **kwargs
        ):
        assert x.shape[-1] == self.dim, "Input must have shape of [B, ..., D]"

        if info_skip_dither is None:
            info_skip_dither = self.info_skip_dither
        
        # Quantizer dropout (only during training)
        if self.training and self.is_skip_q:
            dropout_idx = (rate_option == -1)
            _x = x[~dropout_idx]    # Only those will pass quantizer
            _rate_option = rate_option[~dropout_idx]
        else:
            _x = x
            _rate_option = rate_option
        
        # Projection
        xp = self.in_proj(_x)
        
        # Gain & Inverse Gain
        gain, inv_gain = self.get_gain(xp, rate_option=_rate_option)
        
        # Quantization
        xp_q, aux_data_dict = self.quantize(xp, gain)
        
        # Inverse quantization
        if info_skip_dither:
            xp_bar_norm = self.inv_quantize(xp_q, inv_gain, aux_data_dict, skip_dither=True)
            xp_q_norm = xp_bar_norm - aux_data_dict['noise_shift'] * inv_gain
        else:
            xp_q_norm = self.inv_quantize(xp_q, inv_gain, aux_data_dict)
        
        # Projection
        _x_hat = self.out_proj(xp_q_norm)
        
        # Recover dropout
        if self.training and self.is_skip_q:
            x_hat = x
            x_hat[~dropout_idx] = _x_hat
        else:
            x_hat = _x_hat
        
        # Evaluate entropy
        if info_skip_dither:
            info = self.entropy_model.information(xp_bar_norm, inv_gain=inv_gain)  # [B, ..., D]
        else:
            info = self.entropy_model.information(xp_q_norm, inv_gain=inv_gain)  # [B, ..., D]
        if info is not None:
            if self.num_rate_option == 1:
                entropy_loss = torch.mean(torch.sum(info, dim=-1)) * self.entropy_loss_ratio
            elif self.num_rate_option > 1:
                if isinstance(rate_option, float):
                    entropy_loss = None
                else:   # rate_option is torch.Tensor
                    if self.cont_rate_train:
                        rate_opt_int = torch.floor(_rate_option).long()  # This will NOT be "num_rate_option-1" (maximum), due to torch.rand()!
                        rate_opt_rmd = _rate_option - rate_opt_int
                        loss_ratio = torch.pow(self.entropy_loss_ratio[rate_opt_int], 1 - rate_opt_rmd) * torch.pow(self.entropy_loss_ratio[rate_opt_int + 1], rate_opt_rmd)
                        entropy_loss = torch.mean(torch.mean(torch.sum(info, dim=-1), dim=-1) * loss_ratio)
                    else:
                        entropy_loss = torch.mean(torch.mean(torch.sum(info, dim=-1), dim=-1) * self.entropy_loss_ratio[_rate_option])
        else:
            entropy_loss = None
        
        returns = (x_hat, entropy_loss)
        if return_info:
            returns = (*returns, info)
            
        return returns
    
    def encode(
        self,
        x,
        rate_option=None,   # int or [B]
        **kwargs
        ):
        assert x.shape[-1] == self.dim, "Input must have shape of [B, ..., D]"
        
        # Projection
        xp = self.in_proj(x)
        
        # Gain & Inverse Gain
        gain, inv_gain = self.get_gain(xp, rate_option=rate_option)
        
        # Quantization
        xp_q, aux_data_dict = self.quantize(xp, gain)
        
        return xp_q, inv_gain, aux_data_dict
    
    def decode(
        self,
        xp_q,
        inv_gain,
        aux_data_dict,
        return_info=False,
        skip_dither=False,
        info_skip_dither=None,
        rate_option=None,   # int or [B]
        **kwargs
        ):
        if info_skip_dither is None:
            info_skip_dither = self.info_skip_dither

        # Inverse quantization
        if info_skip_dither:
            xp_bar_norm = self.inv_quantize(xp_q, inv_gain, aux_data_dict, skip_dither=True)
            xp_q_norm = xp_bar_norm - aux_data_dict['noise_shift'] * inv_gain
        else:
            xp_q_norm = self.inv_quantize(xp_q, inv_gain, aux_data_dict)
        
        # Projection
        x_hat = self.out_proj(xp_q_norm)
        
        # Evaluate entropy
        if not skip_dither and info_skip_dither:
            info = self.entropy_model.information(xp_bar_norm, inv_gain=inv_gain)  # [B, ..., D]
        else:
            info = self.entropy_model.information(xp_q_norm, inv_gain=inv_gain)  # [B, ..., D]
        if info is not None:
            if self.num_rate_option == 1:
                entropy_loss = torch.mean(torch.sum(info, dim=-1)) * self.entropy_loss_ratio
            elif self.num_rate_option > 1:
                if isinstance(rate_option, float):
                    entropy_loss = None
                else:
                    entropy_loss = torch.mean(torch.mean(torch.sum(info, dim=-1), dim=-1) * self.entropy_loss_ratio[rate_option])
        else:
            entropy_loss = None
        
        returns = (x_hat, entropy_loss)
        if return_info:
            returns = (*returns, info)
            
        return returns
    

    # def analyze_scale(self, x, **kwargs):
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import LogNorm
        
    #     # Gain & Inverse Gain
    #     gain, inv_gain = self.get_gain(x)
        
    #     gain_np = gain.detach().cpu()
    #     gain_np = gain_np.reshape((32, -1))
    #     plt.figure(figsize=(6, 3))
    #     plt.imshow(gain_np)
    #     plt.savefig("/home/bhkim98/MEGABYTE-pytorch/MEGABYTE_pytorch/plots/sbtok_gain")
        
    #     # Quantization
    #     x_q, aux_data_dict = self.quantize(x, gain)
        
    #     # print(torch.max(x.abs()), torch.min(x.abs()))
    #     # print(torch.max(x_q.abs()), torch.min(x_q.abs()))
        
    #     plt.figure(figsize=(6, 3))
    #     x_np = x.squeeze().detach().cpu()
    #     x_mean_np = torch.mean(x_np, dim=0)
    #     x_mean_np = x_mean_np.reshape((32, -1))
    #     plt.figure(figsize=(6, 3))
    #     plt.imshow(np.absolute(x_mean_np) + 1e-16, norm=LogNorm(vmin=1e-16, vmax=1))
    #     plt.savefig("/home/bhkim98/MEGABYTE-pytorch/MEGABYTE_pytorch/plots/sbtok_lat_mean")
        
    #     xq_np = x_q.squeeze().detach().cpu()
    #     xq_mean_np = torch.mean(xq_np, dim=0)
    #     xq_mean_np = xq_mean_np.reshape((32, -1))
    #     plt.figure(figsize=(6, 3))
    #     plt.imshow(np.absolute(xq_mean_np) + 1e-14, norm=LogNorm(vmin=1e-14, vmax=1e+2))
    #     plt.savefig("/home/bhkim98/MEGABYTE-pytorch/MEGABYTE_pytorch/plots/sbtok_lat_q_mean")
    #     exit()

    #     # Inverse quantization
    #     x_q_norm = self.inv_quantize(x_q, inv_gain, aux_data_dict)