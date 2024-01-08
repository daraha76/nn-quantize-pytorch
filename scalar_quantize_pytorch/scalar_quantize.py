import torch
from torch import nn
import torch.nn.functional as F

import scalar_quantize_pytorch.entropy_models as entropy_models
from scalar_quantize_pytorch.entropy_models import FullyFactorizedEntropyModel

class ScalarQuantize(nn.Module):
    def __init__(
        self,
        dim,
        gain=1,
        learnable_gain=False,
        training_q_method='add_noise',
        entropy_model_config=None,
        entropy_loss_ratio=1.0,
        **kwargs,
        ):
        super().__init__()
        
        # Attributes
        self.dim = dim
        
        # Gain
        gain_shape = (dim,)
        gain_vec = torch.full(gain_shape, gain)
        self.learnable_gain = learnable_gain
        if learnable_gain:
            self.gain = nn.Parameter(torch.log(gain_vec), requires_grad=True)
        else:
            self.gain = nn.Parameter(gain_vec, requires_grad=False)
        
        # Uniform noise generator
        self.training_q_method = training_q_method    # 'add_noise', 'ste', 'univ'
        if training_q_method in ['add_noise']:
            self.noise_sampler = torch.distributions.uniform.Uniform(-0.5, 0.5)
        
        # Entropy model
        if entropy_model_config is None:
            self.entropy_model = getattr(entropy_models, "NoEM")()
        else:
            entropy_model_config['dim'] = self.dim
            entropy_model_type = entropy_model_config.pop('type')
            self.entropy_model = getattr(entropy_models, entropy_model_type)(**entropy_model_config)
        
        # Entropy loss
        self.entropy_loss_ratio = entropy_loss_ratio
    
    def get_gain(
        self,
        x=None,
        **kwargs
        ):
        if self.learnable_gain:
            gain = torch.exp(self.gain)     # [D]
        else:
            gain = self.gain                # [D]
            
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
        
        # Apply gain
        x = x * gain

        # Quantization
        if self.training:
            if self.training_q_method == 'add_noise':
                noise = self.noise_sampler.sample(sample_shape=x.size()).to(x.device)
                x_q = x + noise
            elif self.training_q_method == 'ste':
                x_q = torch.round(x)
                x_q = x + (x_q - x).detach()    # Straight-through estimator
            else:
                x_q = torch.round(x)
        else:
            x_q = torch.round(x)

        return x_q

    def inv_quantize(
        self, 
        x_q, 
        inv_gain,
        **kwargs
        ):       
        
        # Apply inverse gain
        x_q_norm = x_q * inv_gain
                    
        return x_q_norm
    
    def forward(
        self,
        x,
        return_info=False,
        **kwargs
        ):
        assert x.shape[-1] == self.dim, "Input must have shape of [B, ..., D]"
        
        # Gain & Inverse Gain
        gain, inv_gain = self.get_gain(x)
        
        # Quantization
        x_q = self.quantize(x, gain)

        # Inverse quantization
        x_q_norm = self.inv_quantize(x_q, inv_gain)
        
        # Evaluate entropy
        info = self.entropy_model.information(x_q_norm, inv_gain=inv_gain)  # [B, ..., D]
        entropy_loss = torch.mean(torch.sum(info, dim=-1)) * self.entropy_loss_ratio
        
        returns = (x_q_norm, entropy_loss)
        if return_info:
            returns = (*returns, info)
            
        return returns