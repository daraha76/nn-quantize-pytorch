import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack

class FullyFactorizedEntropyModel(nn.Module):
    def __init__(
        self,
        dim,
        num_hidden,
        num_layer,
        **kwargs
        ):
        super().__init__()
        
        self.input_dim = dim
        
        # Layers
        h_in_list = [1] + [num_hidden] * num_layer
        h_out_list = [num_hidden] * num_layer + [1]

        self.mapping_blocks = nn.ModuleList()

        for i, (h_in, h_out) in enumerate(zip(h_in_list, h_out_list)):
            is_final = False if i != num_layer else True
            self.mapping_blocks.append(
                CDFMappingBlock(self.input_dim, h_in, h_out, is_final))

    def cdf(self, x, **kwargs):
        """
        Evalute the channel-wise CDF, P(X < x)

        Args:
            x (Tensor) [B, D] or [B, T, D]: Input vector

        Returns:
            c (Tensor) [B, D] or [B, T, D]: Cumulative probability

        """
  
        # Initial reshape
        h, num_frames = self._initial_reshape(x)

        # Concecutive mapping
        for block in self.mapping_blocks:
            h = block(h)

        # Final reshape
        c = self._final_reshape(h, num_frames)

        return c

    def pdf(self, x, **kwargs):
        """
        Evalute the channel-wise PDF, P(X = x)

        Args:
            x (Tensor) [B, D] or [B, T, D]: Input vector

        Returns:
            p (Tensor) [B, D] or [B, T, D]: Probability

        """

        # Initial reshape
        x, num_frames = self._initial_reshape(x)

        # Jacobian matrix multiplication
        p = None 

        for block in self.mapping_blocks:
            # Jacobian
            jacobian = block.jacobian(x)

            if p is None:
                p = jacobian
            else:
                p = torch.matmul(p, jacobian)

            # Forward propagation
            x = block(x)

        # Final reshape
        p = p.squeeze(-1)   # [B, D, 1, 1] ---> [B, D, 1] or [BT, D, 1, 1] ---> [BT, D, 1]
        p = self._final_reshape(p, num_frames)
            
        return p
            

    def pmf(self, x_qn, inv_gain=None, **kwargs):
                
        prob = (self.cdf(x_qn + inv_gain/2) - self.cdf(x_qn - inv_gain/2))
        
        return prob

    def information(self, x_qn, inv_gain=None):
        return -torch.log2(torch.clamp(self.pmf(x_qn, inv_gain), min=1e-20))

    @torch.no_grad()
    def get_distribution(self,
            target='pdf', xmin=-10, xmax=10,
            num_points=1000, device=torch.device('cpu')):

        x_range = torch.tile(torch.linspace(xmin, xmax, num_points).unsqueeze(1),
                            (1, self.input_dim)).to(device)

        if target == 'pdf':
            dist = self.pdf(x_range)
            
        elif target == 'cdf':
            dist = self.cdf(x_range)
            
        elif target == 'pmf':                
            discrete_num_points = int((xmax - xmin) / inv_gain) + 1
            x_range = torch.tile(torch.linspace(xmin, xmax, discrete_num_points).unsqueeze(1),
                            (1, self.input_dim)).to(device)
            dist = self.pmf(x_range)
            
        else: raise Exception('Target should be pdf, cdf, or pmf')

        x_range = x_range.cpu().numpy()
        dist = dist.cpu().numpy()

        return x_range, dist

    def _initial_reshape(self, x):
        # [B, D] -> [B, D, 1] or [B, T, D] -> [BT, D, 1]

        # Check the shape of the input tensor
        assert x.shape[-1] == self.input_dim, "Channel dimension does not match the entropy model's spec."
        
        bsz = x.shape[0]    # Batch size
        num_frames = int(torch.numel(x) / (bsz * self.input_dim))   # Number of frames
        
        # Reshape
        if x.dim() == 2:
            assert num_frames == 1
            x = rearrange(x, 'b d -> b 1 d')
            
        x = rearrange(x, 'b f d -> (b f) d 1')

        return x, num_frames

    def _final_reshape(self, x, num_frames):
        # [BT, D, 1] -> [B, T, D] or [B, D, 1] -> [B, D]

        x = rearrange(x, '(b f) d 1 -> b f d', f=num_frames)

        if num_frames == 1:
            x = rearrange(x, 'b 1 d -> b d')

        return x


class CDFMappingBlock(nn.Module):
    def __init__(self, num_c, h_in, h_out, is_final=False, cdf_param_means=[0.0, 0.0, 0.0], cdf_param_scales=[0.1, 0.1, 0.1]):
        super().__init__()

        # Attributes
        self.is_final = is_final

        # Asserts
        if is_final:
            assert h_out == 1, "Output of the final block should be a scalar instead of a vector"
        
        # Parameters
        self.h = nn.Parameter(torch.normal(cdf_param_means[0], cdf_param_scales[0],
                              size=(num_c, h_in, h_out)),
                              requires_grad=True)   # [C, H_in, H_out]
        self.b = nn.Parameter(torch.normal(cdf_param_means[1], cdf_param_scales[1],
                              size=(num_c, 1, h_out)),
                              requires_grad=True)   # [C, 1, H_out]

        # Nonlinearity
        if not is_final:
            self.a = nn.Parameter(torch.normal(cdf_param_means[2], cdf_param_scales[2],
                                  size=(num_c, 1, h_out)),
                                  requires_grad=True)   # [C, 1, H_out]

    def forward(self, x):
        """
        Evalute the mapping function for computing CDF
        (Note: each channel is treated independetly)

        Args:
            x (Tensor) [B, C, H_in]: Function input

        Returns:
            y (Tensor) [B, C, H_out]: Function output

        """

        # (Initial) Reshape
        x = x.transpose(0, 1)   # [B, C, H_in] ---> [C, B, H_in]

        # Channel-wise affine transform with the reparameterization
        y = torch.bmm(x, F.softplus(self.h)) + self.b   # [C, B, H_out]

        # Nonlinear activation
        y = self.act(y)   # [C, B, H_out]

        # (Final) Reshape
        y = y.transpose(0, 1)   # [C, B, H_out] ---> [B, C, H_out]

        return y

    def jacobian(self, x):
        """
        Evalutate the JACOBIAN of the mapping function for computing PDF
        (Note: each channel is treated independetly)

        Args:
            x (Tensor) [B, C, H_in]: Function input

        Returns:
            jacobian (Tensor) [B, C, H_in, H_out]: Jacobian matrix

        """

        # (Initial) Reshape
        x = x.transpose(0, 1)   # [B, C, H_in] ---> [C, B, H_in]

        # Channel-wise affine transform with the reparameterization
        y = torch.bmm(x, F.softplus(self.h)) + self.b   # [C, B, H_out]

        # Nonlinear activation
        y = self.act_prime(y)   # [C, B, H_out]

        # Get Jacobian matrix
        diag = torch.diag_embed(y)   # [C, B, H_out, H_out]
        tiled_w = torch.tile(F.softplus(self.h).unsqueeze(1), (1, diag.size(1), 1, 1))   # [C, B, H_in, H_out]

        jacobian = torch.matmul(tiled_w, diag)   # [C, B, H_in, H_out]

        # (Final) Reshape
        jacobian = jacobian.transpose(0, 1)   # [C, B, H_in, H_out] ---> [B, C, H_in, H_out]

        return jacobian

    def act(self, x):
        """
        Apply the nonlinear function

        Args:
            x (Tensor) [C, B, H_out]: Nonlinear function input
        
        Returns:
            g (Tensor) [C, B, H_out]: Nonlinear function output

        """

        if not self.is_final:
            g = x + torch.tanh(self.a) * torch.tanh(x)
        else:
            g = torch.sigmoid(x)

        return g

    def act_prime(self, x):
        """
        Apply the derivative of the nonlinear function

        Args:
            x (Tensor) [C, B, H_out]: Nonlinear function input
        
        Returns:
            g_prime (Tensor) [C, B, H_out]: Nonlinear function output

        """

        if not self.is_final:
            g_prime = 1 + torch.tanh(self.a) * (1 - torch.tanh(x) ** 2)
        else:
            g_prime = torch.sigmoid(x) * (1 - torch.sigmoid(x))

        return g_prime

class Hyperprior(nn.Module):
    # TODO
    def __init__(**kwargs):
        1
    
    def information(**kwargs):
        return None

class NoEM(nn.Module):
    # Empty entropy model
    def __init__(**kwargs):
        1
    
    def information(**kwargs):
        return None
    