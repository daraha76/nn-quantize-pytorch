import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
import numpy as np

class FourierBaseEntropyModel(nn.Module):
    # De la Fuente, Alfredo, Saurabh Singh, and Johannes Ballé. "Fourier Basis Density Model." 2024 Picture Coding Symposium (PCS). IEEE, 2024.
    def __init__(
        self,
        dim,
        num_coeff,
        **kwargs
        ):
        super().__init__()
        
        self.input_dim = dim
        
        # Learnable coefficients
        self.num_coeff = num_coeff # N
        self.coeff_params = nn.Parameter(torch.complex(real=torch.rand(dim, num_coeff+1), imag=torch.rand(dim, num_coeff+1)))   #(a_N, ..., a_0) [D, N]
        
        # Scale & offset
        self.scale = nn.Parameter(torch.ones(1, dim, 1))   # s
        self.offset = nn.Parameter(torch.ones(1, dim, 1) * 1.0e-4) # t
    
    def get_coeffs(self, **kwargs):
        """
        Get Fourier Series coefficients (c_0, ..., c_N) from parameters (a_N, ..., a_0)
        """
        # Calculate FS coefficients c_0, ..., c_N using autocorrelation (TODO: faster algorithm?)
        params_pad = F.pad(self.coeff_params, (0, self.num_coeff), value=0) # (a_N, ..., a_0, 0 x N)
        params_conj = torch.conj(self.coeff_params) # (a*_N, ..., a*_0)
        # Group conv with G = D
        # input: [1, D, 2N+1], weight: [D, D/G=1, N]
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html
        fs_coeffs = F.conv1d(input=params_pad.unsqueeze(0), weight=params_conj.unsqueeze(1), bias=False).squeeze(0)  # (c_N, ..., C_0) [D, N]
        fs_coeffs = torch.flip(fs_coeffs, [-1]) # (c_0, ..., C_N) [D, N]
        
        return fs_coeffs

    def coeffs_reg_loss(self, gamma, **kwargs):
        """
        Regularization loss for smoother density
        """
        weights = gamma * 2 * (np.pi * torch.arange(1, self.num_coeff+1).unsqueeze(0))**2  # [1 , N]
        fs_coeffs_sq = torch.abs(self.get_coeffs)**2   # [D, N+1]
        
        return 2 * torch.sum(weights * fs_coeffs_sq[:, 1:]) / self.input_dim

    def cdf(self, x, **kwargs):
        """
        Evalute the channel-wise CDF, P(X < x)
        """
        # Real value support into [-1, 1] support
        x = F.tanh((x - self.offset) / self.scale)  # [BT, D, 1]
        
        # Get FS coefficients
        fs_coeffs = self.get_coeffs()
        
        # Calculate CDF
        cdf = x / 2     # [BT, D, 1]
        x_exp = torch.exp(torch.tensor([1j * np.pi]) * torch.arange(1, self.input_dim+1).reshape(1, 1, -1)) * x  # [BT, D, N]
        norm_coeffs = fs_coeffs / (torch.tensor([1j * np.pi]) * torch.arange(1, self.input_dim+1).reshape(1, -1) * fs_coeffs[:, 0:1])   # [D, N]
        cdf = cdf + norm_coeffs.unsqueeze(0) * x_exp
        cdf = cdf.real  # TODO: is it neccesary?
        
        return cdf
    
    def pmf(self, x_qn, inv_gain=None, **kwargs):
                
        prob = (self.cdf(x_qn + inv_gain/2) - self.cdf(x_qn - inv_gain/2))
        
        return prob

    def information(self, x_qn, inv_gain=None):
        return -torch.log2(torch.clamp(self.pmf(x_qn, inv_gain), min=1e-20))    
    
    def _initial_reshape(self, x):
        # [B, D] -> [B, D, 1] or [B, T, D] -> [BT, D, 1]

        # Check the shape of the input tensor
        assert x.shape[-1] == self.input_dim, "Channel dimension does not match the entropy model's spec."
        
        bsz = x.shape[0]    # Batch size
        num_frames = int(torch.numel(x) / (bsz * self.input_dim))   # Number of frames
        
        # Reshape
        if x.dim() == 2:
            assert num_frames == 1
            x = rearrange(x, 'b d -> b d 1')
            
        x = rearrange(x, 'b f d -> (b f) d 1')

        return x, num_frames

    def _final_reshape(self, x, num_frames):
        # [BT, D, 1] -> [B, T, D] or [B, D, 1] -> [B, D]

        x = rearrange(x, '(b f) d 1 -> b f d', f=num_frames)

        if num_frames == 1:
            x = rearrange(x, 'b d 1 -> b d')

        return x


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

    def get_percentile_interval(self, out_of_bound_pct, inv_gain=None, num_max_range=1000):
        # Calculate bound in integer domain
        # Decide boundary in normalized domain (after multipling inv_gain)
        # out_of_bound_pct: sum of percentage of out-of-bound regions
        
        oob_half = out_of_bound_pct / 200   # in probability
        
        # num_max_range = 6
        # bin_range     = -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5
        # int_range     =     -3   -2   -1   0   1   2   3
        bin_range = torch.arange(-num_max_range/2 - 0.5, num_max_range/2 + 1.5)
        bin_range = torch.tile(bin_range.unsqueeze(1), (1, self.input_dim))    # [max_range+1+1, d]
        bin_range_norm = bin_range * inv_gain
        cdf_per_dim = self.cdf(bin_range_norm)
        
        int_range = torch.arange(-num_max_range/2, num_max_range/2 + 1)
        int_range = torch.tile(int_range.unsqueeze(1), (1, self.input_dim)) # [max_range+1, d]
        int_range_norm = int_range * inv_gain
        pmf_per_dim_full = self.pmf(int_range_norm, inv_gain)
        
        len_cdf = len(cdf_per_dim)
        bound_per_dim = torch.empty((self.input_dim, 2))    # [d, 2]
        actual_oob_prob_per_dim = torch.empty((self.input_dim, 2))    # [d, 2]
        
        pmf_per_dim =torch.zeros((self.input_dim, num_max_range+1))    # [d, max_range+1]
        
        # bin_range     = -3.5 -2.5 -1.5 -0.5 0.5 1.5 2.5 3.5
        #                        o----|
        #                       idx=2 ^
        #              low_idx=1 ^
        # act_oob       = -------|
        #                                      |---o
        #                                idx=4 ^
        #                                upp_idx=5 ^
        # act_oob       =                          o---------
        # int_range     =     -3   -2   -1   0   1   2   3
        # bound         =           ^            ^
        # bound_per_dim =         [-2,           1]
        # pmf           =           ^    ^   ^   ^
        
        for d_i in range(self.input_dim):
            # Find lower bound idx
            low_idx = len_cdf - 1
            for idx in range(len_cdf):
                if cdf_per_dim[idx, d_i] >= oob_half:
                    low_idx = idx - 1 if idx >= 1 else 0
                    actual_oob_prob_per_dim[d_i, 0] = cdf_per_dim[low_idx, d_i]
                    break
            bound_per_dim[d_i, 0] = bin_range[low_idx, 0] + 0.5
            
            # Find upper bound idx
            upp_idx = 0
            for idx in range(len_cdf - 1, -1, -1):
                if cdf_per_dim[idx, d_i] <= 1 - oob_half:
                    upp_idx = idx + 1 if idx < len_cdf - 1 else len_cdf - 1
                    actual_oob_prob_per_dim[d_i, 1] = 1 - cdf_per_dim[upp_idx, d_i]
                    break
            bound_per_dim[d_i, 1] = bin_range[upp_idx, 0] - 0.5

            # Probability table
            # [low_idx, ..., upp_idx, oob_low, oob_high, 0, ..., 0]
            pmf_per_dim[d_i, 0: upp_idx - low_idx] = pmf_per_dim_full[low_idx: upp_idx, d_i]
            pmf_per_dim[d_i, upp_idx - low_idx: upp_idx - low_idx + 2] = actual_oob_prob_per_dim[d_i, :]
        
        # Aggregate probabilities
        used_max_range = int((bound_per_dim[:, 1] - bound_per_dim[:, 0] + 1).max())
        pmf_per_dim = pmf_per_dim[:, :used_max_range + 2]
            
        return bound_per_dim, cdf_per_dim, pmf_per_dim


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
            x = rearrange(x, 'b d -> b d 1')
            
        x = rearrange(x, 'b f d -> (b f) d 1')

        return x, num_frames

    def _final_reshape(self, x, num_frames):
        # [BT, D, 1] -> [B, T, D] or [B, D, 1] -> [B, D]

        x = rearrange(x, '(b f) d 1 -> b f d', f=num_frames)

        if num_frames == 1:
            x = rearrange(x, 'b d 1 -> b d')

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
    def __init__(self, **kwargs):
        super().__init__()
    
    def information(self, **kwargs):
        return None

class NoEM(nn.Module):
    # Empty entropy model
    def __init__(self, **kwargs):
        super().__init__()
    
    def information(self, x_qn, inv_gain=None, **kwargs):
        return torch.Tensor([0]).to(x_qn.device)
    