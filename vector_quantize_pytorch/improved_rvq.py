from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm

from dac.nn.layers import WNConv1d

# Code referenced from : https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py
"""
MIT License

Copyright (c) 2023-present, Descript

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class ImprovedVQ(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, dim: int, codebook_size: int, codebook_dim: int, 
        cb_loss_weight:float = 1., commitment_weight:float = 1., **kwargs):
        
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Projection for factorized codes
        self.in_proj = weight_norm(nn.Conv1d(dim, codebook_dim, kernel_size=1))
        self.out_proj = weight_norm(nn.Conv1d(codebook_dim, dim, kernel_size=1))
        
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        
        self.cb_loss_weight = cb_loss_weight
        self.commitment_weight = commitment_weight

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_eq, indices = self.decode_latents(z_e)

        commit_loss = self.commitment_weight * F.mse_loss(z_e, z_eq.detach(), reduction="none").mean([1, 2])
        codebook_loss = self.cb_loss_weight * F.mse_loss(z_eq, z_e.detach(), reduction="none").mean([1, 2])

        z_eq = z_e + (z_eq - z_e).detach()  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_eq)

        return z_q, indices, commit_loss, codebook_loss  # [B, D, T], [B, T], [B], [B]

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ImprovedRVQ(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        cb_loss_weight:float = 1., 
        commitment_weight:float = 1.,
        **kwargs
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                ImprovedVQ(dim, codebook_size, codebook_dim[i], cb_loss_weight, commitment_weight)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout
        
        self.cb_loss_weight = cb_loss_weight
        self.commitment_weight = commitment_weight

    def forward(self, z, n_active_cb: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_active_cb : int, optional
            No. of RVQ layers to use
            (n_active_cb < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z_q" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "commit_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        z_q = 0
        residual = z
        commit_loss = 0
        codebook_loss = 0
        codebook_indices = []

        if n_active_cb is None:
            n_active_cb = self.n_codebooks
        if self.training:
            n_active_cb = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_active_cb[:n_dropout] = dropout[:n_dropout]
            n_active_cb = n_active_cb.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_active_cb:
                break

            z_q_i, indices_i, commit_loss_i, codebook_loss_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_active_cb)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commit_loss += (commit_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()

            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=-1)    # [B, T, Nq]

        return z_q, codes, commit_loss, codebook_loss # [B, D, T], [B, T, Nq], ...


class ImprovedRVQList(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    """

    def __init__(
        self,
        dim: int = 512,
        n_quantizers: int = 32,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        cb_loss_weight:float = 1., 
        commitment_weight:float = 1.,
        **kwargs
    ):
        super().__init__()
        
        self.n_quantizers = n_quantizers
        
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers_list = nn.ModuleList(
            [
                ImprovedRVQ(dim, codebook_size, n_codebooks, codebook_dim[i], cb_loss_weight, commitment_weight)
                for i in range(n_quantizers)
            ]
        )
        self.quantizer_dropout = quantizer_dropout
        
        self.cb_loss_weight = cb_loss_weight
        self.commitment_weight = commitment_weight

    def forward(self, z, n_active_cb: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x G x D x T] (G = n_quantizers)
        n_active_cb : int, optional
            No. of quantizers to use
            (n_active_cb < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z_q" : Tensor[B x G x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "commit_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """
        commit_loss = 0
        codebook_loss = 0
        codebook_indices = []
        
        z_q = []
        codes = []

        if n_active_cb is None:
            n_active_cb = self.n_codebooks
            
        # Process with each RVQ in the self.quantizers_list
        for qg_i in self.n_quantizers:
            z_q_g, codes_g, commit_loss_g, codebook_loss_g = self.quantizers_list[qg_i](z[:, qg_i], n_active_cb)

            z_q.append(z_q_g)       # List of [B, D, T]
            codes.append(codes_g)   # List of [B, T ,Nq]
            commit_loss += commit_loss_g
            codebook_loss += codebook_loss_g

        # Output aggregation
        z_q = torch.stack(z_q, dim=1)       # [B, G, D, T]
        codes = torch.stack(codes, dim=1)   # [B, G, T, Nq]

        return z_q, codes, commit_loss, codebook_loss # [B, D, T], [B, T, Nq], ...