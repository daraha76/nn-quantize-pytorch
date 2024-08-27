import random
from math import log2
from functools import partial

from typing import List

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from vector_quantize_pytorch.finite_scalar_quantization import FSQ

from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(val):
    return val is not None

def first(l):
    return l[0]

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

# main class

class ResidualFSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        levels: List[int],
        n_codebooks,
        quantizer_dropout = 0.0,
        **kwargs
    ):
        super().__init__()
        codebook_dim = len(levels)

        self.n_codebooks = n_codebooks
        self.levels = levels
        self.codebook_dim = len(levels)

        self.layers = nn.ModuleList([])

        levels_tensor = torch.Tensor(levels)
        
        self.quantizers = nn.ModuleList(
            [
                FSQ(levels=levels, dim=dim, **kwargs)
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

        self.codebook_size = self.quantizers[0].codebook_size

    def forward(
        self,
        z,
        n_active_cb: int = None
    ):

        z_q = 0
        residual = z
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

            z_q_i, indices_i = quantizer(residual)

            # Create mask to apply quantizer dropout
            mask = (torch.full((z.shape[0],), fill_value=i, device=z.device) < n_active_cb)
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=-1)    # [B, T, Nq]

        return z_q, codes # [B, D, T], [B, T, Nq]
        
        '''
        num_quant, quant_dropout_multiple_of, device = self.n_codebooks, self.quantizer_dropout_multiple_of, z.device

        z = self.project_in(z)

        quantized_out = 0.
        residual = first(self.layers).bound(z)

        all_indices = []

        should_quantizer_dropout = self.training and self.quantizer_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices

        if should_quantizer_dropout:
            rand = random.Random(rand_quantizer_dropout_fixed_seed) if exists(rand_quantizer_dropout_fixed_seed) else random

            rand_quantizer_dropout_index = rand.randrange(self.quantizer_dropout_cutoff_index, num_quant)

            if quant_dropout_multiple_of != 1:
                rand_quantizer_dropout_index = round_up_multiple(rand_quantizer_dropout_index + 1, quant_dropout_multiple_of) - 1

            null_indices = torch.full(z.shape[:2], -1., device = device, dtype = torch.long)

        # go through the layers

        with autocast(enabled = False):
            for quantizer_index, (layer, scale) in enumerate(zip(self.layers, self.scales)):

                if should_quantizer_dropout and quantizer_index > rand_quantizer_dropout_index:
                    all_indices.append(null_indices)
                    continue

                quantized, indices = layer(residual / scale)
                quantized = quantized * scale

                residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_indices.append(indices)

        # project out, if needed

        quantized_out = self.project_out(quantized_out)

        # stack all indices

        all_indices = torch.stack(all_indices, dim = -1)

        ret = (quantized_out, all_indices)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers

        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)
        '''
        
        
# grouped residual fsq

class GroupedResidualFSQ(Module):
    def __init__(
        self,
        *,
        dim,
        groups = 1,
        accept_image_fmap = False,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.groups = groups
        assert (dim % groups) == 0
        dim_per_group = dim // groups

        self.accept_image_fmap = accept_image_fmap

        self.rvqs = nn.ModuleList([])

        for _ in range(groups):
            self.rvqs.append(ResidualFSQ(
                dim = dim_per_group,
                **kwargs
            ))

        self.codebook_size = self.rvqs[0].codebook_size

    @property
    def codebooks(self):
        return torch.stack(tuple(rvq.codebooks for rvq in self.rvqs))

    @property
    def split_dim(self):
        return 1 if self.accept_image_fmap else -1

    def get_codes_from_indices(self, indices):
        codes = tuple(rvq.get_codes_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.stack(codes)

    def get_output_from_indices(self, indices):
        outputs = tuple(rvq.get_output_from_indices(chunk_indices) for rvq, chunk_indices in zip(self.rvqs, indices))
        return torch.cat(outputs, dim = self.split_dim)

    def forward(
        self,
        x,
        return_all_codes = False
    ):
        shape, split_dim = x.shape, self.split_dim
        assert shape[split_dim] == self.dim

        # split the feature dimension into groups

        x = x.chunk(self.groups, dim = split_dim)

        forward_kwargs = dict(
            return_all_codes = return_all_codes,
            rand_quantizer_dropout_fixed_seed = random.randint(0, 1e7)
        )

        # invoke residual vq on each group

        out = tuple(rvq(chunk, **forward_kwargs) for rvq, chunk in zip(self.rvqs, x))
        out = tuple(zip(*out))

        # otherwise, get all the zipped outputs and combine them

        quantized, all_indices, *maybe_all_codes = out

        quantized = torch.cat(quantized, dim = split_dim)
        all_indices = torch.stack(all_indices)

        ret = (quantized, all_indices, *maybe_all_codes)
        return ret
