print# This source code is licensed under the license found in t LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import functools
import sys
import math
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
from einops import rearrange
from timm.models.vision_transformer import Mlp
from opensora.models.layers.fope_paper import get_fope_paper_fourier_embedding, get_fope_paper_rotary_embedding

from opensora.acceleration.communications import all_to_all, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group

approx_gelu = lambda: nn.GELU(approximate="tanh")


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x



def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ===============================================
# General-purpose Layers
# ===============================================
class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # print(f"Nat: conv3d configuration in_chans={in_chans} embed_dim={embed_dim} kernel_size={patch_size} patch_size={patch_size}")
        # Nat: conv3d configuration in_chans=4 embed_dim=1152 kernel_size=[1, 2, 2] patch_size=[1, 2, 2]
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        # print(f"Nat: D={D}, H={H}, W={W}, self.path_size={self.patch_size} x.shape={x.shape}")
        # Nat: D=30, H=90, W=160, self.path_size=[1, 2, 2] x.shape=torch.Size([2, 4, 30, 90, 160])
        # torch.save(x, "/home/stud/ghuang/Open-Sora/before_x.pt")
        x = self.proj(x)  # (B C T H W)
        # torch.save(x, "/home/stud/ghuang/Open-Sora/x.pt")
        # print(f"Nat: after 3d convolution, x={x.shape}")
        # Nat: after 3d convolution, x=torch.Size([2, 1152, 30, 45, 80])
        if self.norm is not None:
            # print(f"Nat: going through normalization")
            # not going through normalization
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
            # print(f"Nat: self.rotary_emb.shape={self.rotary_emb.shape}")
            # torch.save(self.rotary_emb, "/home/stud/ghuang/Open-Sora/z_rotary_emb.pt")

        self.is_causal = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # print(f"B={B}, N={N}, C={C}")
        # B=60, N=3600, C=1152
        # or
        # B=7200, N=30, C=1152 => T=30, C=1152, batch_size=2, spatial=45x80=3600
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # print(f"query.shape={q.shape}, key.shape={k.shape}, value.shape={v.shape}")
        # spatial block: query.shape=torch.Size([60, 16, 3600, 72]), key.shape=torch.Size([60, 16, 3600, 72]), value.shape=torch.Size([60, 16, 3600, 72])                                                                                                       
        # temporal block: query.shape=torch.Size([7200, 16, 30, 72]), key.shape=torch.Size([7200, 16, 30, 72]), value.shape=torch.Size([7200, 16, 30, 72]) 
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            # the condition goes into here
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                # print(q.shape, k.shape)
                # torch.Size([7200, 16, 30, 72]) torch.Size([7200, 16, 30, 72])
                # q = self.rotary_emb(q)
                # k = self.rotary_emb(k)

                with open("/home/stud/ghuang/Open-Sora/causal_mask_ratio", "r") as f:
                    content = f.read()
                    content = content.split('-=-')
                    embedding_type = content[4]

                if embedding_type == "rope":
                    q = get_fope_paper_rotary_embedding(q)
                    k = get_fope_paper_rotary_embedding(k)
                else:
                    q = get_fope_paper_fourier_embedding(q)
                    k = get_fope_paper_fourier_embedding(k)

        if enable_flash_attn: # spatial block is using flash attention
            from flash_attn import flash_attn_func
            # print(f"Using flash attention, using causal mask={self.is_causal}, softmax scale={self.scale}, self.training={self.training}")
            # Using flash attention, using causal mask=False, softmax scale=0.11785113019775792, self.training=False
            # Using flash attention, using causal mask=False
            # it is using flash attention
            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )

            # print(f"q.permute.shape = {q.permute(0, 2, 1, 3).shape}, k.permute.shape={k.permute(0, 2, 1, 3).shape}")
            # q.permute.shape = torch.Size([60, 16, 3600, 72]), k.permute.shape=torch.Size([60, 16, 3600, 72]) 
            # here the spatial causal mask should not be added, as it would only influence the cross attention of each single frame
            # what we want is actually restricting the information leakage across frames

            # spatial_causal_mask = torch.full((60, 16, 3600, 3600), float('-inf'), dtype=torch.bfloat16, device="cuda")
            # spatial_causal_mask[:, :, :1800, :1800] = 0
            # spatial_causal_mask[:, :, 1800:, 1800:] = 0

            # DONE: whether the error would accumulate after so many blocks and layers' computation of scaled_dot_product_attention
            # here, the causal mask is not not added on temporal attention
            # we can't achieve the result
            # DONE: think about which dimension we should add the causal mask, td, or block level, or temporal block, etc.

            # hidden_states = F.scaled_dot_product_attention(
                # q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=spatial_causal_mask, dropout_p=0.0, is_causal=False, scale=self.scale
            # )

            # print(hidden_states.shape, hidden_states.permute(0, 2, 1, 3).shape)
            # torch.Size([60, 16, 3600, 72]) torch.Size([60, 3600, 16, 72])

            # hidden_states = hidden_states.permute(0, 2, 1, 3)

            # print("the shape of the tensor", x.shape, hidden_states.shape)
            # print("Max difference:", torch.max(torch.abs(x - hidden_states)), "Mean value difference:", torch.mean(torch.abs(x - hidden_states)))
            # x = hidden_states
        else:
            # print("Not enabling flash attention...") temporal block is not using flash attention
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                print("Using causal mask in self-attention...")
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                # Returns the lower triangular part of the matrix (2-D tensor) or batch
                # of matrices input, the other elements of the result tensor out are set to 0.
                # The argument diagonal controls which diagonal to consider. If diagonal = 0, 
                # all elements on and below the main diagonal are retained
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            # print(attn.shape)
            # torch.Size([7200, 16, 30, 30])
            temporal_causal_mask = torch.full((7200, 16, 30, 30), float('-inf'), dtype=torch.bfloat16, device="cuda")
            temporal_causal_mask[:, :, :15, :15] = 0
            temporal_causal_mask[:, :, 15:, 15:] = 0
            # add fusion mask
            # leakage mask
            # temporal_causal_mask[:, :, 16:, :15] = 0
            # temporal_causal_mask[:, :, :14, 15:] = 0

            # rec 15 mask
            # temporal_causal_mask[:, :, 15:25, :15] = 0
            # temporal_causal_mask[:, :, 5:15, 15:] = 0

            # rotated_rec_mask
            # temporal_causal_mask = torch.full((7200, 16, 30, 30), float('-inf'), dtype=torch.bfloat16, device="cuda")
            # temporal_causal_mask[:, :, :15, 5:] = 0
            # temporal_causal_mask[:, :, 15:, :25] = 0

            # rec mask
            # temporal_causal_mask[:, :, 15:20, :15] = 0
            # temporal_causal_mask[:, :, 10:15, 15:] = 0

            # rec 10 mask
            # temporal_causal_mask[:, :, 15:20, 5:15] = 0
            # temporal_causal_mask[:, :, 10:15, 15:25] = 0

            # enable temporal causal mask
            attn += temporal_causal_mask

            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class KVCompressAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        sampling="conv",
        sr_ratio=1,
        mem_eff_attention=False,
        attn_half=False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        self.sampling = sampling
        if sr_ratio > 1 and sampling == "conv":
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1 / sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mem_eff_attention = mem_eff_attention
        self.attn_half = attn_half

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == "uniform_every":
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == "ave":
            tensor = F.interpolate(tensor, scale_factor=1 / scale_factor, mode="nearest").permute(0, 2, 3, 1)
        elif sampling == "uniform":
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == "conv":
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        new_N = N
        H, W = HW
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype
        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, H, W, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, H, W, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        q, k = self.q_norm(q), self.k_norm(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )

        elif self.mem_eff_attention:
            attn_bias = None
            if mask is not None:
                attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
                attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            # (B, N, #heads, #dim) -> (B, #heads, N, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            if not self.attn_half:
                attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SeqParallelAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
    ) -> None:
        assert rope is None, "Rope is not supported in SeqParallelAttention"
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            enable_flash_attn=enable_flash_attn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # for sequence parallel here, the N is a local sequence length
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape)

        sp_group = get_sequence_parallel_group()

        # apply all_to_all to gather sequence and split attention heads
        # [B, SUB_N, 3, NUM_HEAD, HEAD_DIM] -> [B, N, 3, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        qkv = all_to_all(qkv, sp_group, scatter_dim=3, gather_dim=1)

        if self.enable_flash_attn:
            qkv_permute_shape = (
                2,
                0,
                1,
                3,
                4,
            )  # [3, B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]
        else:
            qkv_permute_shape = (
                2,
                0,
                3,
                1,
                4,
            )  # [3, B, NUM_HEAD_PER_DEVICE, N, HEAD_DIM]
        qkv = qkv.permute(qkv_permute_shape)

        # ERROR: Should qk_norm first
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        if not self.enable_flash_attn:
            x = x.transpose(1, 2)

        # apply all to all to gather back attention heads and split sequence
        # [B, N, NUM_HEAD_PER_DEVICE, HEAD_DIM]  -> [B, SUB_N, NUM_HEAD, HEAD_DIM]
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)

        # reshape outputs back to [B, N, C]
        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        # print(f"Nat: multheadcross attention d_model={d_model}")
        # Nat: multheadcross attention d_model=1152
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # print(f"Nat: multiheadcrossattention x.shape {x.shape} cond.shape {cond.shape} heads {self.num_heads} head_dim {self.head_dim}")
        # print(f"Nat: multiheadcrossattention query {q.shape} key {k.shape} values {v.shape}")
        # Nat: multiheadcrossattention x.shape torch.Size([2, 28800, 1152]) cond.shape torch.Size([1, 42, 1152]) heads 16 head_dim 72
        # Nat: multiheadcrossattention query torch.Size([1, 57600, 16, 72]) key torch.Size([1, 42, 16, 72]) values torch.Size([1, 42, 16, 72])
        # notice that the 57600=28800*2 is because the batch size is 2, also, 42 means the prompt tokens length is 21, ie, 42/2
        # here the 28800=2*90*160, spatial locations size 90*160
        # the 2 is related to num_seconds, for num_second=4, it is 2. For num_second_8, it is 4. The query dimension is always 2xthe dimension of x because batch_size is 2.
        # the 42 is the number of tokens of the prompt.
        attn_bias = None
        if mask is not None:
            with open("/home/stud/ghuang/Open-Sora/causal_mask_ratio", "r") as f:
                content = f.read()
                content = content.split('-=-')
                prompt_key = content[0]
                token_boundary = int(content[1])
                denoising_step_ratio = float(content[3])
                denoising_step_condition = content[2]

            half_video = N // 2
            half_tokens = token_boundary
            q_attn_bias_input = [half_video] * (B * 2)
            kv_attn_bias_input = [half_tokens, mask[0] - half_tokens, half_tokens, mask[0] - half_tokens]
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(q_attn_bias_input, kv_attn_bias_input)
            with open("/home/stud/ghuang/Open-Sora/denosing_step_information", "r") as f:
                denoising_step = int(f.read())
                if denoising_step_condition == "_causal_mask_experiment_2_":
                    if denoising_step >= denoising_step_ratio * 30:
                        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
                else:
                    if denoising_step < denoising_step_ratio * 30:
                        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            # print(f"N={N}, B={B}, mask={mask}") # the mask shape is token_lenght//2
            # N=108000, B=2, mask=[16, 16]
            # print(f"q={q.shape}, k={k.shape}, v={v.shape}")
            # q=torch.Size([1, 216000, 16, 72]), k=torch.Size([1, 32, 16, 72]), v=torch.Size([1, 32, 16, 72])  
            # print(f"Nat: attention_bias shape is {attn_bias}")
            # Nat: attention_bias shape is BlockDiagonalMask(q_seqinfo=_SeqLenInfo(seqstart=tensor([
            # 0, 108000, 216000], dtype=torch.int32), max_seqlen=108000, min_seqlen=108000, seqstart_py=[0,
            # 108000, 216000]), k_seqinfo=_SeqLenInfo(seqstart=tensor([ 0, 16, 32], dtype=torch.int32),
            # max_seqlen=16, min_seqlen=16, seqstart_py=[0, 16, 32]), _batch_sizes=None)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        # print(f"Nat: after multiheadcrossattention x {x.shape}")
        # Nat: after multiheadcrossattention x torch.Size([1, 216000, 16, 72])
        # Nat: after multiheadcrossattention x torch.Size([1, 57600, 16, 72])
        # equivalent code: (ref https://github.com/facebookresearch/xformers/blob/9a59df217473fc949871792a5c5fb7f274335959/xformers/ops/fmha/__init__.py#L224)
        # scale = 1.0 / query.shape[-1] ** 0.5
        # query = query * scale
        # query = query.transpose(1, 2)
        # key = key.transpose(1, 2)
        # attn = query @ key.transpose(-2, -1)
        # if attn_bias is not None:
        #     attn = attn + attn_bias
        # attn = attn.softmax(-1)
        # attn = F.dropout(attn, p)
        # attn = attn @ value
        # return attn.transpose(1, 2).contiguous()

        def calculate_attention(query, key, attn_bias, p):
            scale = 1.0 / query.shape[-1] ** 0.5
            query = query * scale
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            attn = query @ key.transpose(-2, -1)
            # print(f"Nat: here the attn.shape is {attn.shape} query.shape {query.shape} key.transpose.shape {key.transpose(-2, -1).shape}")
            # Nat: here the attn.shape is torch.Size([1, 16, 216000, 32]) query.shape torch.Size([1, 16, 216000, 72]) key.transpose.shape torch.Size([1, 16, 72, 32])
            # for tensor that are higher than 2-dimensions
            # the last two dimension are treated as matrices while the other dimensions are treated
            # like batches
            # 1. If the dimensions are equal, they remain in the output.
            # 2. If one of the dimensions is 1, it's "stretched" (broadcasted) to match the other dimension.
            # 3. If neither dimension is 1 and they are not equal, you get a broadcasting error.
            if attn_bias is not None:
               attn = attn + attn_bias.materialize(attn.shape).to(attn.device)
            # print(f"Nat: before softmax the attention={attn.shape}")
            attn = attn.softmax(-1)
            attn = F.dropout(attn, p)
            # here, the attn.shape is torch.Size([1, 16, 216000, 32]) for 1 card
            return attn
        # attention.shape torch.Size([1, 16, 216000, 32])

        attention = calculate_attention(q, k, attn_bias, self.attn_drop.p)
        attention = attention[0]
        attention = attention.mean(dim=0)
        # with open("/home/stud/ghuang/Open-Sora/tmp", "r") as f:
        #     content = f.read()
        #     content = content.split("-")
        #     temporal_block_condition = content[0]
        #     temporal_block_ratio = float(content[1])
        # torch.save(attention, f"/mnt/data8/liao/ghuang/visualization/attn_maps_egg_rock_remove_temporal_block_rope_{temporal_block_condition}_{str(temporal_block_ratio)}/" + datetime.now().strftime("%H-%M-%S-%f") + ".pt")

        # print(f"Nat: multiheadcrossattention calculated attention shape {attention.shape}")
        # 4 cards: Nat: multiheadcrossattention calculated attention shape torch.Size([1, 16, 57600, 42])
        # 1 card: Nat: multiheadcrossattention calculated attention shape torch.Size([1, 16, 216000, 42])
        # print(f"Nat: inside multiheadcrossattention, B={B} N={N} C={C} mask={mask}")
        # Nat: inside multiheadcrossattention, B=2 N=108000 C=1152 mask=[16, 16]
        x = x.view(B, -1, C)
        # print(f"Nat: before view x.shape={x.shape}")
        # Nat: before view x.shape=torch.Size([2, 108000, 1152])
        x = self.proj(x)
        # print(f"Nat: after proj x.shape={x.shape}")
        # Nat: after proj x.shape=torch.Size([2, 108000, 1152])
        x = self.proj_drop(x)
        # print(f"Nat: multiheadcrossattention output x {x.shape}")
        # Nat: multiheadcrossattention output x torch.Size([2, 28800, 1152])
        # 16 heads, each head dimension is 72, so 1152. But notice, that the 1152 is the input dimension
        # which is designed to be able to be splitted evenly by 16 heads.
        return x


class SeqParallelMultiHeadCrossAttention(MultiHeadCrossAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        sp_group = get_sequence_parallel_group()
        sp_size = dist.get_world_size(sp_group)
        # print(f"Nat: sp_group={sp_group} sp_size={sp_size}")
        # the sp_size=4 when using 4 cards
        B, SUB_N, C = x.shape  # [B, TS/p, C]
        N = SUB_N * sp_size

        # shape:
        # q, k, v: [B, SUB_N, NUM_HEADS, HEAD_DIM]
        q = self.q_linear(x).view(B, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        kv = split_forward_gather_backward(kv, get_sequence_parallel_group(), dim=3, grad_scale="down")
        k, v = kv.unbind(2)

        # print(f"Nat: before all_to_all q.shape {q.shape} k.shape {k.shape} v.shape {v.shape}")
        # Nat: before all_to_all q.shape torch.Size([2, 28800, 16, 72]) k.shape torch.Size([1, 42, 4, 72]) v.shape torch.Size([1, 42, 4, 72])

        # apply all_to_all to gather sequence and split attention heads
        q = all_to_all(q, sp_group, scatter_dim=2, gather_dim=1)
        # # Suppose you have a tensor of shape [batch_size, 128, 256]
        # With 4 processes, each process initially has [batch_size, 32, 256]
        # After all_to_all, each process will have [batch_size, 128, 256]
        # combining all processes together, it will be 128*4=512, which is 4 times larger than original 128
        # print(f"Nat: after all_to_all q.shape {q.shape} k.shape {k.shape} v.shape {v.shape}")
        # Nat: after all_to_all q.shape torch.Size([2, 115200, 4, 72]) k.shape torch.Size([1, 42, 4, 72]) v.shape torch.Size([1, 42, 4, 72])

        q = q.view(1, -1, self.num_heads // sp_size, self.head_dim)
        k = k.view(1, -1, self.num_heads // sp_size, self.head_dim)
        v = v.view(1, -1, self.num_heads // sp_size, self.head_dim)

        # print(f"Nat: seqparallelmultiheadcrossattention x.shape {x.shape} cond.shape {cond.shape}")
        # Nat: seqparallelmultiheadcrossattention x.shape torch.Size([2, 28800, 1152]) cond.shape torch.Size([1, 42, 1152])
        # print(f"Nat: seqparallelmultiheadcrossattention query {q.shape} key {k.shape} values {v.shape}")
        # Nat: seqparallelmultiheadcrossattention query torch.Size([1, 230400, 4, 72]) key torch.Size([1, 42, 4, 72]) values torch.Size([1, 42, 4, 72])
        # remeber the input shape for STDiT is (2,4,30,90,160) where 4 is the channel, 30 is the denoising steps
        # and 16*90*160=230400, but where does 16 come from?
        # first of all, batch_size=2, so 2*90*60=28800. Then in temporal, the size is 57600. Here
        # the num_seconds=4, so the dimension for spatial is 57600*4=230400 while for temporal it is 57600.
        # here the (4,72), the last two dimensions are self.num_heads, self.head_dim
        # compute attention
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)

        def calculate_attention(query, key, attn_bias, p):
            scale = 1.0 / query.shape[-1] ** 0.5
            query = query * scale
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            attn = query @ key.transpose(-2, -1)
            # if attn_bias is not None:
            #    attn = attn + attn_bias
            attn = attn.softmax(-1)
            attn = F.dropout(attn, p)
            return attn
        attention = calculate_attention(q, k, attn_bias, self.attn_drop.p)
        # print(f"Nat: sequence parallel multihead cross attention shape={attention.shape}")
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        # apply all to all to gather back attention heads and scatter sequence
        x = x.view(B, -1, self.num_heads // sp_size, self.head_dim)
        x = all_to_all(x, sp_group, scatter_dim=1, gather_dim=2)
        # print(f"Nat: after all_to_all x.shape={x.shape}")
        # apply output projection
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(f"Nat: seqparallelmultiheadcrossattention output x {x.shape}")
        # Nat: seqparallelmultiheadcrossattention output x torch.Size([2, 28800, 1152])
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            # modulation: x_modulated​=scale⋅x_normalized​+shift
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x) # hidden_size C=1152, num_patch=4, out_channels=8
        return x


# ===============================================
# Embedding Layers for Timesteps and Class Labels
# ===============================================


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        # print(f"Nat: frequency_embedding_size={self.frequency_embedding_size} hidden_size={self.hidden_size}")
        # Nat: frequency_embedding_size=256 hidden_size=1152

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # print(f"Nat: the dim={dim}")
        # Nat: the dim=256
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        # print(f"Nat: args={args} args.shape={args.shape}")
        # args.shape=torch.Size([2, 128])
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        # print(f"Nat embedding={embedding} embedding.shape={embedding.shape}")
        # embedding.shape=torch.Size([2, 256])
        # torch.save(embedding, "/home/stud/ghuang/Open-Sora/z_timestep_embedding" + datetime.now().strftime("%H-%M-%S-%f") + ".pt")
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs // s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        # self.dim=1152
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim)) # 10000^(-x), x = [0, 0.045, 0.9, ..., 1]
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        print(f"Nat: positionalembedding2d inv_freq={inv_freq}")

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq) # inv_freq.shape=(288), t.shape=(3600)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ): # h=45, w=80, base_size=round((H*W)**0.5)
        grid_h = torch.arange(h, device=device) / scale # length 45
        grid_w = torch.arange(w, device=device) / scale # length 80
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w, # horizontal axis, the columns
            grid_h, # vertical axis, the rows
            indexing="ij",
        )  # here w goes first, grid_h, grid_w becomes 45x80 matrix
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        # print(f"Nat: grid_h={grid_h.shape}, grid_w={grid_w.shape}")
        # Nat: grid_h=torch.Size([3600]), grid_w=torch.Size([3600])
        # torch.save(grid_h, "/home/stud/ghuang/Open-Sora/z_grid_h.pt")
        # torch.save(grid_w, "/home/stud/ghuang/Open-Sora/z_grid_w.pt")
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        positional_embedding = torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)
        # print(f"Nat: emd_h.shape={emb_h.shape} emd_w.shape={emb_w.shape} positional_embedding.shape={positional_embedding.shape}")
        # Nat: emd_h.shape=torch.Size([3600, 576]) emd_w.shape=torch.Size([3600, 576]) positional_embedding.shape=torch.Size([1, 3600, 1152])
        # torch.save(positional_embedding, "/home/stud/ghuang/Open-Sora/z_positional_embedding" + datetime.now().strftime("%H-%M-%S-%f") + ".pt")
        return positional_embedding

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)


# ===============================================
# Sine/Cosine Positional Embedding Functions
# ===============================================
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
