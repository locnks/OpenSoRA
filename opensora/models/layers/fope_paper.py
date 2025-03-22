import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass
from einops import rearrange, repeat
from typing import Optional, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FoPERotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = self.config.d_model // self.config.n_heads

        self.dim = self.config.d_model // self.config.n_heads # feature dimension
        self.inv_freq = self.get_inv_freq(self.dim) # get the corresponding frequency for each hidden dimension
        # print(self.inv_freq.device, self.inv_freq.dtype)
        # cuda:0 torch.float32

        # manually adjust the frequency value to match opensora rope's freq value
        # unit_tensor = torch.tensor([1.0000e+00, 7.7344e-01, 5.9766e-01, 4.6484e-01, 3.5938e-01, 2.7734e-01,
        # 2.1582e-01, 1.6699e-01, 1.2891e-01, 1.0010e-01, 7.7637e-02, 6.0059e-02,
        # 4.6387e-02, 3.5889e-02, 2.7832e-02, 2.1484e-02, 1.6724e-02, 1.2939e-02,
        # 1.0010e-02, 7.7515e-03, 5.9814e-03, 4.6387e-03, 3.6011e-03, 2.7771e-03,
        # 2.1515e-03, 1.6708e-03, 1.2894e-03, 9.9945e-04, 7.7438e-04, 5.9891e-04,
        # 4.6349e-04, 3.5858e-04, 2.7847e-04, 2.1553e-04, 1.6689e-04, 1.2875e-04]).to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # unit_tensor = torch.load("/home/stud/ghuang/Open-Sora/freqs_before.pt")
        # self.inv_freq = unit_tensor.repeat(16, 1)
   
    def get_inv_freq(self, dim):
            inv_freq = 1.0 / (
                    self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            if self.config.fope is True:
                inv_freq[inv_freq < 2*torch.pi/self.config.max_sequence_length] = 0
                inv_freq = inv_freq[inv_freq != 0.0]

            inv_freq = inv_freq.to(dtype=torch.bfloat16)
            inv_freq = inv_freq.repeat(self.config.n_heads, 1) # shape: (n_heads, dim//2)
            return inv_freq    

    def get_rotary_embedding(self, seq_len):
        with torch.autocast(device.type, enabled=False):
            seq = torch.arange(seq_len, device=device, dtype=torch.bfloat16)
            # print(self.inv_freq.shape) torch.Size([16, 36])
            # [1.0000e+00, 7.7426e-01, 5.9948e-01, 4.6416e-01, 3.5938e-01, 2.7826e-01,                                                                                 
            #  2.1544e-01, 1.6681e-01, 1.2915e-01, 1.0000e-01, 7.7426e-02, 5.9948e-02,                                                                                 
            #  4.6416e-02, 3.5938e-02, 2.7826e-02, 2.1544e-02, 1.6681e-02, 1.2915e-02,                                                                                 
            #  1.0000e-02, 7.7426e-03, 5.9948e-03, 4.6416e-03, 3.5938e-03, 2.7826e-03,                                                                                 
            #  2.1544e-03, 1.6681e-03, 1.2915e-03, 1.0000e-03, 7.7426e-04, 5.9948e-04,                                                                                 
            #  4.6416e-04, 3.5938e-04, 2.7826e-04, 2.1544e-04, 1.6681e-04, 1.2915e-04]
            freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # einstein sum, elementise product

            if self.config.fope is True:
                positions = freqs.unsqueeze(0) # output shape (1, h, t, d)
            else:
                # positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # output shape (1, h, t, 2d)
                # 1.0 7.64 ... 1.2915e 1.0 7.64 ... 1.2915
                # 1.0 1.0 7.64 7.64 -> 
                positions = freqs.repeat_interleave(2, dim=2).unsqueeze(0)
                # positions = torch.load("/home/stud/ghuang/Open-Sora/freqs_after.pt").unsqueeze(0).unsqueeze(0).repeat(1, 16, 1, 1)

                # print(f"HERE, torch.cat, positions={positions}")
        return positions.sin(), positions.cos()
        
    def original_rotate_half(self, x):
        B, nh, T, hs = x.size() # 7200, 16, 30, 72
        x = x.view(B, nh, T, 2, hs // 2) # 7200, 16, 30, 2, 36
            
        x1, x2 = x.unbind(dim=-2)
        
        return torch.cat((-x2, x1), dim=-1)

    def rotate_half(self, x):
        x = rearrange(x, '... (d r) -> ... d r', r = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d r -> ... (d r)')

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, x, all_len):
        # In causal/self-attention, we often cache precomputed positional embeddings for the full sequence (all_len).
        # Instead of recomputing embeddings every time for a different input length, we:
        # Compute them once for all_len.
        # Slice the relevant portion when processing a shorter sequence (x_len).
        with torch.autocast(x.device.type, enabled=False):
            x_len = x.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(all_len)
            pos_sin = pos_sin.type_as(x)
            pos_cos = pos_cos.type_as(x)

            # torch.save(pos_sin, "/home/stud/ghuang/Open-Sora/pos_sin.pt")
            # torch.save(pos_cos, "/home/stud/ghuang/Open-Sora/pos_cos.pt")
            
            x_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, all_len - x_len : all_len, :], 
                pos_cos[:, :, all_len - x_len : all_len, :], 
                x,
            )
            
        return x_.type_as(x)

class FoPEFourierEmbedding(FoPERotaryEmbedding):
    def __init__(self, config):
        super().__init__(config)

        self.input_dim = self.inv_freq.size(-1)
        self.output_dim = self.input_dim if self.input_dim <= self.head_dim//4 else self.head_dim//4

        self.sin_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim),
            requires_grad=False
        ).to(dtype=torch.bfloat16)
        self.cos_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim),
            requires_grad=False
        ).to(dtype=torch.bfloat16)
        torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.rope_fourier_init_norm_gain)
        torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.rope_fourier_init_norm_gain)

        if self.input_dim == self.output_dim:    
            self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device).to(dtype=torch.bfloat16)
            self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device).to(dtype=torch.bfloat16)
        else:
            self.sin_coef += self.get_step_eye(self.sin_coef).to(dtype=torch.bfloat16)
            self.cos_coef += self.get_step_eye(self.cos_coef).to(dtype=torch.bfloat16)
    
        
    def get_step_eye(self, _param):
        _param = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _param[..., i*step, i] = 1.0
        
        return _param
    
    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        pos_sin = pos_sin.to(self.sin_coef.device)
        pos_cos = pos_cos.to(self.cos_coef.device)
        
        # print(pos_sin.dtype, self.sin_coef.dtype)
        # torch.bfloat16 torch.float32

        fourier_sin = torch.einsum("bhtD, hDd -> bhtd", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True))
        fourier_cos = torch.einsum("bhtD, hDd -> bhtd", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True))

        fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim//2-fourier_sin.size(-1)), mode="constant", value=1)
        fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim//2-fourier_cos.size(-1)), mode="constant", value=1)

        # fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1).to(t.device)
        # fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1).to(t.device)

        fourier_sin = fourier_sin.repeat_interleave(2, dim=2).unsqueeze(0)
        fourier_cos = fourier_cos.repeat_interleave(2, dim=2).unsqueeze(0)

        return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)

    
@dataclass
class ModelConfig:
    d_model: int = 1152               # Model dimension
    n_heads: int = 16                # Number of attention heads
    rope_theta: float = 10000.0      # Base for the RoPE frequency computation
    max_sequence_length: int = 30  # Maximum sequence length
    fope: bool = False               # Whether to use Fourier RoPE
    rope_fourier_init_norm_gain: float = 0.3  # Gain for Xavier initialization


def get_fope_paper_rotary_embedding(input_tensor):
    config = ModelConfig(fope=False)
    rotary = FoPERotaryEmbedding(config)
    
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[2]
    head_dim = config.d_model // config.n_heads
    all_len = seq_len

    output = rotary(input_tensor, all_len)
    return output


def get_fope_paper_fourier_embedding(input_tensor):
    config = ModelConfig(fope=True)
    fourier = FoPEFourierEmbedding(config)
    
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[2]
    head_dim = config.d_model // config.n_heads
    all_len = seq_len

    output = fourier(input_tensor, all_len)
    return output