import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# riflex, https://github.com/thu-ml/RIFLEx
def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
):
    '''
        k: the index for the intrinsic frequency in RoPE
        L_test: the number of frames for inference
    '''
    
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=pos.device)[: (dim // 2)].float() / dim)) 

    # === RIFLEx modification start ===
    # Reduce intrinsic frequency to stay within a single period after extrapolation (Eq.(8)).
    # Empirical observations show that a few videos may exhibit repetition in the tail frames.
    # To be conservative, we multiply 0.9 to keep extrapolated length below 90% of a period. 
    freqs[k-1] = 0.9 * 2 * torch.pi / L_test
    # === RIFLEx modification end ===

    freqs = torch.outer(pos, freqs)  
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  
    return freqs_cos, freqs_sin



class FoPERotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_dim = self.config.d_model // self.config.n_heads

        self.dim = self.config.d_model // self.config.n_heads # feature dimension
        self.inv_freq = self.get_inv_freq(self.dim) # get the corresponding frequency for each hidden dimension
   
    def get_inv_freq(self, dim):
            inv_freq = 1.0 / (
                    self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            if self.config.fope is True:
                inv_freq[inv_freq < 2*torch.pi/self.config.max_sequence_length] = 0
                inv_freq = inv_freq[inv_freq != 0.0]

            inv_freq = inv_freq.repeat(self.config.n_heads, 1) # shape: (n_heads, dim//2)
            return inv_freq    

    def get_rotary_embedding(self, seq_len):
        with torch.autocast(device.type, enabled=False):
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # einstein sum, elementise product

            if self.config.fope is True:
                positions = freqs.unsqueeze(0) # output shape (1, h, t, d)
            else:
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # output shape (1, h, t, 2d)

        return positions.sin(), positions.cos()
        
    def rotate_half(self, x):
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
            
        x1, x2 = x.unbind(dim=-2)
        
        return torch.cat((-x2, x1), dim=-1)

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

        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1).to(t.device)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1).to(t.device)

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
    all_len = 30

    output = rotary(input_tensor, all_len)
    return output


def get_fope_paper_fourier_embedding(input_tensor):
    config = ModelConfig(fope=True)
    fourier = FoPEFourierEmbedding(config)
    
    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[2]
    head_dim = config.d_model // config.n_heads
    all_len = 30

    output = fourier(input_tensor, all_len)
    return output