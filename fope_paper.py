import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

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

class FourierEmbedding(RotaryEmbedding):
    def __init__(self, config):
        super().__init__(config)

        self.input_dim = self.inv_freq.size(-1)
        self.output_dim = self.input_dim if self.input_dim <= self.head_dim//4 else self.head_dim//4

        self.sin_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim),
            requires_grad=False
        )
        self.cos_coef = nn.Parameter(
            torch.randn(self.config.n_heads, self.input_dim, self.output_dim),
            requires_grad=False
        )
        torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.rope_fourier_init_norm_gain)
        torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.rope_fourier_init_norm_gain)

        if self.input_dim == self.output_dim:    
            self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
            self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
        else:
            self.sin_coef += self.get_step_eye(self.sin_coef)
            self.cos_coef += self.get_step_eye(self.cos_coef)
    
        
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
    rope_fourier_init_norm_gain: float = 1.0  # Gain for Xavier initialization

def test_rotary_embedding():
    print("Testing RotaryEmbedding...")
    
    # Create a config with standard RoPE
    config = ModelConfig(fope=False)
    
    # Create the rotary embedding layer
    rotary = RotaryEmbedding(config)
    
    # Create sample input tensor
    # [batch_size, n_heads, seq_len, head_dim]
    batch_size = 7200
    seq_len = 30
    head_dim = config.d_model // config.n_heads
    
    x = torch.randn(batch_size, config.n_heads, seq_len, head_dim, device=device)
    
    # Max sequence length to compute positional embeddings for
    all_len = 30
    output = rotary(x, all_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # print(f"Output mean: {output.mean().item():.6f}")
    # print(f"Output std: {output.std().item():.6f}")
    print()


def test_fourier_embedding():
    print("Testing FourierEmbedding...")
    
    # Create a config with Fourier RoPE
    config = ModelConfig(fope=True)
    
    # Add head_dim attribute needed by FourierEmbedding
    FourierEmbedding.head_dim = config.d_model // config.n_heads
    
    # Create the Fourier embedding layer
    fourier = FourierEmbedding(config)
    
    # Create sample input tensor
    # [batch_size, n_heads, seq_len, head_dim]
    batch_size = 7200
    seq_len = 30
    head_dim = config.d_model // config.n_heads
    
    x = torch.randn(batch_size, config.n_heads, seq_len, head_dim, device=device)
    
    # Max sequence length to compute positional embeddings for
    all_len = 30
    
    output = fourier(x, all_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # print(f"Output mean: {output.mean().item():.6f}")
    # print(f"Output std: {output.std().item():.6f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_rotary_embedding()
    test_fourier_embedding()