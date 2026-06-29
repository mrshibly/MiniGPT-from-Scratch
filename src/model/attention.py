import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Type

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x shape: (B, T, n_heads, head_dim)
    # freqs_cis shape: (T, head_dim // 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:x.shape[1], :].view(1, x.shape[1], 1, -1)
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_rotated.type_as(x)

class MultiHeadAttention(nn.Module):
    """ Fully vectorized multi-head self-attention with RoPE and PyTorch Flash Attention """

    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_size = config.d_model // config.n_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Precompute RoPE frequencies for vector head size
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.head_size, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention in parallel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Multi-head attended values of shape (B, T, C).
        """
        B, T, C = x.shape
        
        # Project and reshape Q, K, V for all heads in parallel: (B, T, n_heads, head_size)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_size)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_size)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_size)
        
        # Apply Rotary Embeddings across all heads simultaneously
        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)
        
        # Transpose for PyTorch Scaled Dot Product Attention (SDPA): (B, n_heads, T, head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        
        # Transpose back and combine heads into (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.model.config import MiniGPTConfig

    print("Testing Vectorized Multi-Head Attention...")
    config = MiniGPTConfig.tiny()
    mha = MultiHeadAttention(config)
    
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)
    out_mha = mha(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of heads: {config.n_heads}")
    print(f"Output shape: {out_mha.shape}")
    print("Success! Vectorized multi-head attention is working.")
