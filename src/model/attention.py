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

class Head(nn.Module):
    """ one head of self-attention with RoPE """
    def __init__(self, head_size: int, config: Type['MiniGPTConfig']):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(config.d_model, head_size, bias=config.bias)
        self.query = nn.Linear(config.d_model, head_size, bias=config.bias)
        self.value = nn.Linear(config.d_model, head_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_freqs_cis(head_size, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        # Reshape for RoPE: (B, T, 1, head_size)
        q = q.view(B, T, 1, self.head_size)
        k = k.view(B, T, 1, self.head_size)
        
        # Apply Rotary Embeddings
        q = apply_rotary_emb(q, self.freqs_cis)
        k = apply_rotary_emb(k, self.freqs_cis)
        
        # Reshape back for attention: (B, T, head_size)
        q = q.view(B, T, self.head_size)
        k = k.view(B, T, self.head_size)

        # Use PyTorch 2.0 Flash Attention
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config):
        super().__init__()
        head_size = config.d_model // config.n_heads
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Multi-head attended values of shape (B, T, C).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

if __name__ == "__main__":
    import sys
    import os
    # Add project root to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.model.config import MiniGPTConfig

    print("Testing Single Attention Head...")
    config = MiniGPTConfig.tiny()
    head_size = config.d_model // config.n_heads
    
    head = Head(head_size, config)
    
    # Dummy input (Batch, Time, Channels)
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)
    
    out = head(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Head size: {head_size}")
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: ({B}, {T}, {head_size})")
    
    print("Success! Causal self-attention head is working.")

    print("\nTesting Multi-Head Attention...")
    mha = MultiHeadAttention(config)
    
    out_mha = mha(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of heads: {config.n_heads}")
    print(f"Output shape: {out_mha.shape}")
    print(f"Expected shape: ({B}, {T}, {config.d_model})")
    
    print("Success! Multi-head attention is working.")
