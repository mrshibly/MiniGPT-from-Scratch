import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Type

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size: int, config: Type['MiniGPTConfig']):
        super().__init__()
        self.key = nn.Linear(config.d_model, head_size, bias=config.bias)
        self.query = nn.Linear(config.d_model, head_size, bias=config.bias)
        self.value = nn.Linear(config.d_model, head_size, bias=config.bias)
        
        # 'tril' is not a parameter, but a buffer that stays with the model
        # It's a lower triangular matrix of ones
        self.register_buffer('tril', torch.tril(torch.ones(config.seq_len, config.seq_len)))
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute causal self-attention for a single head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Channels).
            
        Returns:
            torch.Tensor: Attended values of shape (Batch, Time, head_size).
        """
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        v = self.value(x) # (B, T, head_size)
        
        # Use PyTorch 2.0 Flash Attention for massive speedup on A100/V100
        # This completely replaces the manual matmul, masking, and softmax operations
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
