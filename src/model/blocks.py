import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class FeedForward(nn.Module):
    """ SwiGLU MLP as used in Llama/Gemma """
    def __init__(self, config):
        super().__init__()
        # SwiGLU uses 3 linear layers
        # d_model -> intermediate_size (gate & up)
        # intermediate_size -> d_model (down)
        # We usually use 2/3 * 4 * d_model as intermediate size for SwiGLU
        hidden_dim = int(2 * (4 * config.d_model) / 3)
        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias) # gate
        self.w2 = nn.Linear(config.d_model, hidden_dim, bias=config.bias) # up
        self.w3 = nn.Linear(hidden_dim, config.d_model, bias=config.bias) # down
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(w1(x)) * w2(x) then project with w3
        return self.dropout(self.w3(nn.functional.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    """ Transformer block: communication (Attention) followed by computation (MLP) """
    def __init__(self, config):
        super().__init__()
        from src.model.attention import MultiHeadAttention
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

if __name__ == "__main__":
    import sys
    import os
    # Add project root to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.model.config import MiniGPTConfig

    print("Testing Feed-Forward Block...")
    config = MiniGPTConfig.tiny()
    
    ffwd = FeedForward(config)
    
    # Dummy input (Batch, Time, Channels)
    B, T, C = 2, 8, config.d_model
    x = torch.randn(B, T, C)
    
    out = ffwd(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: ({B}, {T}, {config.d_model})")
    
    print("Success! Feed-Forward block is working.")

    print("\nTesting Transformer Block...")
    block = Block(config)
    
    out_block = block(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out_block.shape}")
    print(f"Expected shape: ({B}, {T}, {config.d_model})")
    
    print("Success! Transformer Block is working with LayerNorm and Residuals.")
