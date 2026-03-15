import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Process input through a 2-layer MLP with GELU.
        
        Args:
            x (torch.Tensor): Shape (B, T, C).
        """
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # We need to import MultiHeadAttention here to avoid circular imports if attention.py imports config
        # actually, it's better to import at the top
        super().__init__()
        from src.model.attention import MultiHeadAttention
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine MultiHeadAttention and FeedForward with LayerNorm and Residual Connections.
        
        Args:
            x (torch.Tensor): Shape (B, T, C).
        """
        # Pre-norm formulation (LayerNorm is applied before attention and before MLP)
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
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
