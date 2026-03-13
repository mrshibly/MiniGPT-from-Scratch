from dataclasses import dataclass

@dataclass
class MiniGPTConfig:
    """
    Configuration for the MiniGPT model.
    Contains hyperparameters for the Transformer.
    """
    vocab_size: int = 32000      # Size of the vocabulary
    seq_len: int = 512           # Maximum sequence length (context window)
    d_model: int = 512           # Dimension of the hidden embeddings
    n_layers: int = 12           # Number of transformer blocks
    n_heads: int = 8             # Number of attention heads
    dropout: float = 0.1         # Dropout probability
    bias: bool = False           # Whether to use biases in Linear layers. (False is "modern" GPT style)

    @classmethod
    def tiny(cls):
        """
        A tiny configuration for local debugging and fast compilation.
        ~10M parameters.
        """
        return cls(
            vocab_size=16384,
            seq_len=128,
            d_model=256,
            n_layers=4,
            n_heads=4
        )

    @classmethod
    def main(cls):
        """
        The main target configuration for the 30-day curriculum.
        ~50M parameters.
        """
        return cls(
            vocab_size=32000,
            seq_len=512,
            d_model=512,
            n_layers=12,
            n_heads=8
        )

    @classmethod
    def stretch(cls):
        """
        A larger configuration to test scaling on cloud GPUs.
        ~124M parameters (GPT-2 Small size).
        """
        return cls(
            vocab_size=32000, # or 50257 for exact GPT-2
            seq_len=512,
            d_model=768,
            n_layers=12,
            n_heads=12
        )
