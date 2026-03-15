import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.nn import functional as F
from src.model.config import MiniGPTConfig
from src.model.blocks import Block

class MiniGPT(nn.Module):
    """ The full GPT Language Model, with a context size of config.seq_len. """
    
    def __init__(self, config: MiniGPTConfig):
        """
        Initializes the MiniGPT model.

        Args:
            config (MiniGPTConfig): Configuration object containing model parameters.
        """
        super().__init__()
        assert config.vocab_size is not None
        assert config.seq_len is not None
        self.config = config

        # The core Transformer module dictionary
        self.transformer = nn.ModuleDict(dict(
            # Token Embedding (Vocabulary Size x Hidden Dimension)
            wte = nn.Embedding(config.vocab_size, config.d_model),
            
            # Positional Embedding (Sequence Length x Hidden Dimension)
            wpe = nn.Embedding(config.seq_len, config.d_model),
            
            # Dropout attached to the embeddings
            drop = nn.Dropout(config.dropout),
            
            # The core stack of Transformer blocks
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            
            # The final layer normalization
            ln_f = nn.LayerNorm(config.d_model),
        ))
        
        # The Language Modeling Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying:
        # GPT-2 shares the weights between the token embedding and the final projection layer.
        # This massively reduces parameters and generally improves performance because
        # tokens map to the same mathematical feature space symmetrically.
        self.transformer.wte.weight = self.lm_head.weight

        # This initializes weights to a scaled normal distribution
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        GPT-2 style initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Perform a forward pass through the model.
        
        Args:
            idx (torch.Tensor): LongTensor of shape (B, T) containing token indices.
            targets (torch.Tensor, optional): LongTensor of shape (B, T) containing target token indices.
            
        Returns:
            logits (torch.Tensor): Tensor of shape (B, T, vocab_size) with word scores.
            loss (torch.Tensor, optional): Scalar CrossEntropy loss if targets are provided.
        """
        device = idx.device
        b, t = idx.size()
        
        # Ensure sequence length is valid
        assert t <= self.config.seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.seq_len}"

        # 1. Generate position ids: [0, 1, 2, ..., T-1], shape (T)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # 2. Extract Token Embeddings: shape (B, T, d_model)
        tok_emb = self.transformer.wte(idx) 
        
        # 3. Extract Positional Embeddings: shape (T, d_model) -> broadcasting handles batch B
        pos_emb = self.transformer.wpe(pos) 
        
        # 4. Add them together and apply dropout
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 5. Pass through the sequence of Transformer blocks
        for block in self.transformer.h:
            x = block(x)
            
        # 6. Apply final LayerNorm
        x = self.transformer.ln_f(x)
        
        # 7. Project back to vocabulary size
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # PyTorch cross_entropy expects (B*T, C) shape for 2D inputs
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (B, T)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        
        Args:
            idx (torch.Tensor): Seed sequence of shape (B, T).
            max_new_tokens (int): Number of tokens to append.
            temperature (float): Scaling factor for the logits (lower = more deterministic).
            top_k (int, optional): If set, only sample from the top k most likely tokens.
            
        Returns:
            torch.Tensor: The sequence with generated tokens appended, shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at seq_len
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

if __name__ == "__main__":
    # Small test script
    print("Testing the Full MiniGPT Model...")
    
    # 1. Load the tiny config
    config = MiniGPTConfig.tiny()
    print(f"Config loaded: Vocab={config.vocab_size}, SeqLen={config.seq_len}, D_model={config.d_model}, Layers={config.n_layers}")
    
    # 2. Instantiate the blank model
    model = MiniGPT(config)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    # 3. Create a dummy sequence shaped (Batch, Time)
    B, T = 2, 16 
    # Create random integers simulating token IDs
    dummy_input = torch.randint(0, config.vocab_size, (B, T)) 
    dummy_targets = torch.randint(0, config.vocab_size, (B, T)) 
    
    print(f"Input shape: {dummy_input.shape}")
    
    # 4. Forward pass
    logits, loss = model(dummy_input, targets=dummy_targets)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Expected shape: ({B}, {T}, {config.vocab_size})")
    print(f"Calculated Loss: {loss.item():.4f}")
    
    print("\nTesting Generation Loop...")
    # Give it a 4-token prompt context
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    print(f"Prompt shape: {prompt.shape}")
    
    # Generate 20 new tokens with temperature and top_k
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    print(f"Expected shape: (1, 24)")
    
    print("\nSuccess! The Full Transformer successfully generated logits, calculated loss, and generated text autoregressively.")

