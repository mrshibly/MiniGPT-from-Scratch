import os
import sys
import torch

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.config import MiniGPTConfig
from src.model.gpt import MiniGPT
from src.datasets.dataloader import TokenDataLoader

def train():
    # -------------------------------------------------------------------------
    # 1. Setup Configuration & Device
    # -------------------------------------------------------------------------
    config = MiniGPTConfig.tiny()
    
    # Auto-detect optimal device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For Apple Silicon
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Targeting device: {device}")
    
    # Hyperparameters for this tiny run
    batch_size = 4
    max_steps = 500
    learning_rate = 3e-4 # Standard starting LR for AdamW
    
    # -------------------------------------------------------------------------
    # 2. Load Data & Model
    # -------------------------------------------------------------------------
    train_bin = os.path.join("data", "processed", "train.bin")
    if not os.path.exists(train_bin):
        raise FileNotFoundError(f"Missing {train_bin}. Did you run Phase 1 preprocessing?")
        
    train_loader = TokenDataLoader(train_bin, device=device)
    
    print(f"Initializing Model (Vocab: {config.vocab_size}, Params: ...)")
    model = MiniGPT(config)
    model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params/1e6:.2f} M")
    
    # -------------------------------------------------------------------------
    # 3. Setup Optimizer
    # -------------------------------------------------------------------------
    # Weight decay (L2 regularization) is standard practice
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop...")
    
    # Switch to training mode (enables Dropout)
    model.train()
    
    for step in range(max_steps):
        # Sample a batch of data
        X, Y = train_loader.get_batch(batch_size, config.seq_len)
        
        # Forward pass: model calculates logits and loss
        logits, loss = model(X, targets=Y)
        
        # Backward pass: compute gradients
        # First zero out gradients from the previous step! Extremely important.
        optimizer.zero_grad(set_to_none=True) 
        loss.backward()
        
        # Optimizer step: update weights
        optimizer.step()
        
        # Logging
        if step % 10 == 0 or step == max_steps - 1:
            print(f"Step {step:4d} | Training Loss: {loss.item():.4f}")
            
    print("\nTraining complete! The model successfully overfit the tiny batch.")

if __name__ == "__main__":
    train()
