import os
import sys
import torch
import math

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.config import MiniGPTConfig
from src.model.gpt import MiniGPT
from src.datasets.dataloader import TokenDataLoader

@torch.no_grad()
def evaluate():
    checkpoint_path = os.path.join("checkpoints", "ckpt.pt")
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Please train the model first.")
        return

    # Auto-detect optimal device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    print(f"Loading checkpoint from {checkpoint_path} to {device}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model with saved config
    config = checkpoint['config']
    model = MiniGPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    val_bin = os.path.join("data", "processed", "val.bin")
    if not os.path.exists(val_bin):
        print(f"Validation data not found at {val_bin}. Cannot evaluate perplexity.")
        return

    val_loader = TokenDataLoader(val_bin, device=device)
    
    # Evaluate across a set number of batches
    eval_iters = 50
    batch_size = 8
    
    print(f"Evaluating Perplexity over {eval_iters} batches...")
    
    total_loss = 0.0
    for i in range(eval_iters):
        X, Y = val_loader.get_batch(batch_size, config.seq_len)
        _, loss = model(X, targets=Y)
        total_loss += loss.item()
        
    avg_loss = total_loss / eval_iters
    # Perplexity is exactly defined as exp(CrossEntropyLoss)
    perplexity = math.exp(avg_loss)
    
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Validation Cross-Entropy Loss: {avg_loss:.4f}")
    print(f"Validation Perplexity:       {perplexity:.4f}")
    print(f"--------------------------\n")
    
    if perplexity < 100:
        print("Impressive! The model is highly confident in its predictions (<100 PPL).")
    elif perplexity < 500:
        print("Good! The model has definitely learned basic English grammar and vocabulary.")
    else:
        print("The perplexity is high. The model may need more training time or data.")

if __name__ == "__main__":
    evaluate()
