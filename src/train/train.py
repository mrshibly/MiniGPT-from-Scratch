import os
import sys
import time
import math
import torch

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.config import MiniGPTConfig
from src.model.gpt import MiniGPT
from src.datasets.dataloader import TokenDataLoader
from src.tokenizer.minigpt_tokenizer import MiniGPTTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
out_dir = 'checkpoints'
eval_interval = 250
eval_iters = 20
log_interval = 20
always_save_checkpoint = True

# Data configuration
batch_size = 8 # Lower for 50M model on free Colab GPUs
max_steps = 10000

# Learning Rate configuration
max_lr = 4e-4 
min_lr = max_lr * 0.1
warmup_steps = 200
lr_decay_iters = max_steps

# Device and precision
if torch.cuda.is_available():
    device = "cuda"
    device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    device_type = "mps"
else:
    device = "cpu"
    device_type = "cpu"

print(f"Targeting device: {device}")

# We use bfloat16 for matrix multiplications if supported to save memory and go faster
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# AMP context manager (Automatic Mixed Precision)
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.autocast(device_type=device_type, dtype=ptdtype) if device_type == 'cuda' else torch.autocast(device_type=device_type, dtype=ptdtype, enabled=False) if device_type == 'mps' else torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=False)


# -----------------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)

# Data loaders
train_bin = os.path.join("data", "processed", "train.bin")
val_bin = os.path.join("data", "processed", "val.bin")

if not os.path.exists(train_bin) or not os.path.exists(val_bin):
    raise FileNotFoundError("Missing train.bin or val.bin. Run Phase 1 preprocessing!")

train_loader = TokenDataLoader(train_bin, device=device_type)
val_loader = TokenDataLoader(val_bin, device=device_type)

# Setup Model
config = MiniGPTConfig.standard() # Use 50M param config
model = MiniGPT(config)
model.to(device)

# Load Tokenizer for Generation
try:
    tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer", "tokenizer.json")
    tokenizer = MiniGPTTokenizer(tok_path)
except Exception as e:
    print(f"Warning: Could not load tokenizer for text generation. Text generation will be skipped. ({e})")
    tokenizer = None


# Optimizer and Scaler
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-1, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None

params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {params/1e6:.2f} M | Using Precision: {dtype}")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    """ Runs several batches through train/val to get a stable, less-noisy exact loss. """
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(batch_size, config.seq_len)
            with ctx:
                logits, loss = model(X, targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

@torch.no_grad()
def generate_sample(max_tokens=50):
    if tokenizer is None: return
    model.eval()
    # Start with just the End of Text token as a prompt
    context = torch.tensor([[tokenizer.eot_token_id]], dtype=torch.long, device=device)
    # Generate
    generated_ids = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    # Decode
    text = tokenizer.decode(generated_ids)
    print(f"\n--- GENERATED SAMPLE ---\n{text}\n------------------------\n")
    model.train()

def get_lr(it):
    """ Learning rate scheduler with Warmup and Cosine Decay """
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (lr_decay_iters - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
best_val_loss = 1e9
t0 = time.time()

print("\nStarting Training...")
for step in range(max_steps):

    # 1. Validation & Checkpointing Phase
    if step % eval_interval == 0 or step == max_steps - 1:
        losses = estimate_loss()
        print(f"\nStep {step:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
        # Save exact checkpoint if validation improved
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if step > 0: # don't save the random initialization
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"-> Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                
        # Generate some text to watch the model "learn"
        if step > 0:
            generate_sample()

    # Determine and set the learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Forward & Backward Pass Phase
    X, Y = train_loader.get_batch(batch_size, config.seq_len)
    
    # Cast inner pass to mixed precision (e.g. bfloat16) for speed and memory
    with ctx:
        logits, loss = model(X, targets=Y)
        
    # Scale gradients and step backward
    optimizer.zero_grad(set_to_none=True)
    
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    # 3. Logging Phase
    if step % log_interval == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # Calculate tokens processed per second
        tokens_per_sec = (batch_size * config.seq_len) / dt
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Time: {dt*1000:.2f}ms | Tok/sec: {tokens_per_sec:.2f}")

print("\nDone!")
