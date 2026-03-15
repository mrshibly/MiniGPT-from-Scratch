import os
import sys
import torch
import gradio as gr

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.config import MiniGPTConfig
from src.model.gpt import MiniGPT
from src.tokenizer.minigpt_tokenizer import MiniGPTTokenizer

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the best checkpoint
# Note: User needs to train the model first to have this file!
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "ckpt.pt")
DEFAULT_TOKENIZER = os.path.join("src", "tokenizer", "tokenizer.json")

model = None
tokenizer = None

def load_resources(checkpoint_path):
    global model, tokenizer
    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint not found at {checkpoint_path}. Please train the model first."
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Initialize model and load weights
        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # Load Tokenizer (check multiple paths)
        tok_paths = [
            os.path.join("src", "tokenizer", "tokenizer.json"),
            os.path.join("data", "tokenizer", "tokenizer.json"),
            "tokenizer.json"
        ]
        tokenizer = None
        for p in tok_paths:
            if os.path.exists(p):
                tokenizer = MiniGPTTokenizer(p)
                break
        
        if tokenizer is None:
            return "Model loaded, but Tokenizer file not found. Inference will not work."
        
        return f"Successfully loaded model from {checkpoint_path} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters)."
    except Exception as e:
        return f"Error loading model: {str(e)}"

def generate_text(prompt, max_new_tokens, temperature, top_k):
    if model is None or tokenizer is None:
        return "Please load a model checkpoint first!"
    
    # Encode prompt (ensure it's not empty)
    if not prompt.strip():
        # Use End of Text token as default prompt
        idx = torch.tensor([[tokenizer.eot_id]], dtype=torch.long, device=device)
    else:
        # tokens is likely already a tensor based on MiniGPTTokenizer implementation
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        idx = tokens.unsqueeze(0).to(device) # Add batch dimension -> (1, T)
    
    # Generate
    generated_idx = model.generate(
        idx, 
        max_new_tokens=int(max_new_tokens), 
        temperature=temperature, 
        top_k=int(top_k) if top_k > 0 else None
    )
    
    # Decode
    full_text = tokenizer.decode(generated_idx[0].tolist())
    return full_text

# -----------------------------------------------------------------------------
# Gradio UI definition
# -----------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 MiniGPT: Interactive Demo")
    gr.Markdown("Welcome to the MiniGPT inference interface. Load your trained checkpoint and start generating text!")
    
    with gr.Row():
        ckpt_input = gr.Textbox(label="Checkpoint Path", value=DEFAULT_CHECKPOINT)
        load_btn = gr.Button("Load Model", variant="primary")
    
    status_output = gr.Markdown("*Status: Awaiting model load...*")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Input Prompt", placeholder="Type something to start...", lines=3)
            with gr.Row():
                tokens_count = gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Tokens to Generate")
                temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature")
            top_k = gr.Slider(minimum=0, maximum=200, value=50, step=1, label="Top-K (0 to disable)")
            generate_btn = gr.Button("✨ Generate", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(label="Generated Result", lines=12)

    # Wire up events
    load_btn.click(load_resources, inputs=[ckpt_input], outputs=[status_output])
    generate_btn.click(generate_text, inputs=[prompt, tokens_count, temp, top_k], outputs=[output])

if __name__ == "__main__":
    # Try to load default if it exists
    load_resources(DEFAULT_CHECKPOINT)
    demo.launch()
