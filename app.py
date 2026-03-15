import os
import sys
import torch
import gradio as gr

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model.config import MiniGPTConfig
from src.model.gpt import MiniGPT
from src.tokenizer.minigpt_tokenizer import MiniGPTTokenizer

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Paths adjusted for root-level app.py
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "ckpt.pt")
DEFAULT_TOKENIZER = os.path.join("data", "tokenizer", "tokenizer.json")

model = None
tokenizer = None

def load_resources():
    global model, tokenizer
    if not os.path.exists(DEFAULT_CHECKPOINT):
        return f"❌ Checkpoint not found at {DEFAULT_CHECKPOINT}"
    
    try:
        # Load checkpoint
        checkpoint = torch.load(DEFAULT_CHECKPOINT, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        # Initialize model and load weights
        model = MiniGPT(config)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        
        # Load Tokenizer
        if os.path.exists(DEFAULT_TOKENIZER):
            tokenizer = MiniGPTTokenizer(DEFAULT_TOKENIZER)
        else:
            return f"⚠️ Model loaded, but Tokenizer file not found at {DEFAULT_TOKENIZER}"
        
        return f"✅ Successfully loaded model from {DEFAULT_CHECKPOINT} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters)."
    except Exception as e:
        return f"❌ Error loading model: {str(e)}"

def generate_text(prompt, max_new_tokens, temperature, top_k):
    if model is None or tokenizer is None:
        return "Model and tokenizer are not loaded yet. Please wait..."
    
    if not prompt.strip():
        # Use End of Text token as default prompt
        idx = torch.tensor([[tokenizer.eot_id]], dtype=torch.long, device=device)
    else:
        tokens = tokenizer.encode(prompt, return_tensors="pt")
        idx = tokens.unsqueeze(0).to(device) # Add batch dimension -> (1, T)
    
    # Generate
    with torch.no_grad():
        generated_idx = model.generate(
            idx, 
            max_new_tokens=int(max_new_tokens), 
            temperature=float(temperature), 
            top_k=int(top_k) if top_k > 0 else None
        )
    
    # Decode
    return tokenizer.decode(generated_idx[0].tolist())

# -----------------------------------------------------------------------------
# Gradio UI Design
# -----------------------------------------------------------------------------

custom_css = """
/* Professional Deep Dark Theme */
body {
    background-color: #0d1117;
    color: #c9d1d9;
}

.gradio-container {
    background: radial-gradient(circle at 50% -20%, #1c2128, #0d1117) !important;
    border: none !important;
}

#header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

h1 {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    font-weight: 800;
    font-size: 2.8rem !important;
    background: linear-gradient(90deg, #7928CA, #FF0080);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

.info-text {
    font-size: 1.1rem;
    color: #8b949e;
}

/* Glassmorphism for panels */
.panel-style {
    background: rgba(22, 27, 34, 0.7) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(12px) saturate(180%);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Primary Button Styling */
button.primary-btn {
    background: linear-gradient(135deg, #6e40c9 0%, #d83bd2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.8rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(110, 64, 201, 0.3);
}

button.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(110, 64, 201, 0.5);
    filter: brightness(1.1);
}

/* Slider and Inputs */
.gr-box {
    border-radius: 10px !important;
}

input[type="range"] {
    accent-color: #d83bd2;
}
"""

with gr.Blocks(css=custom_css, title="MiniGPT | Nano-Scale Intelligence") as demo:
    with gr.Column(elem_id="container"):
        with gr.Div(elem_id="header"):
            gr.Markdown("# ⚡ MiniGPT: Nano-Scale Intelligence")
            gr.Markdown("A custom-trained GPT model optimized for efficiency and performance. | [GitHub Repository](https://github.com/mrshibly/MiniGPT-from-Scratch)", elem_classes="info-text")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group(elem_classes="panel-style"):
                    prompt = gr.Textbox(
                        label="Input Prompt", 
                        placeholder="What would you like to ask the model?", 
                        lines=5,
                        info="Type a starting phrase or a question."
                    )
                    
                    with gr.Row():
                        tokens_count = gr.Slider(minimum=1, maximum=512, value=128, step=1, label="Response Length")
                        temp = gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Creativity (Temp)")
                        top_k = gr.Slider(minimum=0, maximum=200, value=50, step=1, label="Focus (Top-K)")
                    
                    generate_btn = gr.Button("✨ Generate Response", variant="primary", elem_classes="primary-btn")
                
                gr.Examples(
                    examples=[
                        ["The future of artificial intelligence is", 128, 0.8, 50],
                        ["Once upon a time in a digital world,", 200, 0.9, 45],
                        ["To be, or not to be, that is the question:", 100, 0.7, 40],
                        ["Python code to sort a list:", 150, 0.1, 10]
                    ],
                    inputs=[prompt, tokens_count, temp, top_k]
                )

            with gr.Column(scale=2):
                with gr.Group(elem_classes="panel-style"):
                    output = gr.Textbox(
                        label="AI Response", 
                        lines=16, 
                        placeholder="Model output will appear here...",
                        show_copy_button=True,
                        interactive=False
                    )
                    status = gr.Markdown("*Status: Ready*", elem_id="status-display")

        # Footer
        gr.Markdown("---")
        gr.Markdown("Built with PyTorch & Gradio. This Space showcases a custom Transformer architecture trained from scratch.", elem_classes="info-text")

    # Wire up events
    generate_btn.click(
        fn=generate_text, 
        inputs=[prompt, tokens_count, temp, top_k], 
        outputs=output,
        api_name="generate"
    )

if __name__ == "__main__":
    load_resources()
    demo.launch()
