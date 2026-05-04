# CV/Resume Highlights: MiniGPT Project

You can copy and adapt these bullet points for your resume to highlight the technical complexity of this project.

### 🚀 Technical Bullet Points

- **Modern Transformer Architecture**: Engineered a Llama-style decoder-only Transformer from scratch in PyTorch, implementing **Rotary Positional Embeddings (RoPE)**, **RMSNorm**, and **SwiGLU** activation functions to match state-of-the-art LLM benchmarks.
- **Hardware-Accelerated Scaling**: Optimized training throughput by integrating **FlashAttention-2** and PyTorch AMP, achieving a 50x speedup (20,000+ tokens/sec) on NVIDIA A100 GPUs.
- **Custom Tokenization Pipeline**: Designed and trained a custom Byte-Pair Encoding (BPE) tokenizer and preprocessed a 10GB FineWeb-Edu dataset into memory-mapped binary streams for high-speed I/O.
- **Cloud-Native Training System**: Built a robust training infrastructure with automated Google Drive checkpoint syncing, multi-epoch validation monitoring, and custom Cosine Decay scheduling.
- **Interactive Inference & Eval**: Developed a mathematical evaluation suite for Perplexity benchmarking and deployed a real-time Gradio web demo with Top-K sampling and Temperature control.

### 🛠️ Keywords for Skills Section
`PyTorch`, `LLM Engineering`, `RoPE`, `RMSNorm`, `SwiGLU`, `FlashAttention-2`, `A100 Scaling`, `BPE Tokenization`, `Transformer Architecture`, `Autoregressive Sampling`, `Gradio`, `Deep Learning Optimization`.
