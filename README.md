---
title: MiniGPT-from-Scratch
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.16.2
app_file: app.py
pinned: false
---

# MiniGPT-from-Scratch 🚀

A high-performance, from-scratch implementation of a modern decoder-only Transformer language model (~124M Parameters, GPT-2 Small scale) trained on the **10 GB FineWeb-Edu dataset**.

This flagship portfolio project transitions classic Transformer designs into modern Llama/GPT-3 style architectures, engineered for maximum GPU throughput and low-latency inference.

---

## 🌟 Key Engineering Highlights

* **Vectorized Multi-Head FlashAttention-2**: Fully parallelized attention computations using PyTorch `scaled_dot_product_attention` (SDPA) with **110,000+ tokens/sec** throughput on NVIDIA RTX GPUs.
* **Modern Architectural Elements**:
  * **Rotary Positional Embeddings (RoPE)** for dynamic position encoding.
  * **RMSNorm (Pre-Layer Normalization)** for training stability.
  * **SwiGLU Activation Functions** in MLP feed-forward blocks for improved parameter efficiency.
* **Custom Byte-Pair Encoding (BPE) Pipeline**: Trained custom BPE tokenizer and memory-mapped binary dataset streams (`uint16`) for zero-overhead streaming I/O.
* **Blackwell Architecture Support**: Custom PyTorch Nightly Integration with CUDA 12.8 support for next-gen NVIDIA hardware (sm_120 / RTX 50-series).
* **Interactive Demo & Benchmarking**: Built-in Gradio web application with real-time temperature control and top-k sampling.

---

## 🛠️ Tech Stack

* **Language**: Python 3.11+
* **Deep Learning Framework**: PyTorch (AMP `bfloat16` Mixed Precision & TF32)
* **Tokenization**: Hugging Face `tokenizers` (Custom BPE)
* **Web UI / Demo**: Gradio
* **Dataset**: FineWeb-Edu (10 GB Corpus)

---

## 📁 Clean Project Structure

```text
MiniGPT-from-Scratch/
├── data/
│   ├── raw/              # Streamed dataset files
│   ├── processed/        # Memory-mapped binary token streams (train.bin, val.bin)
│   └── tokenizer/        # Custom BPE vocabulary & tokenizer configs
├── src/
│   ├── datasets/         # Ingestion, cleaning, and memory-mapping loaders
│   ├── model/            # Transformer architecture (RoPE, RMSNorm, SwiGLU, Attention)
│   ├── tokenizer/        # BPE training scripts and wrappers
│   └── train/            # AMP training loop with Cosine Decay & checkpointing
├── checkpoints/          # Saved model checkpoints (.pt)
├── run_124m.ps1          # Automated high-performance PowerShell GPU launcher
├── app.py                # Interactive Gradio Web UI
└── requirements.txt      # Project dependencies
```

---

## 🚀 Quick Start & Usage

### 1. Installation

```bash
git clone https://github.com/mrshibly/MiniGPT-from-Scratch.git
cd MiniGPT-from-Scratch
pip install -r requirements.txt
```

### 2. Dataset Preparation (10 GB FineWeb-Edu)

Run the automated ingestion pipeline to download, clean, tokenize, and encode binary streams:

```bash
python src/datasets/download_fineweb.py --max_gb 10.0
python src/datasets/clean_text.py
python src/tokenizer/train_tokenizer.py
python src/datasets/prepare_data.py
```

### 3. Launch Model Training (~124M Parameters)

On Windows with NVIDIA GPU, run the optimized PowerShell launcher:

```powershell
.\run_124m.ps1
```

Or execute directly via Python:

```bash
python src/train/train.py
```

### 4. Interactive Web Demo

Once a training checkpoint is saved in `checkpoints/ckpt.pt`:

```bash
python app.py
```

---

## 📊 Model Specifications

| Parameter | Specification | Note |
| :--- | :--- | :--- |
| **Total Parameters** | **~124 Million** | GPT-2 Small scale |
| **Layers (`n_layers`)** | **12** | Transformer blocks |
| **Attention Heads (`n_heads`)** | **12** | Vectorized multi-head SDPA |
| **Embedding Dim (`d_model`)** | **768** | Hidden representation size |
| **Context Length (`seq_len`)** | **512** | Context window |
| **Vocabulary Size** | **16,384** | Custom BPE Tokenizer |
| **Precision** | **bfloat16 / TF32** | PyTorch Autocast AMP |

---

## 📄 License

MIT License
