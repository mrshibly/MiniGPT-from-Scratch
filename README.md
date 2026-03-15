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

A from-scratch implementation of a decoder-only Transformer (GPT) language model, built in 30 days. This project is designed as a deep-dive into LLM engineering, following the architecture of GPT-2.

## 🌟 Features
- **Custom BPE Tokenizer**: Fully trained on the dataset using Hugging Face `tokenizers`.
- **GPT Architecture**: Multi-head causal self-attention, GELU MLPs, LayerNorm (Pre-Norm), and residual connections.
- **Optimized Training**: Supports PyTorch AMP (Mixed Precision), Cosine Decay with Warmup, and weight tying.
- **Interactive Demo**: Built-in Gradio web app for real-time text generation.
- **Evaluation**: Integrated perplexity calculation for model benchmarking.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch
- **Tokenization**: Hugging Face Tokenizers
- **Interface**: Gradio
- **Data**: FineWeb-Edu (Sample)

## 📁 Project Structure
```text
MiniGPT/
├── data/               # Raw and processed datasets
├── notebooks/          # Colab/Kaggle training templates
├── src/
│   ├── datasets/       # Data loading and preprocessing logic
│   ├── model/          # Transformer architecture (GPT, Attention, Blocks)
│   ├── tokenizer/      # BPE training and wrapper
│   ├── train/          # Training loop with AMP and validation
│   └── app.py          # Gradio Web Demo
├── checkpoints/        # Saved model weights (.pt)
└── requirements.txt    # Project dependencies
```

## 🚀 How to Use

### 1. Setup
```bash
git clone https://github.com/mrshibly/MiniGPT-from-Scratch.git
cd MiniGPT-from-Scratch
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python src/datasets/download_fineweb.py
python src/datasets/clean_text.py
python src/tokenizer/train_tokenizer.py
python src/datasets/prepare_data.py
```

### 3. Train
To train locally (CPU/GPU):
```bash
python src/train/train.py
```
*Note: For full 50M training, use the [Colab Template](notebooks/Colab_Training_Template.ipynb).*

### 4. Interactive Demo
Once you have a checkpoint in `checkpoints/ckpt.pt`:
```bash
python src/app.py
```

## 📊 Model Configurations
| Config | Params | Layers | Heads | d_model |
|--------|--------|--------|-------|---------|
| Tiny   | ~7M    | 4      | 4     | 256     |
| Standard| ~50M   | 6      | 8     | 512     |

## 📊 Latest Training Benchmarks
- **Dataset**: 500MB FineWeb-Edu
- **Parameters**: 27.54 Million
- **Validation Loss**: 4.8415
- **Perplexity**: 126.65
- **Status**: Successfully generating coherent English-like sentences.

## 📄 License
MIT
