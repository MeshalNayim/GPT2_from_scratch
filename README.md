# LLM from Scratch

This project is a Large language model built **completely from scratch** using PyTorch and a few essential libraries. It demonstrates how to build, train, and generate text with a transformer-based model without relying on heavy frameworks or pre-built APIs.
The LLM implements GPT-2 Small architecure and configuration with 124 Million trainable parameters
---

## Project Overview

The repository is organized into clear modules representing the core parts of an LLM pipeline:

### 1. **llm/**
- Core model components, including:
  - Transformer blocks
  - Multi-head attention
  - Feed-forward layers
  - Normalization
  - GPT model architecture
- Configurations for model parameters

### 2. **data/**
- Text data downloading utility
- Tokenizer implemented via TikToken
- Dataset and DataLoader creation of input-target pairs

### 3. **train_and_test/**
- Training loop with loss calculation and model optimization
- Evaluation utilities for validation

### 4. **generate/**
- Functions for generating text from the trained model using different decoding strategies

### 5. **scripts/**
- Entry point scripts to run training and generation workflows easily

---

## Key Features

- **Built from scratch:** No heavy transformers libraries; model architecture implemented manually using PyTorch.
- **Minimal dependencies:** Uses only PyTorch and TikToken for tokenization.
- **Clean modular structure:** Separate modules for clarity and maintainability.
- **Configurable:** Easily change model size and training parameters through configuration.
- **Creativity Tuning:** Adjust the temprature Scaling and Top-K Sampling Factors to tune your LLM.

---

---

## More to Come

This repository is a work in progress. More features, improvements, and detailed documentation will be added soon.

Stay tuned!

