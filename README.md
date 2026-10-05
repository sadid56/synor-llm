# 🧠 Synor AI — 100M SOTA Foundation Model

**Synor** is a from-scratch, native 100 Million Parameter (`101,528,064`) Generative Pretrained Transformer (Causal Decoder LLM) built in Python and PyTorch with zero third-party base models or proprietary APIs. 

It implements a modern SOTA architecture (RoPE, RMSNorm, SwiGLU, Grouped-Query Attention, and KV-Cache) designed for fast, pure vanilla neural training and inference on Apple Silicon (MPS), CUDA GPUs, and CPU.

---

## 📐 Architecture Specifications (`100m`)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **101,528,064** | Canonical 100M parameter foundation scale |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings (dynamic relative position tracking) |
| **Normalization** | **RMSNorm** | Root Mean Square Normalization ($\epsilon = 10^{-5}$) |
| **Activation & MLP** | **SwiGLU** | Gated Linear Units ($hidden\_dim = 2048$) |
| **Attention** | **GQA** | Grouped-Query Attention (12 Q-heads, 4 KV-heads) |
| **Inference Engine** | **KV-Cache** | $O(1)$ fast single-step token autoregressive generation |
| **Layers (`n_layer`)** | **10** | Hierarchical abstraction transformer decoder blocks |
| **Embedding Dim (`n_embd`)** | **768** | Hidden dimension |
| **Context Window (`block_size`)** | **512 tokens** | Multi-turn dialogue context window |
| **Vocabulary** | **50,257** | Byte-Pair Encoding (BPE) sub-word tokenizer |
| **Weight Tying** | **Enabled** | Output LM head shares weights with token embedding layer |

---

## 🚀 Quick Start

### 1. Interactive Companion Chat (Pure Vanilla Brain)
Run the conversational REPL directly on your trained model weights (`checkpoints/best_model.pt`):
```bash
python3 chat.py
```
* **Pure Neural Generation:** Direct autoregressive token sampling from model weights with zero heuristic rules, emotions, or web-scraping wrappers.
* **Commands:** `clear` (clear screen), `exit` (quit).

### 2. Stream Generation CLI (One-Shot)
Test prompts with customized temperature, top-k/top-p, or tokens:
```bash
python3 generate.py --prompt "User: Hello! Who are you?\nAssistant:" --temp 0.7 --tokens 120
```

### 3. Universal Dataset Ingestion
Download and automatically format any text or JSON/JSONL dataset from Hugging Face or GitHub into training-ready dialogue pairs:
```bash
# Interactive Mode:
python3 scripts/download.py

# Command-Line Mode (SFT Dialogue Data):
python3 scripts/download.py --url "https://huggingface.co/datasets/.../data.jsonl" --name custom_chat.txt

# Pretraining Corpus Mode:
python3 scripts/download.py --url "https://.../corpus.txt" --name my_corpus.txt --stage pretrain
```

### 4. Production FastAPI OpenAI-Compatible Server
Run the local HTTP server supporting Server-Sent Events (SSE) streaming:
```bash
python3 serve.py --port 8000
```
- Health Check: `GET http://localhost:8000/health`
- Chat Completions: `POST http://localhost:8000/v1/chat/completions`

### 5. Automated Benchmark & Evaluation
Measure validation loss and perplexity on the current checkpoint:
```bash
python3 eval.py --checkpoint checkpoints/best_model.pt
```

### 6. Run Unit Tests
Verify model architecture, KV-Cache mathematical equivalence, and tokenization:
```bash
pytest tests/
```

---

## 🏋️ Training Pipeline

### Supervised Fine-Tuning (SFT / Masked Loss)
Trains the model as an articulate assistant using datasets in `data/sft/`. User prompts are masked with `-100` so loss trains exclusively on Assistant responses:
```bash
python3 train.py --config 100m --stage sft --iters 5000 --batch-size 4 --grad-accum-steps 2 --lr 1e-4
```

### Checkpointing System
* **`checkpoints/best_model.pt`**: Automatically captures the historical lowest validation loss record (`★ New Record`). Recommended for production chat & generation.
* **`checkpoints/latest.pt`**: Most recent step checkpoint. Used to seamlessly resume training from where you left off (`--resume True`).
