# 🧠 Synor AI — 100M SOTA Foundation Model

**Synor** is a from-scratch, native 100 Million Parameter (`101,528,064`) Generative Pretrained Transformer (Causal Decoder LLM) built in Python and PyTorch. It features a modern SOTA architecture (RoPE, RMSNorm, SwiGLU, GQA, and KV-Cache) with zero third-party pre-trained weights and zero static heuristics.

---

## 📐 Architecture Specifications (`100m`)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **101,528,064** | Canonical 100M parameter foundation scale |
| **Positional Encoding** | **RoPE** | Rotary Position Embeddings (dynamic relative position tracking) |
| **Normalization** | **RMSNorm** | Root Mean Square Normalization ($\epsilon = 10^{-5}$) |
| **Activation & MLP** | **SwiGLU** | Gated Linear Units ($hidden\_dim = 2048$) |
| **Attention** | **GQA + FlashAttention** | Grouped-Query Attention (12 Q-heads, 4 KV-heads) |
| **Inference Engine** | **KV-Cache** | $O(1)$ fast single-step token generation |
| **Layers (`n_layer`)** | **10** | Hierarchical abstraction transformer decoder blocks |
| **Embedding Dim (`n_embd`)** | **768** | Standard hidden dimension |
| **Context Window (`block_size`)** | **512 tokens** | Multi-turn dialogue context window |
| **Vocabulary** | **50,257** | Byte-Pair Encoding (BPE) sub-word tokenizer |
| **Weight Tying** | **Enabled** | Output LM head shares weights with input embeddings |

---

## 🚀 Quick Start

### 1. Interactive Companion Chat
Run the conversational REPL:
```bash
python3 chat.py
```
Commands inside the REPL:
- **/mood**: Inspect the latent 3D emotional state manifold.
- **/learn**: Trigger on-demand continual learning from conversation memory.
- **exit**: Safely exit session.

### 2. Stream Generation CLI
```bash
python3 generate.py --prompt "User: Write a short email.\nAssistant:" --tokens 100 --temp 0.7
```

### 3. Production FastAPI OpenAI-Compatible Streaming Server
Run the local HTTP server supporting Server-Sent Events (SSE) streaming:
```bash
python3 serve.py --port 8000
```
- Health Check: `GET http://localhost:8000/health`
- Chat Completions: `POST http://localhost:8000/v1/chat/completions`

### 4. Automated Benchmark & Evaluation
Measure model validation loss and perplexity:
```bash
python3 eval.py --checkpoint checkpoints/best_model.pt
```

### 5. Run Unit Tests
Verify model architecture, KV-Cache precision, and tokenization:
```bash
pytest
```

---

## 🏋️ Two-Stage Foundation Training Pipeline

### Stage 1: Broad Pre-Training (Causal Next-Token Prediction)
Trains the model on extensive encyclopedic knowledge and text corpora:
```bash
python3 train.py --config 100m --stage pretrain --iters 1000 --batch-size 4 --grad-accum-steps 2 --lr 3e-4
```

### Stage 2: Instruction SFT (Supervised Fine-Tuning with Masked Target Loss)
Trains the model as an articulate personal assistant. User prompts are masked with `-100` so cross-entropy loss trains exclusively on Assistant responses:
```bash
python3 train.py --config 100m --stage sft --iters 2000 --batch-size 4 --grad-accum-steps 2 --lr 1e-4 --resume
```
