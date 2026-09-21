# 🧠 Synor AI — 100M Foundation Model

**Synor** is a from-scratch, native 100 Million Parameter (`109,840,128`) Generative Pretrained Transformer (Causal Decoder LLM) built in Python and PyTorch. It features a pure neural architecture with zero third-party pre-trained weights and zero static `if/elif` heuristics.

---

## 📐 Architecture Specifications (`100m`)

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | **109,840,128** | Canonical 100M scale for personal AI foundation modeling |
| **Layers (`n_layer`)** | **10** | Deep hierarchical abstraction (syntax $\to$ semantics $\to$ dialogue reasoning) |
| **Embedding Dim (`n_embd`)** | **768** | Standard GPT hidden dimension |
| **Attention Heads (`n_head`)** | **12** | 64 dimensions per head ($768 \div 12 = 64$) |
| **Context Window (`block_size`)** | **512 tokens** | Long-context dialogue and multi-turn document comprehension |
| **Vocabulary** | **50,257** | Native Sub-word Byte-Pair Encoding (BPE) |
| **Attention Kernel** | **PyTorch FlashAttention** | Hardware-accelerated Apple Silicon MPS memory efficiency |
| **Weight Tying** | **Enabled** | Output LM head shares weights directly with input embeddings |

---

## 🚀 Quick Start

### 1. Interactive Companion Chat (Zero Static Rules)
Run the neural conversational console:
```bash
python3 chat.py
```
Inside the interactive REPL:
- **Pure Neural Response**: Generated autoregressively from the model's 100M weights.
- **/mood**: Inspect the latent 3D emotional state (valence, arousal, dominance).
- **/learn**: Trigger on-demand continual learning from recent conversation memory.
- **exit**: Safely exit session.

### 2. Stream Generation CLI
```bash
python3 generate.py --prompt "User: Why human feels tired ?\nAssistant:" --tokens 80 --temp 0.3
```

---

## 🏋️ Two-Stage Foundation Training Pipeline

Synor implements modern LLM training using two decoupled stages:

### Stage 1: Broad Pre-Training (Causal Next-Token Prediction)
Trains the model on extensive science, biology, Python engineering, algorithms, Bangladesh geography, and world history:
```bash
python3 train.py --config 100m --stage pretrain --iters 500 --batch-size 4 --grad-accum-steps 2 --lr 3e-4
```

### Stage 2: Instruction SFT (Supervised Fine-Tuning with Masked Target Loss)
Trains the model to behave as an articulate, empathetic AI companion. User prompts are masked with `-100` so that cross-entropy loss exclusively trains on the Assistant's responses:
```bash
python3 train.py --config 100m --stage sft --iters 200 --batch-size 4 --grad-accum-steps 2 --lr 1e-4 --resume
```

---

## 🛠️ Project Structure

```
├── synor/
│   ├── model.py           # PyTorch Causal Transformer (SelfAttention, MLP, SynorLM)
│   ├── config.py          # Presets (100m, large, medium, small, tiny)
│   ├── bpe_tokenizer.py   # Byte-Pair Encoding Sub-word Tokenizer (50,257 vocab)
│   ├── dataset.py         # TextDataset (Pretrain) & SFTDataset (Masked Loss)
│   ├── trainer.py         # Cosine LR scheduler, gradient accumulation, atomic checkpoints
│   ├── sampler.py         # Top-K, Top-P nucleus sampling, temperature & stop strings
│   ├── emotion.py         # Continuous 3D emotional state manifold
│   ├── memory.py          # Episodic multi-turn conversation buffer
│   └── learner.py         # Continual auto-learning backpropagation engine
├── scripts/
│   ├── build_100m_corpus.py  # Assembles pretrain & SFT datasets
│   └── download_corpus.py    # Curated encyclopedia knowledge ingestion
├── chat.py                # Pure neural conversational REPL
├── generate.py            # CLI text generation engine
├── train.py               # Master training CLI (pretrain & SFT)
└── checkpoints/
    ├── best_model.pt      # Best validation loss checkpoint
    └── latest.pt          # Latest training step checkpoint
```
