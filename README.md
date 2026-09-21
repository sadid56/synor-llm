# 🧠 Synor LLM

**Synor** is a from-scratch, production-grade Generative Pretrained Transformer (Causal Decoder LLM) built in Python and PyTorch. It implements the core architecture behind modern generative models like GPT and Gemini (Multi-Head Self-Attention, GELU FeedForward, Pre-LayerNorm, weight tying, and FlashAttention-powered compute).

---

## 💻 1. How to Use Locally (Python API)

You can import and generate text with Synor directly in your Python code:

```python
import torch
from synor import SynorLM, CharTokenizer, TextSampler

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 1. Load tokenizer and trained weights
tokenizer = CharTokenizer.load("data/meta.pkl")
checkpoint = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)

model = SynorLM(checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model_state_dict"])

# 2. Generate text
sampler = TextSampler(model=model, tokenizer=tokenizer, device=device)
output = sampler.generate(prompt="ROMEO:", max_new_tokens=200, temperature=0.8)
print(output)
```

---

## 🏋️ 2. How to Train

### Add Your Training Data
Place any `.txt` text files (literature, books, articles, code, conversations) into the `data/raw/` directory. Synor automatically scans and trains on all text files inside this folder.

### Train from Scratch
```bash
python3 train.py --config tiny --iters 1000
```

Available presets for `--config`:
- `tiny` (~1.8M parameters — fast on laptops and M-series Macs)
- `small` (~10M parameters)
- `medium` (~30M parameters)
- `large` (~85M parameters)

### Continue / Resume Training
Training automatically preserves the AdamW optimizer state and learning rate schedule:
```bash
python3 train.py --resume --iters 500
```

---

## 💬 3. How to Chat & Generate Text

### Generate from a Prompt (CLI)
```bash
python3 generate.py --prompt "CITIZEN:" --tokens 300 --temp 0.8
```

### Interactive Terminal Chat
Start a real-time conversation session with your model:
```bash
python3 chat.py
```
Type your prompt and press **Enter**. Type `exit` or `quit` to end the session.

### 🌐 Live Web Search & Grounding
Inside `chat.py`, you can ask real-time or factual questions (e.g., `"who is the CEO of Google"`, `"capital of France"`, or explicitly `"/search <query>"`). Synor automatically searches DuckDuckGo and Wikipedia in real time, retrieves verified information, and honestly admits if no information could be found.
