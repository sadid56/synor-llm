# 🧠 How Synor AI Works: A Deep Low-Level Technical Walkthrough

This document provides a comprehensive, first-principles technical breakdown of **Synor AI**—a Causal Transformer Decoder Language Model. It covers every layer from the absolute lowest level: tokenization, embedding spaces, Query-Key-Value self-attention matrices, causal masking, GELU feed-forward networks, backpropagation, and probabilistic text generation.

---

## 🧭 The Core Philosophy: What is an LLM Actually Doing?

An AI foundation model is not magical; it is an autoregressive **Next-Token Prediction Engine** governed by linear algebra and probability calculus.

When you provide the prompt:
```text
User: Hi
Assistant: 
```

The model uses its **1.8 million parameters** (neural network weights) to calculate the statistical probability distribution over its vocabulary:
> *"Given the exact sequence of characters that came before, what is the single most probable character to follow?"*

The model evaluates the probabilities:
- Probability of `'H'` = **89.4%**
- Probability of `'B'` = **1.8%**
- Probability of `'Z'` = **0.01%**

The model samples `'H'`. The prompt now becomes `...Assistant: H`. It then feeds this updated sequence back through its neural network to predict the next character: `'e'`, then `'l'`, then `'l'`, then `'o'`—producing coherent human language one token at a time.

---

## 🔄 End-to-End Execution Pipeline

```
[Raw User Text] "Hi"
       │
       ▼
[1. Tokenizer] ─── Encodes string into discrete token integers: [20, 47]
       │
       ▼
[2. Embedding Layer] ─── Projects token IDs to 192D continuous vectors + adds Positional Embeddings
       │
       ▼
[3. Transformer Blocks (x4 Stacked Layers)]
       ├── Pre-LayerNorm (Normalizes mean & variance)
       ├── Causal Multi-Head Attention (Computes contextual token-to-token relationships)
       ├── Residual Addition (High-speed gradient & information highway)
       ├── Pre-LayerNorm
       ├── Feed-Forward Network / MLP (4x expansion to 768 dim + GELU activation)
       └── Residual Addition
       │
       ▼
[4. Final LayerNorm & LM Head] ─── Projects 192D vectors back to 65 raw vocabulary scores (Logits)
       │
       ▼
[5. Softmax & Sampling Engine] ─── Converts logits to probabilities; applies Temperature, Top-K, Top-P
       │
       ▼
[Generated Character] "Hello! How can I help you today?"
```

---

## 🔬 Step 1: Tokenization (Mapping Characters to Integer IDs)

Neural networks cannot process raw ASCII or Unicode characters directly. The input string must first be mapped into discrete integers known as **Token IDs**.

Implemented in [`synor/tokenizer.py`](synor/tokenizer.py):
```python
# Internal Vocabulary Map (65 unique characters)
stoi = {'\n': 0, ' ': 1, '!': 2, 'H': 20, 'a': 39, 'e': 43, 'i': 47, ...}
```

When you pass the string `"Hi"`:
```python
tokens = tokenizer.encode("Hi")
# Output: [20, 47]
```

In PyTorch, this is represented as a 1D tensor:
$$\mathbf{x} = [20, 47] \in \mathbb{Z}^{T}$$
where $T = 2$ is the current sequence length.

---

## 🌌 Step 2: Vector Embeddings & Positional Encoding

Passing raw numbers like `20` or `47` directly into matrix multiplications would imply that character `47` is mathematically "larger" than character `20`. To give characters rich, semantic meaning, each discrete token is projected into a **192-dimensional continuous latent space** ($d = 192$).

### 1. Token Embeddings ($W_{tok} \in \mathbb{R}^{V \times d}$)
The model maintains a learned lookup table:
- `'H'` (Token 20) $\to$ `[0.12, -0.45, 0.89, ..., 0.03]` (192 floating-point numbers)
- `'i'` (Token 47) $\to$ `[-0.08, 0.71, -0.22, ..., 0.15]` (192 floating-point numbers)

### 2. Positional Embeddings ($W_{pos} \in \mathbb{R}^{S \times d}$)
Standard self-attention is permutation-invariant: it does not inherently know whether a word came first or last. `"Cat bites dog"` and `"Dog bites cat"` contain identical tokens, but opposite meanings.

To inject sequence order, Synor uses a learned positional embedding table up to context window $S = 128$:
- Position $0$ $\to$ Position Vector $0$ (192 dimensions)
- Position $1$ $\to$ Position Vector $1$ (192 dimensions)

### 3. Final Vector Summation:
$$\mathbf{x}_t = \text{Embedding}_{tok}(\text{token}_t) + \text{Embedding}_{pos}(t) \quad \in \mathbb{R}^{1 \times 192}$$

Now, each token vector encodes both **what character it is** and **where it sits in the sequence**.

---

## 🧠 Step 3: The Transformer Engine — Causal Multi-Head Self-Attention

Self-Attention is the architectural breakthrough introduced in the seminal paper *"Attention Is All You Need"* (Vaswani et al., 2017). It allows every token in a sequence to dynamically look at and extract context from every preceding token.

### The Query, Key, Value (Q, K, V) Paradigm:
For every token vector $\mathbf{x}$, the model computes three distinct representations using learned projection matrices:
1. **Query ($Q$):** *"What kind of information am I looking for?"*
2. **Key ($K$):** *"What information do I contain?"*
3. **Value ($V$):** *"What actual contextual information will I pass along?"*

> **Real-World Analogy (Database / Search Engine):**
> - What you type into the search bar = **Query ($Q$)**
> - The titles and tags of all database records = **Key ($K$)**
> - The actual content of the retrieved document = **Value ($V$)**

### Mathematical Formulation:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \times K^T}{\sqrt{d_k}} + M\right) \times V$$

1. **$Q \times K^T$ (Dot Product):** Multiplies every Query with every Key. If a Query and Key point in the same direction in vector space, their dot product is high, indicating strong relevance.
2. **$\sqrt{d_k}$ Scaling:** Divides by $\sqrt{32} \approx 5.65$ to prevent values from exploding, ensuring gradients remain stable during Softmax.
3. **Causal Mask ($M$):** Because language generation is autoregressive (left-to-right), future tokens must not leak into the past. An upper-triangular mask of $-\infty$ is applied:
   $$\begin{bmatrix} 
   0 & -\infty & -\infty \\ 
   0 & 0 & -\infty \\ 
   0 & 0 & 0 
   \end{bmatrix}$$
   Softmax($-\infty$) equals $0$, mathematically guaranteeing zero attention to future tokens.
4. **Softmax:** Converts dot-product scores into a normalized probability distribution where each row sums to $1.0$ (100%).
5. **$\times V$:** Computes the weighted average of the Values based on attention weights.

### Multi-Head Architecture ($n_{head} = 6$):
Rather than calculating attention once across the entire 192 dimensions, Synor splits the vectors into **6 parallel attention heads**, each operating in a 32-dimensional subspace ($192 \div 6 = 32$):
- Head 1 may focus on immediate adjacent characters (spelling).
- Head 2 may track dialogue prefixes (`User:` vs `Assistant:`).
- Head 3 may focus on sentence punctuation and boundaries.

The outputs of all 6 heads are concatenated back into a 192D vector and projected through an output linear layer (`c_proj`).

---

## ⚡ Step 4: Feed-Forward Network (MLP) & GELU Activation

While Self-Attention routes and shares information across tokens, the **Feed-Forward Network (MLP)** is where the model "processes" that information and stores learned associations.

```python
self.net = nn.Sequential(
    nn.Linear(192, 4 * 192),  # 1. Expand dimension 4x: 192 -> 768
    nn.GELU(),                # 2. Non-linear decision gate
    nn.Linear(4 * 192, 192),  # 3. Contract back: 768 -> 192
    nn.Dropout(0.1),
)
```

- **4x Expansion ($192 \to 768$):** Expanding into a higher-dimensional space gives the network the mathematical capacity to model complex non-linear interactions between concepts.
- **GELU (Gaussian Error Linear Unit):** Unlike traditional ReLU which harshly clamps negative values to zero ($f(x) = \max(0, x)$), GELU provides a smooth probabilistic curve that weights inputs by their likelihood under a normal distribution, improving gradient flow.

---

## 🛡️ Architectural Stability: Pre-LayerNorm & Residual Stream

Synor stacks **4 identical Transformer blocks** sequentially (`n_layer = 4`). Passing signals through deep networks poses the risk of vanishing or exploding gradients. Synor uses two proven stabilization techniques:

### 1. Residual Connections ($x = x + \text{SubLayer}(x)$)
Each sublayer (Attention and MLP) adds its output directly back to its input. This acts as an uninterrupted "gradient highway", allowing error signals during backpropagation to flow directly from the final layer back to the very first layer without attenuation.

### 2. Pre-LayerNorm Architecture
Unlike original 2017 Post-LN Transformers, Synor applies Layer Normalization **before** entering the Attention and MLP layers:
$$\mathbf{x} = \mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x}))$$
This standardizes the mean to $0$ and variance to $1$, making training exceptionally stable even at higher learning rates.

---

## 🎯 Step 5: How the AI Learns (Loss & Backpropagation)

At initialization, all 1.8M parameters are random numbers. The training loop teaches the model using gradient descent:

```
[Model Predictions] ──┐
                      ├──► [Cross-Entropy Loss] ──► [Backpropagation] ──► [AdamW Optimizer]
[Target Next Tokens] ──┘
```

1. **Forward Pass:** The model processes a training batch and outputs raw unnormalized prediction scores ($\mathbf{z} \in \mathbb{R}^{B \times T \times 65}$).
2. **Cross-Entropy Loss:** Quantifies the error between predicted probabilities and ground truth next tokens:
   $$\mathcal{L} = -\sum_{i=1}^{V} y_i \log(\hat{y}_i)$$
   *(When training began on Synor, loss was **4.23**; through training, loss converged to **0.21**).*
3. **Backpropagation:** Computes the analytical gradient $\frac{\partial \mathcal{L}}{\partial \theta}$ for all 1,813,824 parameters using the calculus chain rule.
4. **AdamW Optimizer:** Updates every weight vector $\theta$ using adaptive learning rates, tracking running momentum ($m_t$) and squared gradients ($v_t$):
   $$\theta_{t+1} = \theta_t - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta_t \lambda \theta_t$$
   with a Cosine Annealing learning rate schedule and linear warmup.

---

## 🗣️ Step 6: Text Generation & Sampling (How It Talks)

During inference ([`synor/sampler.py`](synor/sampler.py)), the model converts the final 192D hidden state into text:

1. **Language Model Head (`lm_head`):** Projects 192 dimensions into 65 logits (one for each character in the vocabulary).
   *(Using **Weight Tying**, `lm_head` shares weights with `token_embedding`, reducing memory and improving representation quality).*
2. **Temperature Scaling ($T$):** Divides logits by $T$:
   $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
   - **Low Temperature ($0.2 - 0.4$):** Sharpens probabilities; makes the model factual, deterministic, and accurate.
   - **High Temperature ($0.8 - 1.2$):** Flattens probabilities; increases creativity and diversity.
3. **Top-K Filtering ($k = 40$):** Truncates the vocabulary to only the top 40 most likely tokens, setting all others to $-\infty$.
4. **Top-P (Nucleus) Filtering ($p = 0.9$):** Selects the smallest dynamic set of tokens whose cumulative probability exceeds 90%, discarding unlikely outliers.
5. **Categorical Sampling (`torch.multinomial`):** Samples a token from the filtered distribution, appends it to the sequence, and streams it to the screen.
6. **Stop Strings (`\nUser:`, `\n`):** When the model generates a boundary delimiter indicating its conversational turn is complete, the generator immediately halts.

---

## 📊 Synor AI Technical Architecture Specifications

| Architecture Attribute | Value | Mathematical Rationale |
| :--- | :--- | :--- |
| **Model Type** | Causal Transformer Decoder | Autoregressive language modeling (GPT-style) |
| **Total Parameters** | **1,813,824 (~1.8M)** | Full neural capacity across embeddings and 4 blocks |
| **Embedding Dimension ($n_{embd}$)** | **192** | Latent semantic representation width |
| **Attention Heads ($n_{head}$)** | **6 Heads** | 6 parallel attention subspaces ($192 \div 6 = 32$ dim per head) |
| **Transformer Blocks ($n_{layer}$)** | **4 Layers** | Depth of hierarchical abstraction and reasoning |
| **Context Window ($block\_size$)** | **128 Tokens** | Maximum receptive field length for autoregressive attention |
| **Vocabulary Size ($vocab\_size$)** | **65 Tokens** | Character set mapping all English letters, digits, and symbols |
| **Attention Kernel** | FlashAttention (PyTorch SDPA) | Hardware-accelerated $O(N)$ memory attention on Apple Silicon/CUDA |
| **FeedForward Dimension** | **768** | $4 \times n_{embd}$ expansion with GELU non-linearity |
| **Weight Tying** | Enabled | Ties `token_embedding.weight` to `lm_head.weight` |

---

## 💡 Conclusion

Synor AI operates on the exact mathematical principles that power world-class frontier models like **GPT-4**, **Llama 3**, and **Gemini**. The difference lies strictly in scale: frontier industrial models use billions of parameters trained across petabytes of text on clusters of thousands of GPUs, while **Synor AI** is engineered from first principles to be modular, transparent, and completely understandable on your local machine.
