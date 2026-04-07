# 09 – How LLMs Work Internally

> **Why read this?** You don't need to understand transformer internals to use LLMs, but understanding the basics makes every error message, every parameter, and every prompt technique make immediate sense.

---

## 🧠 The Big Picture

An LLM is fundamentally a **next-token prediction machine**:

```
Input:  "The capital of France is"
Output: probability distribution over all possible next tokens

  "Paris"   → 97.3%
  "located" → 1.2%
  "a"       → 0.5%
  ...

The model picks "Paris" → now input becomes "The capital of France is Paris"
Repeats until [END] token is generated
```

Everything else — answering questions, writing code, explaining concepts — emerges from doing this extremely well, at extreme scale.

---

## 🔤 Step 1: Tokenization

Before any processing, text is split into **tokens**. A token is not a word — it's a subword unit:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Transformers are the backbone of modern AI"
tokens = tokenizer.tokenize(text)
# ['Transform', 'ers', 'Ġare', 'Ġthe', 'Ġbackbone', 'Ġof', 'Ġmodern', 'ĠAI']
# 8 tokens, not 7 words!

ids = tokenizer.encode(text)
# [8291, 364, 389, 262, 27169, 286, 3660, 9552]
# Each token → an integer (its ID in the vocabulary)
```

**Why subword tokenization?**
- `"unhappiness"` → `["un", "happi", "ness"]` — handles unseen words by composing known pieces
- Rare words don't each need their own vocabulary entry
- The vocabulary (~50,000 entries) covers essentially all text

**Token cost matters**: API pricing is per token. 1 token ≈ 4 characters ≈ 0.75 words in English. Non-English text often uses more tokens per word.

---

## 🏗️ Step 2: The Transformer Architecture

```
Token IDs [8291, 364, 389, ...]
    │
    ▼
Token Embeddings (lookup table → each ID becomes a 768-dim vector)
    │
    ▼
Position Encoding (adds "where in the sequence" information)
    │
    ▼
┌─────────────────────────────────────────┐
│  Transformer Block (repeated N times)   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   Multi-Head Self-Attention     │   │  ← "Which other tokens should I
│  │   (each token attends to all    │   │     pay attention to?"
│  │    other tokens)                │   │
│  └─────────────────────────────────┘   │
│           │ (residual connection)       │
│  ┌─────────────────────────────────┐   │
│  │   Feed-Forward Network          │   │  ← "Process this attended info"
│  │   (2 linear layers + activation)│   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
Final Linear Layer + Softmax
    │
    ▼
Probability over vocabulary (which token comes next?)
```

**GPT-2**: 12 layers, 768-dim → 117M parameters  
**GPT-3**: 96 layers, 12288-dim → 175B parameters  
**DeepSeek-R1**: 61 layers, mixture-of-experts → 671B parameters total

---

## 🎯 Attention Mechanism — The Key Innovation

Self-attention is what makes transformers powerful. For each token, it answers: "which other tokens in the sequence are most relevant to understanding me?"

```
Sentence: "The bank can guarantee deposits will be covered"

For the token "bank" (financial institution):
  - High attention to: "deposits", "covered", "guarantee"
  - Low attention to: "The", "can", "will", "be"

For the same word "bank" in "I sat by the river bank":
  - High attention to: "river", "sat", "by"
  - Low attention to: "I", "the"

Same word, different context → different understanding
This is how transformers handle ambiguity.
```

**Multi-Head Attention**: Multiple attention mechanisms run in parallel. One head might focus on syntactic relationships, another on semantic meaning, another on coreference (tracking what "it" refers to).

---

## 📏 Context Window

The context window is the maximum number of tokens the model can "see" at once. Everything outside it is invisible.

```
Context window = 8192 tokens (example)

[VISIBLE TO MODEL]
System prompt (200 tokens)
Conversation history (5000 tokens)
Current user message (100 tokens)
Retrieved RAG context (2000 tokens)
─────────────────────────────────────
Total: 7300 tokens (within window ✓)

If you add 1000 more tokens: 8300 → oldest tokens get dropped
```

**Why this matters**:
- Very long conversations lose early context
- RAG chunks must fit within the remaining space after system + conversation
- Models with larger context windows cost more per token

---

## 🧠 Why LLMs "Hallucinate"

Hallucination = the model generating plausible-sounding but false information. It happens because:

1. **Training objective mismatch**: The model was trained to predict *fluent text*, not to only say things it's certain about.

2. **No knowledge flag**: The model has no internal "I don't know" signal. It just predicts the most probable next token.

3. **Pattern completion**: If the pattern strongly suggests an answer (even if wrong), the model follows it.

```
Prompt: "The first person to walk on Mars was..."
The model was trained on lots of text with this pattern:
"The first person to [milestone] was [name]"
It will complete this even though no one has walked on Mars yet.
```

**Fixes**:
- RAG: ground answers in retrieved documents
- System prompt: "Only answer based on the provided context. Say 'I don't know' if the answer isn't there."
- Low temperature: reduces creative fabrication
- Evaluation: systematically detect hallucinations (see file 06)

---

## ⚡ Model Sizes and What They Can Do

| Size | Parameters | Typical use | Example |
|------|-----------|-------------|---------|
| Tiny | <1B | On-device, real-time | Phi-3 mini |
| Small | 1–7B | Fast inference, limited reasoning | Llama 3.1 8B |
| Medium | 7–30B | Good balance, most tasks | Mistral 7B, Qwen 14B |
| Large | 30–70B | Strong reasoning | Llama 3.1 70B |
| XL | 70B+ | Best quality, expensive | Llama 3.1 405B |

**Rule of thumb**: Use the smallest model that accomplishes the task. Smaller = faster + cheaper.

---

## 🔢 Key Numbers to Remember

| Thing | Value |
|-------|-------|
| 1 token | ~4 chars / ~0.75 English words |
| GPT-4 context window | 128K tokens |
| DeepSeek-R1 context | 128K tokens |
| 1 page of text | ~500 tokens |
| 1 book (300 pages) | ~150K tokens |
| Minimum for fine-tuning | ~100 examples |
| Semantic embedding size (MiniLM) | 384 dimensions |
| Temperature sweet spot | 0.5–0.7 |

---

This file is theory-only. The understanding here will make every debug session faster.

Next: [10_production.md](10_production.md) — deploying responsibly.
