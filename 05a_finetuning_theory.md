# 05a – Fine-Tuning (Theory)

> **Core idea**: Pre-training teaches a model language. Fine-tuning teaches it *your specific style, format, or domain behavior*.

---

## 🧠 What Happens During Pre-Training vs Fine-Tuning

### Pre-Training (done by researchers, takes months)
```
Training data: The entire internet (trillions of words)
Objective: Predict the next token
Result: Model learns grammar, facts, reasoning, language patterns
Cost: Millions of dollars, thousands of GPUs, months of time
```

### Fine-Tuning (what you do, takes hours)
```
Training data: Your small curated dataset (hundreds to thousands of examples)
Objective: Adjust model weights to prefer your style/format/domain
Result: Model behaves exactly like your examples
Cost: Free (Google Colab) to cheap ($10-50 cloud GPU)
```

**Analogy**: Pre-training is going through 16 years of school. Fine-tuning is the 3-day onboarding at a new job — you already know how to work, now you learn this company's specific processes.

---

## 🆚 When to Use Fine-Tuning vs RAG vs Prompt Engineering

This is one of the most important decisions in applied AI:

| Approach | Best for | Not good for | Cost | Speed |
|----------|----------|-------------|------|-------|
| **Prompt Engineering** | One-off tasks, format control, general instructions | Consistent style at scale, private data | Free | Fast |
| **RAG** | Private/specific facts, up-to-date info, large document sets | Style, personality, consistent behavior | Low | Fast |
| **Fine-Tuning** | Consistent style/tone, specific output format, domain behavior | Adding new facts (models often "forget") | Medium | Slow to set up |

### Decision Tree

```
Do you need the model to know specific facts from your documents?
  → YES → Use RAG
  
Does the model already know the domain, but you need a specific style/tone?
  → YES → Fine-tune

Do you just need to guide the model's behavior for one task?
  → YES → Prompt engineering (try this first, always)

Do you need extreme speed and the model to feel "native" to your domain?
  → YES → Fine-tune (after RAG + prompting fail)
```

---

## ⚠️ The "Forgetting" Problem

Fine-tuning on new factual information often causes **catastrophic forgetting** — the model forgets general knowledge while learning your new facts.

```
Before fine-tuning:
  Q: "What is the capital of France?"  → "Paris"
  
After fine-tuning on medical records:
  Q: "What is the capital of France?"  → "Paris" (usually still fine)
  Q: "What does ATP stand for?"        → "adenosine triphosphate" (if in training data)
  
BUT if fine-tuning dataset is small and very domain-specific:
  Q: "Tell me a joke"                  → (model may fail or give weird answers)
```

**Solution**: Mix your fine-tuning data with some general-purpose examples (called "replay data") to prevent forgetting.

---

## 🔧 Types of Fine-Tuning

### Full Fine-Tuning
Update all model weights. Maximum effect, maximum compute.

### LoRA (Low-Rank Adaptation) — What You Should Use
Add small "adapter" matrices alongside the original weights. Train only the adapters.

```
Original model weights (frozen): [W]
LoRA adapters (trained): [A][B]  ← much smaller matrices

During inference: output = W·x + A·B·x
```

**Why LoRA?**: 
- 10-100x fewer trainable parameters
- Same quality as full fine-tuning in most cases
- Fits on free Google Colab GPU
- Can be swapped in/out without reloading the base model

### QLoRA (Quantized LoRA)
LoRA + model weights quantized to 4-bit. Runs on very limited GPU memory (even 8GB VRAM).

---

## 📊 What Makes a Good Fine-Tuning Dataset

| Quality Factor | What it means |
|----------------|---------------|
| **Format consistency** | Every example must follow the exact same input/output format |
| **Diversity** | Cover many variations of the task, not just one type |
| **Quality over quantity** | 500 excellent examples > 5000 noisy ones |
| **No contamination** | Keep validation examples separate from training |
| **Representative** | Examples should match the real distribution of inputs |

**Minimum viable dataset size**:
- Simple format/style: ~100 examples
- Domain-specific chatbot: ~500–1000 examples
- Complex reasoning: 2000+ examples

---

Continue to [05b_finetuning_project.md](05b_finetuning_project.md) — fine-tune a model on Colab.
