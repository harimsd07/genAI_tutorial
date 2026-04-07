# 🌟 Generative AI – Complete Self-Contained Learning Guide

> **Promise**: Study this guide from start to finish and you will understand every major concept in modern Generative AI — from first principles to working projects. No external resources needed. Every analogy, every concept, every line of code is explained here.

---

## 📚 What's Inside?

| File | Topic | What You'll Build |
|------|-------|-------------------|
| [01_setup.md](01_setup.md) | Environment Setup | Working API connection + diagnostic script |
| [02a_prompt_eng.md](02a_prompt_eng.md) | Prompt Engineering (Theory) | Deep understanding of all prompting techniques |
| [02b_prompt_project.md](02b_prompt_project.md) | Prompt Engineering (Project) | Prompt Playground web app with A/B testing |
| [03_rag.md](03_rag.md) | RAG – Retrieval-Augmented Generation | PDF chat app + multi-document knowledge base |
| [04a_function_calling_theory.md](04a_function_calling_theory.md) | Function Calling & Agents (Theory) | How AI uses tools and plans actions |
| [04b_function_calling_project.md](04b_function_calling_project.md) | Function Calling & Agents (Project) | Multi-tool AI agent with weather, news, math |
| [05a_finetuning_theory.md](05a_finetuning_theory.md) | Fine-Tuning (Theory) | When and why to fine-tune vs other approaches |
| [05b_finetuning_project.md](05b_finetuning_project.md) | Fine-Tuning (Project) | Custom Shakespeare model on free Colab GPU |
| [06_evaluation.md](06_evaluation.md) | Evaluation & Testing | Automated AI quality scoring harness |
| [07_multimodal.md](07_multimodal.md) | Multi-modal AI | Image captioning, VQA, text-to-image app |
| [08_vector_databases.md](08_vector_databases.md) | Vector Databases (Deep Dive) | ChromaDB + persistent knowledge store |
| [09_llm_internals.md](09_llm_internals.md) | How LLMs Work Internally | Transformers, attention, tokenization explained |
| [10_production.md](10_production.md) | Production & Best Practices | Rate limiting, caching, cost control, deployment |
| [cheat_sheet.md](cheat_sheet.md) | Quick Reference | Every snippet, every model, one page |

---

## 🗺️ Learning Path

```
START HERE
    │
    ▼
[01] Setup ──────────────────────────────────────────────────────────► Get API working
    │
    ▼
[09] How LLMs Work (optional but highly recommended early)
    │   Understand what's happening under the hood
    ▼
[02] Prompt Engineering ─────────────────────────────────────────────► Most used skill
    │   Theory first (02a), then build the playground (02b)
    ▼
[03] RAG ────────────────────────────────────────────────────────────► Most useful pattern
    │   Making AI read YOUR documents
    ▼
[08] Vector Databases (Deep Dive) ───────────────────────────────────► Powers RAG properly
    │
    ▼
[04] Function Calling / Agents ──────────────────────────────────────► AI that takes actions
    │
    ▼
[06] Evaluation ─────────────────────────────────────────────────────► Measure your work
    │
    ▼
[05] Fine-Tuning ────────────────────────────────────────────────────► Advanced: change style
    │
    ▼
[07] Multi-Modal ────────────────────────────────────────────────────► Vision + Language
    │
    ▼
[10] Production Best Practices ──────────────────────────────────────► Ship it!
```

---

## 🧠 Core Mental Model

Before you read anything else, internalize this:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      THE GENERATIVE AI STACK                            │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │    Input    │  │  Retrieved  │  │   LLM Core   │  │   Output    │  │
│  │             │  │  Context    │  │              │  │             │  │
│  │ • Text      │─►│             │─►│ • Reasoning  │─►│ • Text      │  │
│  │ • Images    │  │ • RAG docs  │  │ • Generation │  │ • JSON      │  │
│  │ • Audio     │  │ • Tool res. │  │ • Planning   │  │ • Images    │  │
│  │ • Files     │  │ • Memory    │  │ • Tool calls │  │ • Actions   │  │
│  └─────────────┘  └─────────────┘  └──────────────┘  └─────────────┘  │
│                                                                         │
│  What YOU control:  Prompt  │  Retrieved context  │  Temperature/params │
└─────────────────────────────────────────────────────────────────────────┘
```

**The key insight**: LLMs are fixed once trained. Everything that makes them useful in your application happens in *what you feed them* (prompt engineering + RAG) and *what tools they can call* (function calling).

---

## 🛠️ Prerequisites

| Requirement | Level |
|-------------|-------|
| Python basics | Know how to write a function, use pip, run a script |
| Terminal/Command line | Know how to `cd`, run commands |
| RAM | 4GB minimum (8GB recommended) |
| Internet | Required for API calls |
| Time | 15 min setup + ~1 hour per topic |

**No ML/math background needed.** This guide builds understanding through analogy first, then code.

---

## 💻 What You'll Need to Install (Summary)

```bash
pip install huggingface_hub streamlit chromadb sentence-transformers \
            pymupdf requests python-dotenv transformers datasets accelerate
```

Full instructions in `01_setup.md`.

---

## 🧩 How the Projects Connect

Every project in this guide builds on the last:

```
Prompt Playground  ──►  RAG PDF Chat  ──►  AI Agent  ──►  Full Assistant
     (02b)                 (03)             (04b)          (combines all)
      │                     │                │
      │  You learn to       │  You add       │  You add
      │  control output     │  document      │  actions +
      │  quality            │  knowledge     │  tool use
```

By the end, you'll have the building blocks for a **production-quality AI assistant** that:
- Answers from your documents (RAG)
- Can take actions (function calling)  
- Understands images (multi-modal)
- Is systematically tested (evaluation)

---

## 📝 How to Use This Guide

1. **Read the theory file first** for each topic — don't skip analogies, they lock in understanding.
2. **Copy and run every code snippet** — reading code is not the same as running it.
3. **Break things on purpose** — change parameters, swap models, cause errors. Errors teach you what parameters do.
4. **Do the challenges** at the end of each project section.
5. Keep `cheat_sheet.md` open in a second window as quick reference.

Let's begin → Open [01_setup.md](01_setup.md)
