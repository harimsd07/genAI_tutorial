# 🔥 Generative AI – Complete Cheat Sheet

> One page with everything you need. Keep this open in a second window.

---

## Core Concepts (One Line Each)

| Concept | What it is |
|---------|-----------|
| **Token** | ~4 chars of text. The atom of LLMs. Costs money. |
| **Prompt** | Everything fed to the model: system + history + user message |
| **Temperature** | How random the output is. 0=deterministic, 1=creative |
| **Embedding** | Text converted to a vector. Similar text = similar vectors |
| **RAG** | Give the AI relevant document chunks right before it answers |
| **Fine-tuning** | Retrain on your data to change behavior/style. Use LoRA. |
| **Function calling** | AI outputs a tool command; your code executes it |
| **Agent** | LLM in a loop; takes multiple actions to complete a task |
| **Hallucination** | Model confidently says something false. Prevent with RAG. |
| **Context window** | How much text the model can "see" at once. Measured in tokens. |
| **System prompt** | Instructions given before the conversation. Sets behavior. |
| **Few-shot** | Giving examples in the prompt to guide format/style |
| **Chain-of-thought** | Ask model to reason step-by-step before answering |
| **Vector DB** | Database for storing embeddings. Enables fast similarity search. |

---

## Hugging Face Snippets

### Basic setup (always start with this)
```python
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
```

### Chat completion (streaming)
```python
def chat(messages, model="deepseek-ai/DeepSeek-R1", temp=0.7, max_tokens=500):
    response = client.chat_completion(
        messages=messages, model=model,
        temperature=temp, max_tokens=max_tokens, stream=True
    )
    full = ""
    for chunk in response:
        if hasattr(chunk, 'choices') and chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                full += content
                print(content, end="", flush=True)
    print()
    return full
```

### Chat completion (non-streaming, simpler)
```python
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="deepseek-ai/DeepSeek-R1",
    max_tokens=200,
    stream=False
)
text = response.choices[0].message.content
```

### Messages format
```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},  # optional
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},  # history
    {"role": "user", "content": "Is it good for beginners?"},  # current
]
```

### Local embeddings
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["text1", "text2"])
similarity = util.cos_sim(embeddings[0], embeddings[1]).item()  # 0.0 to 1.0
```

### RAG in 10 lines
```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
chunks = ["chunk1 text", "chunk2 text", "chunk3 text"]
chunk_embeddings = embed_model.encode(chunks)

question = "your question"
q_embedding = embed_model.encode(question)
similarities = util.cos_sim(q_embedding, chunk_embeddings)[0]
best_idx = similarities.argmax().item()
best_chunk = chunks[best_idx]
# Now put best_chunk in your prompt as context
```

### ChromaDB (persistent vector DB)
```python
import chromadb
from sentence_transformers import SentenceTransformer

client_db = chromadb.Client()
collection = client_db.create_collection("my_docs")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Add documents
texts = ["doc1", "doc2", "doc3"]
embeddings = embedder.encode(texts).tolist()
collection.add(documents=texts, embeddings=embeddings, ids=["id1", "id2", "id3"])

# Query
q_emb = embedder.encode(["my question"]).tolist()
results = collection.query(query_embeddings=q_emb, n_results=3)
relevant_docs = results['documents'][0]
```

### Image captioning
```python
with open("photo.jpg", "rb") as f:
    caption = client.image_to_text(f.read(), model="Salesforce/blip-image-captioning-large")
print(caption)
```

### Text-to-image
```python
image = client.text_to_image("a cat on a sofa", model="black-forest-labs/FLUX.1-dev")
image.save("cat.png")  # PIL Image object
```

### Visual Q&A
```python
with open("photo.jpg", "rb") as f:
    result = client.visual_question_answering(
        f.read(), question="What color is it?", model="dandelin/vilt-b32-finetuned-vqa"
    )
```

---

## Free Models Quick Reference

| Task | Model ID | Notes |
|------|----------|-------|
| **Best general chat** | `deepseek-ai/DeepSeek-R1` | Reasoning model, very capable |
| **Fast chat** | `meta-llama/Llama-3.1-8B-Instruct` | Quick, good quality |
| **General chat** | `HuggingFaceH4/zephyr-7b-beta` | Reliable free-tier model |
| **Multilingual** | `Qwen/Qwen2.5-7B-Instruct` | Good for non-English |
| **Local embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, 384-dim |
| **Better embeddings** | `sentence-transformers/all-mpnet-base-v2` | Slower, 768-dim, more accurate |
| **Image captioning** | `Salesforce/blip-image-captioning-large` | Best free captioner |
| **Visual Q&A** | `dandelin/vilt-b32-finetuned-vqa` | Simple visual questions |
| **Text-to-image** | `black-forest-labs/FLUX.1-dev` | Highest quality free |
| **Text-to-image (alt)** | `stabilityai/stable-diffusion-2-1` | Alternative if FLUX unavailable |
| **Vision LLM** | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Complex image understanding |
| **Code** | `bigcode/starcoder2-15b` | Code generation |
| **Fine-tuning base** | `microsoft/phi-2` | Small, fine-tunable on Colab |

---

## Temperature Guide

| Temperature | Use for |
|-------------|---------|
| 0.0 | Code generation, structured output, JSON extraction |
| 0.1–0.3 | Factual Q&A, classification, evaluation |
| 0.4–0.6 | General chat, explanations, summaries |
| 0.7–0.8 | Creative writing, brainstorming |
| 0.9–1.0 | Poetry, experimental, maximum variety |

---

## Prompt Templates

### JSON extraction
```
Extract [WHAT] from the text below.
Return ONLY a JSON object: {"key1": ..., "key2": ...}
If a field is missing, use null. No explanation, no markdown.

Text: [INPUT]
```

### Code review
```
You are a senior [LANGUAGE] engineer. Review this code.
For each issue: Issue | Severity (Critical/High/Medium/Low) | Fix
Code: [CODE]
```

### Summarization
```
Summarize the following [CONTENT TYPE] for [AUDIENCE].
Length: [SHORT/MEDIUM/DETAILED]. Format: [BULLETS/PARAGRAPH]
[CONTENT]
```

### RAG answer
```
Answer the question based ONLY on the provided context.
If not in context, say "I couldn't find that in the document."
Context: [RETRIEVED CHUNKS]
Question: [USER QUESTION]
```

---

## Error Quick Reference

| Error | Fix |
|-------|-----|
| `Model not supported by any provider` | Switch to `deepseek-ai/DeepSeek-R1` |
| `Rate limit exceeded` | Add retry with `time.sleep(60)` |
| `404 Not Found` | Use `chat_completion` not `text_generation` |
| `Invalid token` | Check `.env` file, token starts with `hf_` |
| `json.JSONDecodeError` | Strip thinking blocks: `re.sub(r'<think>.*?</think>', '', text)` |
| `IndexError on chunk.choices` | Add `if hasattr(chunk, 'choices') and chunk.choices:` check |
| `CUDA out of memory` | Reduce batch size, use 4-bit quantization (QLoRA) |

---

## Project Files Summary

| File | Command | What it does |
|------|---------|-------------|
| `projects/prompt_playground.py` | `streamlit run ...` | A/B test prompts with temperature control |
| `projects/rag_pdf_chat.py` | `streamlit run ...` | Chat with any PDF |
| `projects/agent.py` | `streamlit run ...` | Multi-tool AI agent |
| `projects/evaluate.py` | `streamlit run ...` | Score and compare AI outputs |
| `projects/multimodal_app.py` | `streamlit run ...` | Image captioning, VQA, text-to-image |

---

## Decision Guide

```
Do you need the AI to know YOUR specific documents/data?
  → YES → RAG (03_rag.md)
  
Do you need the AI to take actions in the real world?
  → YES → Function Calling (04a, 04b)
  
Do you need to change the AI's personality/style/format consistently?
  → YES → Fine-tuning (05a, 05b)
  
Is the AI's output bad/unreliable?
  → Need better quality → Improve system prompt or use RAG
  → Need to measure how bad → Evaluation (06_evaluation.md)
  
Do you need to understand images?
  → YES → Multi-modal (07_multimodal.md)
  
Want to deploy publicly?
  → YES → Production (10_production.md)
```

Happy building! 🚀
