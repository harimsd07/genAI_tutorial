# 📖 Code Explanation: `evaluate.py`

> **Purpose**: Every line of `projects/evaluate.py` explained — how to measure AI quality using semantic similarity, LLM-as-judge, and groundedness checking.

---

## 🧱 Big Picture

```
Problem: How do you know if your AI system is any good?
You can't read 10,000 responses manually.

Solution: Use numbers to measure quality automatically.

Metric 1: Semantic Similarity
  → "Is the generated answer saying the same thing as the correct answer?"
  → Works like: Convert both to vectors, measure how similar the vectors are
  → Returns a number 0.0 to 1.0 (1.0 = identical meaning)

Metric 2: LLM-as-Judge
  → "Ask a second AI to rate the first AI's response on a scale of 0-10"
  → Returns structured scores with reasoning

Metric 3: Groundedness
  → "Is everything the AI said actually supported by the source document?"
  → Detects hallucinations in RAG systems

Think of it as a quality control department for your AI outputs.
```

---

## 📦 Section 1: Imports

```python
import streamlit as st, json, re, pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
```

**New here: `pandas as pd`**

**What `pandas` is**: The most popular Python library for working with structured data (tables, spreadsheets, databases). It provides a `DataFrame` object — like an Excel spreadsheet in Python — with powerful operations for filtering, sorting, grouping, and displaying data.

**How we use it**: `pd.DataFrame(rows)` converts a list of dictionaries into a table, which `st.dataframe()` then displays as an interactive table in the web app.

**Analogy**: `pandas` is like Microsoft Excel built into Python. Raw data (a list of dictionaries) is like data scattered on scraps of paper. `pd.DataFrame()` neatly organizes it into a proper spreadsheet with column headers and rows.

---

## 🔬 Section 2: Semantic Similarity Function

```python
def semantic_similarity(a: str, b: str) -> float:
    ea = embedder.encode(a, convert_to_tensor=True)
    eb = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())
```

**`convert_to_tensor=True`**: Tells the embedder to return a **PyTorch tensor** instead of a NumPy array. Tensors are the native data format for PyTorch (the deep learning framework), and `util.cos_sim()` requires tensors.

**PyTorch tensor vs NumPy array**: Both are multi-dimensional arrays of numbers. Tensors can run on GPUs and support automatic differentiation. For our purposes here, the key difference is just the type: `cos_sim()` needs a tensor.

**Analogy**: It's like the difference between a file saved as `.docx` vs `.pdf`. Same content, different format. `convert_to_tensor=True` saves the embedding in `.tensor` format instead of `.numpy` format.

---

**`util.cos_sim(ea, eb)`**: Computes the cosine similarity between two vectors.

**What cosine similarity measures**: The angle between two vectors in high-dimensional space. Two vectors pointing in the same direction have a cosine similarity of 1.0. Two vectors pointing in completely different directions have 0.0.

**Why angle and not distance?**: Distance-based similarity is sensitive to vector magnitude (a long vector and a short vector can point in the same direction but have a large distance). Cosine similarity only cares about direction (meaning), not magnitude (length).

**Analogy**: Two people both pointing at the same mountain have the same "direction" even if one person is taller than the other. Cosine similarity measures where they're pointing, not how tall they are.

---

**`.item()`**: `util.cos_sim()` returns a tensor (even for a single value). `.item()` extracts the single Python float from the tensor. Without this, `cos_sim` would return something like `tensor([[0.95]])` instead of `0.95`.

**`float(...)`**: Ensures the result is a Python float, not some other numeric type.

---

## ⚖️ Section 3: LLM-as-Judge Function

```python
def llm_judge(question: str, generated: str, golden: str = None) -> dict:
    prompt = f"""Evaluate this AI answer. Return ONLY JSON, no other text.
...
JSON: {{"accuracy": 0-10, "clarity": 0-10, "overall": 0-10, "reasoning": "one sentence"}}"""
```

**The scoring prompt design**: We ask for multiple dimensions:
- `accuracy` — is the information correct?
- `clarity` — is it well-expressed?
- `overall` — holistic score

**Why multiple dimensions?**: A response can be clear but wrong (high clarity, low accuracy). Separate dimensions give more actionable feedback — if accuracy is low, improve your RAG. If clarity is low, improve your prompt.

**`golden: str = None`**: Optional parameter. If provided, we ask the judge to compare the generated answer to the golden answer. If not, the judge rates the answer on its own merit.

---

```python
    r = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="deepseek-ai/DeepSeek-R1", max_tokens=200, temperature=0, stream=False
    )
```

**`temperature=0`**: Use 0 for evaluation. We want consistent, reproducible scoring — not random variation. Two identical inputs should always receive the same score.

**`stream=False`**: When we don't need real-time streaming (we just want the final result), `stream=False` is simpler — it returns the complete response directly.

---

```python
    text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content, flags=re.DOTALL).strip()
    m    = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group()) if m else {"overall": 0, "error": "parse failed"}
```

**`r.choices[0].message.content`**: When `stream=False`, the response is a complete message object, not a generator. We access the content directly.

**The pattern**: Strip thinking blocks → find JSON → parse JSON → return. Same robust parsing strategy as in `agent.py`.

---

## 🔍 Section 4: Groundedness Check

```python
def check_groundedness(answer: str, context: str) -> dict:
    prompt = f"""Is this answer fully supported by the context?
Context: {context}
Answer: {answer}
Return ONLY JSON: {{"grounded": true/false, "score": 0-10, "unsupported": ["list unsupported claims"]}}"""
```

**What groundedness means**: In RAG systems, the AI answers from retrieved documents. Groundedness measures "did the AI only say things that were actually in those documents?" An un-grounded answer is one that makes claims not supported by the provided context — a hallucination.

**`"unsupported": ["list unsupported claims"]`**: We ask the judge to list specific claims that aren't in the context. This makes the evaluation actionable — you know exactly what the AI made up.

**Analogy**: A student took an open-book exam (RAG). After marking, the teacher checks: "Were all your answers actually from the textbook, or did you write things from memory/imagination?" The `unsupported` list is the teacher's red pen marking unsupported claims.

---

## 📊 Section 5: The Streamlit Tabs

```python
tab1, tab2, tab3 = st.tabs(["Single Test", "Batch Test", "A/B Compare"])
```

**`st.tabs([...])`**: Creates a tabbed interface — multiple sections you can switch between by clicking tab headers. Returns one context object per tab.

**`with tab1:`, `with tab2:`, `with tab3:`**: Everything indented inside each `with` block appears in that tab.

**Analogy**: `st.tabs` is like a folder with tabbed dividers. You click "Single Test" tab and see those contents; click "Batch Test" and see different contents. Only one tab is visible at a time.

---

## 📦 Section 6: Batch Evaluation

```python
for i, case in enumerate(cases):
    status.text(f"Testing {i+1}/{len(cases)}: {case['question'][:50]}…")
    ans    = generate_answer(case["question"], sys_prompt)
    golden = case.get("golden_answer", "")
    sim    = semantic_similarity(ans, golden) if golden else None
    scores = llm_judge(case["question"], ans, golden or None)
    results.append({...})
    prog.progress((i+1)/len(cases))
```

**`status.text(...)`**: Updates a text placeholder (created with `st.empty()`) on each iteration to show which question is being processed. Without this, the UI would just be frozen.

**`prog.progress((i+1)/len(cases))`**: Updates a progress bar widget from 0.0 to 1.0.  `(i+1)/len(cases)` is the fraction completed — for question 3 of 10 that's `4/10 = 0.4 = 40%`.

---

```python
rows = [{"Q": r["question"][:60], "Generated": r["generated"][:80],
         "Similarity": f"{r['similarity']:.0%}" if r["similarity"] else "N/A",
         "LLM Score": r["llm_scores"].get("overall", "N/A")} for r in results]
st.dataframe(pd.DataFrame(rows), use_container_width=True)
```

**List comprehension creating `rows`**: For each result `r`, creates a display-friendly dictionary with truncated text and formatted numbers.

**`f"{r['similarity']:.0%}" if r["similarity"] else "N/A"`**: Ternary — if similarity exists, format as percentage; if `None` (no golden answer provided), display "N/A".

**`pd.DataFrame(rows)`**: Converts the list of dicts to a pandas DataFrame (table). Column names come from the dictionary keys.

**`st.dataframe(..., use_container_width=True)`**: Renders an interactive table that fills the full width and allows sorting, scrolling, and column resizing.

---

```python
st.download_button("⬇️ Download Results", json.dumps(results, indent=2),
                   "eval_results.json", "application/json")
```

**`st.download_button(...)`**: Creates a button that, when clicked, downloads data as a file to the user's computer.
- Arg 1: button label
- Arg 2: file content (the JSON string)
- Arg 3: suggested filename
- Arg 4: MIME type (tells the browser what kind of file this is)

**`json.dumps(results, indent=2)`**: `json.dumps()` (dump to string) converts Python data to a JSON-formatted string. `indent=2` makes it human-readable. (Compare: `json.dump()` writes to a file; `json.dumps()` returns a string.)

---

## 🔑 New Concepts in This File

| Concept | What it is | Analogy |
|---------|-----------|---------|
| `pandas` / `pd.DataFrame` | Table/spreadsheet in Python | Excel built into Python |
| `convert_to_tensor=True` | Return tensor instead of array | Saving in `.tensor` vs `.numpy` format |
| `util.cos_sim(a, b)` | Cosine similarity between vectors | Measuring the angle between two pointing directions |
| `.item()` | Extract single value from tensor | Unwrapping a single-item gift box |
| `st.tabs([...])` | Tabbed sections in UI | File folder with tabbed dividers |
| `prog.progress(fraction)` | Update a progress bar | Progress bar on a download |
| `json.dumps()` | Convert Python object to JSON string | Translating Python to JSON text |
| `json.dump()` | Write Python object to a JSON file | Translating and printing to a file |
| `st.download_button()` | Button that downloads a file | "Save As" button |
| `any(condition for x in iterable)` | True if any match | "Is there at least one red ball?" |
