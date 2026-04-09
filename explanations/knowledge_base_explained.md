# 📖 Code Explanation: `knowledge_base.py`

> **Purpose**: Every line of `projects/knowledge_base.py` explained — persistent vector storage, multi-file indexing, and source attribution.

---

## 🧱 Big Picture

```
rag_pdf_chat.py: one PDF, in-memory DB, resets on restart
knowledge_base.py: MANY files, persistent DB, survives restarts

New concepts introduced:
  - PersistentClient: saves ChromaDB to disk
  - Source metadata: track which document each chunk came from
  - Multi-file indexing: index PDFs, TXT, MD files
  - Deduplication: don't re-index the same file twice
  - Source-filtered search: "search only in document X"
```

---

## 🗄️ Section 1: Persistent ChromaDB

```python
DB_PATH = "./knowledge_base_db"

@st.cache_resource
def init():
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    db  = chromadb.PersistentClient(path=DB_PATH)
    col = db.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})
    hf  = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return emb, col, hf
```

**`chromadb.PersistentClient(path=DB_PATH)`**: Creates a ChromaDB client that saves all data to the `./knowledge_base_db/` folder on disk.

**`chromadb.Client()` vs `chromadb.PersistentClient()`**:
- `Client()` — stores everything in RAM. Data disappears when you close the app.
- `PersistentClient(path=...)` — stores to disk. Data survives app restarts.

**Analogy**:
- `Client()` is like writing notes on a whiteboard. Erase the board (close the app) and everything is gone.
- `PersistentClient()` is like writing in a notebook. Close the notebook (close the app), reopen it, and your notes are still there.

---

**`db.get_or_create_collection("kb", ...)`**: Two operations in one:
- If collection "kb" already exists in the database, return it
- If it doesn't exist yet, create it first, then return it

**Why not just `create_collection()`?**: If you use `create_collection()` and the collection already exists (from a previous run), it throws an error. `get_or_create` is idempotent — you can call it 100 times and it behaves correctly every time.

**Analogy**: "Find or create a folder named 'kb'." If the folder already exists, open it. If not, make a new one then open it. Either way, you end up with the folder open.

---

## 📂 Section 2: Multi-Format File Extraction

```python
def extract(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n\n".join(f"[Page {i+1}] {p.get_text().strip()}"
                           for i, p in enumerate(doc) if p.get_text().strip())
    return file_bytes.decode("utf-8", errors="replace")
```

**`filename.lower().endswith(".pdf")`**: 
- `.lower()` — converts filename to lowercase ("Report.PDF" → "report.pdf")
- `.endswith(".pdf")` — returns True if the string ends with ".pdf"
- Combined: handles "file.PDF", "file.Pdf", "file.pdf" all correctly

**Generator expression inside `join()`**: `"\n\n".join(f"[Page {i+1}]..." for i, p in enumerate(doc) if p.get_text().strip())` — this creates each formatted page string one at a time and joins them. The `if p.get_text().strip()` part skips blank pages.

**`file_bytes.decode("utf-8", errors="replace")`**: For non-PDF files (`.txt`, `.md`), decode the raw bytes as UTF-8 text. `errors="replace"` handles any bytes that aren't valid UTF-8 by replacing them with the `?` character instead of raising an error.

**Analogy**: UTF-8 is the most common text encoding, but some files use other encodings. `errors="replace"` is like saying "if you encounter a foreign word you can't translate, write '???' instead of giving up."

---

## 🔑 Section 3: Deduplication in `index_file()`

```python
def index_file(file_bytes, filename):
    doc_hash = hashlib.md5(file_bytes).hexdigest()
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        return f"ℹ️ Already indexed ({len(existing['ids'])} chunks)"
```

**Deduplication check**: Before indexing, we check if we've already indexed a file with this filename. If chunks with `source=filename` already exist in the collection, we skip re-indexing.

**`collection.get(where={"source": filename})`**: Retrieves all items from the collection whose `source` metadata field equals `filename`. Returns a dict with `"ids"`, `"documents"`, `"metadatas"` keys.

**`if existing["ids"]:`**: If the list of IDs is non-empty, we've already indexed this file.

**Why check filename instead of hash?**: The hash is the "real" unique identifier, but we check filename for the user's benefit. If they upload a new version of the same file with the same name, they'll know to delete the old one first.

---

```python
    metas  = [{"source": filename, "doc_hash": doc_hash} for _ in chunks]
```

**Metadata per chunk**: Every chunk gets the same metadata — which file it came from. This is crucial for:
1. **Source attribution**: When answering, we can say "According to policy.pdf..."
2. **Source filtering**: Let users search only within one document
3. **Deletion**: Remove all chunks from a specific file: `collection.delete(where={"source": filename})`

**`for _ in chunks`**: Iterates over chunks but ignores the value (using `_`). We just need to create one metadata dict per chunk, all identical.

---

## 🔎 Section 4: Filtered Search

```python
def search(query, top_k=4, source_filter=None):
    ...
    kwargs = {"query_embeddings": q_emb, "n_results": min(top_k, collection.count())}
    if source_filter and source_filter != "All":
        kwargs["where"] = {"source": source_filter}
    res = collection.query(**kwargs)
```

**`kwargs` dictionary**: Instead of building different function calls for the filtered vs unfiltered case, we build a `kwargs` (keyword arguments) dictionary and optionally add the `where` filter. Then we unpack it with `**kwargs`.

**Why `**kwargs` instead of if/else?**: Cleaner than:
```python
if source_filter:
    res = collection.query(query_embeddings=q_emb, n_results=..., where={"source": source_filter})
else:
    res = collection.query(query_embeddings=q_emb, n_results=...)
```

**Analogy**: Building a shopping list in a basket (`kwargs`), then adding optional items (the `where` filter) based on conditions, then going shopping (`collection.query(**kwargs)`).

---

## 🤖 Section 5: Answering with Source Attribution

```python
def answer(question, retrieved):
    context = "\n\n---\n\n".join(
        f"[{r['source']} | {r['sim']:.0%}]\n{r['text']}" for r in retrieved
    )
```

**Generator expression inside `join()`**: For each retrieved item, creates a formatted string with:
- `[filename | similarity%]` — source header
- The actual chunk text below it

**Why include source in context?**: When we inject the source header, the AI can see which document each piece came from. We then instruct the AI to cite its sources in the response.

**Analogy**: Imagine a research assistant handing you papers from different journals. Each paper has a label: "[Nature Journal | 95% match]". This way, when you write your report, you know which journal to cite.

---

## 🖥️ Section 6: Dynamic Source Dropdown

```python
def get_sources():
    if collection.count() == 0: return []
    return sorted(set(m["source"] for m in collection.get()["metadatas"]))
```

**`collection.get()["metadatas"]`**: Retrieves all metadata from the collection as a list of dicts.

**`m["source"] for m in collection.get()["metadatas"]`**: Generator that extracts the `"source"` value from each metadata dict.

**`set(...)`**: Converts to a set — removes duplicates (since many chunks share the same source file).

**`sorted(...)`**: Converts back to a list and alphabetically sorts the filenames.

**Real-time**: This function is called every time the UI renders, so the dropdown always reflects the current state of the database — newly indexed files appear immediately.

---

```python
if c2.button("🗑️", key=f"rm_{src}"):
    collection.delete(where={"source": src})
    st.rerun()
```

**`st.rerun()`**: Forces Streamlit to immediately restart the script from the top. Used here to refresh the document list after deletion — the deleted document should disappear from the sidebar immediately.

**Why `key=f"rm_{src}"`**: Multiple delete buttons in a loop all need unique keys. Using the source filename (`rm_policy.pdf`, `rm_guide.txt`) ensures each button is uniquely identified.

---

## 🔑 New Concepts in This File

| Concept | What it is | Analogy |
|---------|-----------|---------|
| `PersistentClient(path=...)` | Saves DB to disk | Writing in a notebook vs whiteboard |
| `get_or_create_collection()` | Safe create-or-open | "Find or make this folder" |
| `.decode("utf-8", errors="replace")` | Convert bytes to text safely | Translate with "???" for unknown words |
| Metadata per chunk | Tag each chunk with source info | Filing papers by origin journal |
| `kwargs` + `**kwargs` | Build args dict dynamically | Optional items in a shopping basket |
| `set()` | Remove duplicates | Unique item list |
| `collection.delete(where=...)` | Remove items by filter | "Delete all papers from this journal" |
| `st.rerun()` | Force immediate script restart | "Refresh the page now" |
| `collection.get()` | Retrieve all items | "Show me everything in this drawer" |

---

---

# 📖 Master Index: All Explanation Files

| File | What it explains |
|------|-----------------|
| `prompt_playground_explained.md` | Streamlit basics, session_state, streaming, A/B testing |
| `rag_pdf_chat_explained.md` | PyMuPDF, ChromaDB, SentenceTransformers, chunking, retrieval |
| `agent_explained.md` | Function calling, ReAct loop, JSON parsing, sandboxed eval, requests |
| `evaluate_explained.md` | Semantic similarity, LLM-as-judge, groundedness, pandas, tabs |
| `multimodal_explained.md` | PIL images, Base64 encoding, BytesIO, data URLs, Vision LLMs |
| `knowledge_base_explained.md` | PersistentClient, deduplication, metadata filtering, source attribution |

---

# 📖 Complete Python Concepts Reference

Every Python concept used across all 6 project files, in one place.

## Import Patterns

| Syntax | Meaning | Example |
|--------|---------|---------|
| `import X` | Import entire module | `import os` |
| `import X as Y` | Import with alias | `import streamlit as st` |
| `from X import Y` | Import specific item | `from dotenv import load_dotenv` |
| `from X import Y, Z` | Import multiple items | `from sentence_transformers import SentenceTransformer, util` |

## Data Structures

| Structure | Syntax | Use case |
|-----------|--------|---------|
| List | `[a, b, c]` | Ordered collection |
| Dict | `{"key": value}` | Key-value lookup |
| Set | `{a, b, c}` | Unique values, no duplicates |
| Tuple | `(a, b, c)` | Immutable ordered collection |
| List comprehension | `[expr for x in iterable if condition]` | Build filtered/transformed lists |
| Dict comprehension | `{k: v for k, v in items}` | Build filtered/transformed dicts |
| Generator expression | `(expr for x in iterable)` | Lazy version of list comprehension |

## Function Patterns

| Pattern | Syntax | Meaning |
|---------|--------|---------|
| Default parameter | `def f(x, y=10):` | `y` is optional, defaults to 10 |
| Type hints | `def f(x: str) -> int:` | Document expected types |
| `*args` | `def f(*args):` | Accept any number of positional args |
| `**kwargs` | `def f(**kwargs):` | Accept any number of keyword args |
| Unpack dict as kwargs | `f(**my_dict)` | `{"a": 1}` becomes `f(a=1)` |

## String Operations

| Operation | Code | Example |
|-----------|------|---------|
| f-string | `f"Hello {name}"` | `f"Hello Alice"` → `"Hello Alice"` |
| Format | `"Hello {name}".format(name="Alice")` | Same result |
| Strip | `s.strip()` | Remove whitespace from both ends |
| Split | `s.split()` | Split on whitespace → list |
| Split regex | `re.split(pattern, s)` | Split on pattern |
| Join | `", ".join(["a","b"])` | `"a, b"` |
| Slice | `s[2:8]` | Characters from index 2 to 7 |
| Tail slice | `s[:60]` | First 60 characters |
| Head slice | `s[-4:]` | Last 4 characters |
| Contains | `"hello" in s` | True/False |
| Startswith | `s.startswith("abc")` | True/False |
| Replace | `s.replace("a", "b")` | Replace all "a" with "b" |
| Lower | `s.lower()` | All lowercase |

## Control Flow

| Pattern | Code | Meaning |
|---------|------|---------|
| Ternary | `a if condition else b` | Compact if/else |
| Short-circuit OR | `a or b` | Return `a` if truthy, else `b` |
| Short-circuit AND | `a and b` | Return `a` if falsy, else `b` |
| Try/except | `try: ... except Exception as e: ...` | Handle errors |
| Try/except/pass | `try: ... except Exception: pass` | Ignore errors silently |

## Streamlit Reference

| Widget | Code | Returns |
|--------|------|---------|
| Title | `st.title("text")` | None |
| Header | `st.header("text")` | None |
| Subheader | `st.subheader("text")` | None |
| Text | `st.write("text")` | None |
| Markdown | `st.markdown("**bold**")` | None |
| Button | `st.button("Click me")` | bool |
| Text input | `st.text_input("Label")` | str |
| Text area | `st.text_area("Label")` | str |
| Slider | `st.slider("Label", min, max, default)` | number |
| Selectbox | `st.selectbox("Label", options)` | selected item |
| Radio | `st.radio("Label", options)` | selected item |
| File uploader | `st.file_uploader("Label", type=["pdf"])` | file object or None |
| Chat input | `st.chat_input("placeholder")` | str or None |
| Progress bar | `st.progress(0.5)` | progress widget |
| Spinner | `with st.spinner("Loading..."):`| context manager |
| Columns | `c1, c2 = st.columns(2)` | column objects |
| Tabs | `t1, t2 = st.tabs(["A","B"])` | tab objects |
| Expander | `with st.expander("Title"):` | context manager |
| Empty placeholder | `box = st.empty()` then `box.markdown(...)` | placeholder |
| Session state | `st.session_state.key = value` | persists between re-runs |
| Stop execution | `st.stop()` | None |
| Force rerun | `st.rerun()` | None |
| Cache resource | `@st.cache_resource` decorator | Runs once only |
| Download button | `st.download_button("Label", data, filename)` | None |
| Error box | `st.error("message")` | None |
| Success box | `st.success("message")` | None |
| Info box | `st.info("message")` | None |
| Chat message | `with st.chat_message("user"):` | context manager |
| Code block | `st.code("code string")` | None |
| DataFrame | `st.dataframe(pd.DataFrame(rows))` | None |

## ChromaDB Reference

| Operation | Code |
|-----------|------|
| In-memory client | `chromadb.Client()` |
| Persistent client | `chromadb.PersistentClient(path="./db")` |
| Create collection | `db.create_collection("name", metadata={"hnsw:space": "cosine"})` |
| Get collection | `db.get_collection("name")` |
| Get or create | `db.get_or_create_collection("name")` |
| Delete collection | `db.delete_collection("name")` |
| Add items | `col.add(documents=[...], embeddings=[...], ids=[...], metadatas=[...])` |
| Query | `col.query(query_embeddings=[...], n_results=3)` |
| Query with filter | `col.query(..., where={"source": "file.pdf"})` |
| Get all | `col.get()` |
| Get filtered | `col.get(where={"source": "file.pdf"})` |
| Delete by filter | `col.delete(where={"source": "file.pdf"})` |
| Count | `col.count()` |
| Upsert | `col.upsert(ids=[...], documents=[...], embeddings=[...])` |
