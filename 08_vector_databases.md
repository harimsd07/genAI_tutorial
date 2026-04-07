# 08 – Vector Databases (Deep Dive)

> **Why a dedicated file?** Vector databases are the backbone of RAG, semantic search, recommendation systems, and memory for AI agents. Understanding them properly unlocks a huge range of applications.

---

## 🧠 What Is a Vector Database?

A traditional database stores rows of structured data and searches by exact match or range:
```sql
SELECT * FROM products WHERE price < 1000 AND category = 'laptop'
```

A vector database stores embeddings (lists of numbers) and searches by **semantic similarity**:
```
"Find me documents that mean something similar to this query"
```

This is fundamentally different — you're searching by *meaning*, not by keywords.

```
Traditional DB search:          Vector DB search:
────────────────────            ────────────────────────────────
"laptop" matches "laptop" ✓     "laptop" matches "notebook computer" ✓
"laptop" matches "notebook" ✗   "laptop" matches "portable workstation" ✓
(exact string match)            (semantic similarity)
```

---

## 📐 How Similarity Search Works Internally

When you query a vector database with "How do I return a product?", it:

1. Converts your query to a vector: `[0.23, -0.87, 0.41, ...]` (384 numbers for MiniLM)
2. Compares this vector against every stored vector using **cosine similarity**
3. Returns the top-K most similar vectors (and their associated text)

### Cosine Similarity — The Math (Simple Version)

Two vectors point in "directions" in high-dimensional space. Cosine similarity measures the angle between them:

```
Cosine similarity = cos(θ) = (A · B) / (|A| × |B|)

Where:
  A · B = dot product (multiply element-wise, then sum)
  |A|   = magnitude of vector A (square root of sum of squares)

Result range: -1.0 to +1.0
  +1.0 = identical direction (same meaning)
   0.0 = perpendicular (unrelated)
  -1.0 = opposite direction (opposite meaning, rare in practice)
```

**In code:**
```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

# Example
a = np.array([0.23, -0.87, 0.41])  # embedding of "return policy"
b = np.array([0.21, -0.90, 0.38])  # embedding of "refund process"
c = np.array([0.88,  0.12, -0.55]) # embedding of "stock market crash"

print(cosine_similarity(a, b))  # ~0.98 (very similar)
print(cosine_similarity(a, c))  # ~0.10 (unrelated)
```

---

## 🗂️ ChromaDB — Complete Guide

ChromaDB is a local, open-source vector database. No server setup, no cloud account, runs in-process.

### Installation
```bash
pip install chromadb
```

### The Three Storage Modes

```python
import chromadb

# Mode 1: In-memory (lost when script ends — good for testing)
client = chromadb.Client()

# Mode 2: Persistent (saved to disk — use for real applications)
client = chromadb.PersistentClient(path="./my_vector_db")

# Mode 3: HTTP client (connects to a running ChromaDB server)
client = chromadb.HttpClient(host="localhost", port=8000)
```

### Collections

A **collection** is like a table in a traditional database — it groups related documents.

```python
# Create collection with cosine similarity metric
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # similarity metric
    # Other options: "l2" (Euclidean), "ip" (inner product)
)

# Get existing collection
collection = client.get_collection("my_documents")

# Get or create (safe — won't error if already exists)
collection = client.get_or_create_collection("my_documents")

# List all collections
collections = client.list_collections()

# Delete collection
client.delete_collection("my_documents")
```

### Adding Documents

```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Your documents
documents = [
    "The return policy allows 30 days with receipt",
    "Electronics must be unopened to qualify for return",
    "Refunds are processed within 5-7 business days",
    "Items purchased on sale are final sale and cannot be returned",
    "International orders are not eligible for free returns",
]

# Metadata — attach any extra info to each document
metadatas = [
    {"source": "policy.pdf", "page": 1, "section": "general"},
    {"source": "policy.pdf", "page": 1, "section": "electronics"},
    {"source": "policy.pdf", "page": 2, "section": "refunds"},
    {"source": "policy.pdf", "page": 2, "section": "sale_items"},
    {"source": "policy.pdf", "page": 3, "section": "international"},
]

# Create embeddings
embeddings = embedder.encode(documents).tolist()

# Add to collection
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]  # must be unique strings
)

print(f"Collection now has {collection.count()} documents")
```

### Querying

```python
# Basic query
question = "Can I return something I bought last month?"
q_embedding = embedder.encode([question]).tolist()

results = collection.query(
    query_embeddings=q_embedding,
    n_results=3  # return top 3 matches
)

# Results structure:
# {
#   'ids': [['doc_0', 'doc_2', 'doc_1']],
#   'documents': [['The return policy allows 30 days...', 'Refunds are processed...', ...]],
#   'metadatas': [[{'source': 'policy.pdf', 'page': 1, ...}, ...]],
#   'distances': [[0.05, 0.18, 0.22]]  # lower = more similar for cosine
# }

for doc, meta, dist in zip(
    results['documents'][0],
    results['metadatas'][0],
    results['distances'][0]
):
    similarity = 1 - dist  # convert distance to similarity score
    print(f"Similarity: {similarity:.2%}")
    print(f"Source: {meta['source']}, Page: {meta['page']}")
    print(f"Text: {doc}\n")
```

### Filtered Queries (Metadata Filtering)

ChromaDB supports filtering by metadata before doing similarity search:

```python
# Only search within a specific section
results = collection.query(
    query_embeddings=q_embedding,
    n_results=3,
    where={"section": "general"}  # metadata filter
)

# Multiple conditions
results = collection.query(
    query_embeddings=q_embedding,
    n_results=3,
    where={
        "$and": [
            {"source": {"$eq": "policy.pdf"}},
            {"page": {"$gte": 2}}  # page 2 or higher
        ]
    }
)

# Filter by document content (text search within results)
results = collection.query(
    query_embeddings=q_embedding,
    n_results=3,
    where_document={"$contains": "electronics"}
)
```

### Updating and Deleting

```python
# Update a document (and its embedding)
new_text = "The return policy now allows 60 days with receipt"
collection.update(
    ids=["doc_0"],
    documents=[new_text],
    embeddings=embedder.encode([new_text]).tolist(),
    metadatas=[{"source": "policy_v2.pdf", "page": 1}]
)

# Upsert (insert or update — safer for production)
collection.upsert(
    ids=["doc_0"],
    documents=[new_text],
    embeddings=embedder.encode([new_text]).tolist(),
)

# Delete specific documents
collection.delete(ids=["doc_3", "doc_4"])

# Delete by metadata filter
collection.delete(where={"section": "sale_items"})

# Check count after deletion
print(f"Remaining: {collection.count()}")
```

---

## 🏗️ Project: Multi-Document Knowledge Base

Build a persistent knowledge base that can index multiple documents and answer questions across all of them, with source attribution.

### Create `projects/knowledge_base.py`

```python
"""
Multi-Document Knowledge Base with source attribution.
Indexes multiple text/PDF files and answers questions across all of them.
Run with: streamlit run projects/knowledge_base.py
"""

import streamlit as st
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re
import hashlib
from pathlib import Path

load_dotenv()

# ─── Initialization ───────────────────────────────────────────
DB_PATH = "./knowledge_base_db"

@st.cache_resource
def init_resources():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    db = chromadb.PersistentClient(path=DB_PATH)
    hf = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    collection = db.get_or_create_collection(
        "knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    return embedder, collection, hf

embedder, collection, hf_client = init_resources()

# ─── Document Processing ──────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF or plain text file."""
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append(f"[Page {i+1}] {text}")
        return "\n\n".join(pages)
    else:
        # Plain text
        return file_bytes.decode("utf-8", errors="replace")

def smart_chunk(text: str, source: str, chunk_size: int = 350, overlap: int = 50):
    """
    Split text into chunks with metadata.
    Returns list of dicts: {text, source, chunk_id, char_start}
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = []
    current_len = 0
    char_pos = 0
    
    for sentence in sentences:
        words = len(sentence.split())
        
        if current_len + words > chunk_size and current:
            chunk_text = " ".join(current).strip()
            if len(chunk_text) > 50:  # skip tiny chunks
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "char_start": char_pos,
                    "word_count": current_len
                })
            
            # Overlap: keep last N words
            overlap_words = []
            overlap_len = 0
            for s in reversed(current):
                s_words = len(s.split())
                if overlap_len + s_words > overlap:
                    break
                overlap_words.insert(0, s)
                overlap_len += s_words
            
            current = overlap_words
            current_len = overlap_len
        
        current.append(sentence)
        current_len += words
        char_pos += len(sentence) + 1
    
    if current:
        chunk_text = " ".join(current).strip()
        if len(chunk_text) > 50:
            chunks.append({
                "text": chunk_text,
                "source": source,
                "char_start": char_pos,
                "word_count": current_len
            })
    
    return chunks

def index_document(file_bytes: bytes, filename: str) -> dict:
    """Process and index a document. Returns indexing stats."""
    # Check if already indexed
    doc_hash = hashlib.md5(file_bytes).hexdigest()
    
    # Check for existing docs from this file
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        return {
            "status": "already_indexed",
            "filename": filename,
            "existing_chunks": len(existing["ids"])
        }
    
    # Extract and chunk
    text = extract_text(file_bytes, filename)
    chunks = smart_chunk(text, filename)
    
    if not chunks:
        return {"status": "error", "message": "No text could be extracted"}
    
    # Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    
    # Prepare ChromaDB inputs
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": c["source"],
            "char_start": c["char_start"],
            "word_count": c["word_count"],
            "doc_hash": doc_hash
        }
        for c in chunks
    ]
    
    # Add to collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return {
        "status": "indexed",
        "filename": filename,
        "chunks": len(chunks),
        "total_words": sum(c["word_count"] for c in chunks)
    }

def search_knowledge_base(question: str, top_k: int = 5, source_filter: str = None) -> list:
    """Search across all indexed documents."""
    q_emb = embedder.encode([question]).tolist()
    
    query_kwargs = {
        "query_embeddings": q_emb,
        "n_results": min(top_k, collection.count())
    }
    
    if source_filter and source_filter != "All documents":
        query_kwargs["where"] = {"source": source_filter}
    
    results = collection.query(**query_kwargs)
    
    items = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        items.append({
            "text": doc,
            "source": meta["source"],
            "similarity": 1 - dist,
        })
    
    return items

def get_indexed_sources() -> list:
    """Get list of all indexed document names."""
    if collection.count() == 0:
        return []
    all_items = collection.get()
    sources = list(set(m["source"] for m in all_items["metadatas"]))
    return sorted(sources)

def answer_question(question: str, retrieved: list) -> str:
    """Generate an answer using retrieved context with source attribution."""
    if not retrieved:
        return "No relevant documents found for this question."
    
    # Format context with source labels
    context_parts = []
    for i, item in enumerate(retrieved):
        context_parts.append(
            f"[Source: {item['source']} | Relevance: {item['similarity']:.0%}]\n{item['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    messages = [
        {
            "role": "system",
            "content": """You are a knowledge base assistant. Answer questions using ONLY the provided context.

Rules:
1. Base your answer strictly on the provided context.
2. At the end of your answer, list which source(s) you used in format: "Sources: filename1, filename2"
3. If the context doesn't contain the answer, say: "I couldn't find this in the indexed documents."
4. Be concise but complete."""
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    full_response = ""
    response = hf_client.chat_completion(
        messages=messages,
        model="deepseek-ai/DeepSeek-R1",
        max_tokens=600,
        temperature=0.1,
        stream=True
    )
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
    
    return full_response

# ─── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="Knowledge Base", page_icon="🗃️", layout="wide")
st.title("🗃️ Multi-Document Knowledge Base")
st.markdown("_Index multiple documents. Ask questions across all of them._")

col_left, col_right = st.columns([1, 2])

# ─── Left Column: Document Management ────────────────────────
with col_left:
    st.subheader("📂 Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📥 Index All Uploaded", type="primary"):
        for f in uploaded_files:
            with st.spinner(f"Indexing {f.name}..."):
                result = index_document(f.read(), f.name)
                if result["status"] == "indexed":
                    st.success(f"✅ {f.name}: {result['chunks']} chunks indexed")
                elif result["status"] == "already_indexed":
                    st.info(f"ℹ️ {f.name}: already indexed ({result['existing_chunks']} chunks)")
                else:
                    st.error(f"❌ {f.name}: {result.get('message', 'Unknown error')}")
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Indexed Documents:**")
    
    sources = get_indexed_sources()
    
    if not sources:
        st.info("No documents indexed yet.")
    else:
        for source in sources:
            items = collection.get(where={"source": source})
            chunk_count = len(items["ids"])
            
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"📄 **{source}**  \n_{chunk_count} chunks_")
            with col_b:
                if st.button("🗑️", key=f"del_{source}", help=f"Remove {source}"):
                    collection.delete(where={"source": source})
                    st.success(f"Removed {source}")
                    st.rerun()
        
        st.markdown(f"**Total chunks in DB:** {collection.count()}")
    
    # Demo: add sample documents
    st.markdown("---")
    if st.button("Add Sample Documents"):
        samples = {
            "python_basics.txt": """
Python is a high-level, interpreted programming language known for its simple syntax.
Python was created by Guido van Rossum and first released in 1991.
Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.
Python uses indentation to define code blocks instead of curly braces.
The Python Package Index (PyPI) hosts over 400,000 packages.
Python is widely used in data science, machine learning, web development, and automation.
Popular Python frameworks include Django and Flask for web development, and NumPy and Pandas for data science.
            """,
            "machine_learning_intro.txt": """
Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
In supervised learning, models are trained on labeled data where the correct answers are provided.
In unsupervised learning, the model finds patterns in data without labels.
Reinforcement learning trains agents to make decisions by rewarding correct actions.
Common machine learning algorithms include linear regression, decision trees, neural networks, and support vector machines.
Deep learning is a subset of machine learning that uses neural networks with many layers.
            """,
        }
        for name, content in samples.items():
            result = index_document(content.encode(), name)
            st.write(f"{name}: {result['status']}")
        st.rerun()

# ─── Right Column: Q&A Interface ─────────────────────────────
with col_right:
    st.subheader("💬 Ask Questions")
    
    if collection.count() == 0:
        st.info("👈 Index some documents first, or click 'Add Sample Documents'")
    else:
        # Source filter
        source_options = ["All documents"] + get_indexed_sources()
        selected_source = st.selectbox("Search in:", source_options)
        
        top_k = st.slider("Number of context chunks", 1, 8, 3)
        
        # Chat history
        if "kb_history" not in st.session_state:
            st.session_state.kb_history = []
        
        # Display history
        for exchange in st.session_state.kb_history:
            with st.chat_message("user"):
                st.write(exchange["question"])
            with st.chat_message("assistant"):
                st.write(exchange["answer"])
                with st.expander("📎 Retrieved chunks"):
                    for item in exchange["retrieved"]:
                        st.markdown(
                            f"**{item['source']}** | Similarity: {item['similarity']:.0%}"
                        )
                        st.text(item["text"][:200] + "...")
                        st.markdown("---")
        
        question = st.chat_input("Ask anything about your documents...")
        
        if question:
            with st.chat_message("user"):
                st.write(question)
            
            with st.chat_message("assistant"):
                with st.spinner("Searching..."):
                    retrieved = search_knowledge_base(question, top_k, selected_source)
                
                if not retrieved:
                    st.warning("No relevant content found.")
                else:
                    answer = answer_question(question, retrieved)
                    st.markdown(answer)
                    
                    with st.expander(f"📎 {len(retrieved)} retrieved chunks"):
                        for item in retrieved:
                            st.markdown(
                                f"**{item['source']}** | "
                                f"Similarity: {item['similarity']:.0%}"
                            )
                            st.text(item["text"][:300] + "...")
                            st.markdown("---")
                    
                    st.session_state.kb_history.append({
                        "question": question,
                        "answer": answer,
                        "retrieved": retrieved
                    })
```

---

## Step 2: Run It

```bash
streamlit run projects/knowledge_base.py
```

The database is **persistent** — documents you index survive between restarts (stored in `./knowledge_base_db/`).

---

## 🔬 Comparing Vector DB Options

| Database | Type | Best for | Free tier |
|----------|------|----------|-----------|
| **ChromaDB** | Local / self-hosted | Learning, small projects, local apps | ✅ Fully free |
| **Pinecone** | Managed cloud | Production at scale, serverless | ✅ Free tier (1 index) |
| **Weaviate** | Self-hosted / cloud | Multi-modal, graph features | ✅ Free cloud tier |
| **Qdrant** | Self-hosted / cloud | High performance, filtering | ✅ Free cloud tier |
| **FAISS** | Library (no server) | Extreme speed, research | ✅ Open source |
| **pgvector** | PostgreSQL extension | Already using Postgres | ✅ Open source |

**When to graduate from ChromaDB to a managed solution:**
- More than ~100,000 documents
- Need multi-user concurrent access
- Need geographic distribution
- Need automatic backups

---

## 🧪 Challenges

1. **Relevance threshold filtering**: Modify the search to only return chunks with similarity above 0.6. Below that threshold, tell the user "I don't have information about this."

2. **Chunk overlap visualization**: Build a debug view that shows two adjacent chunks side by side, highlighting the overlapping sentences between them.

3. **Hybrid search**: Combine vector search with keyword search. First do a keyword filter (`where_document={"$contains": keyword}`), then rank the filtered results by vector similarity. This often gives better precision than pure vector search.

4. **Semantic deduplication**: After indexing a document, find and remove near-duplicate chunks (similarity > 0.95). This keeps the index clean and saves storage.

---

## ✅ What You Learned

- The difference between traditional and vector database search
- How cosine similarity is calculated (with actual math)
- ChromaDB: all CRUD operations, metadata filtering, persistent storage
- Multi-document indexing with source attribution
- When to use ChromaDB vs managed vector databases

Next: [09_llm_internals.md](09_llm_internals.md) — understand what's happening inside the model.
