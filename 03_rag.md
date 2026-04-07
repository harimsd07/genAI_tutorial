# 03 – Retrieval-Augmented Generation (RAG)

> **Core problem solved**: LLMs are trained on public data up to a cutoff date. They know nothing about your private documents, your company's internal knowledge, or recent events. RAG solves this without retraining.

---

## 🧠 Understanding the Problem RAG Solves

Imagine you hired a brilliant consultant who has read every public book and website ever written — but has never seen your company's documents. When you ask "What is our refund policy?", they'd have to guess.

You have two options:
1. **Fine-tune** the model on your documents (expensive, slow, inflexible)
2. **RAG**: Before they answer, hand them the relevant pages from your documents

RAG is option 2. It's faster, cheaper, and lets you update your documents without retraining.

---

## 📐 How RAG Works — Every Step Explained

```
INDEXING PHASE (done once, offline):
─────────────────────────────────────

  Your Document
  ┌────────────────────────────────────────────┐
  │ "Our return policy allows customers to     │
  │  return items within 30 days with receipt. │
  │  Electronics must be unopened..."          │
  └────────────────────────────────────────────┘
          │
          ▼
  [1. CHUNKING]
  Split into small overlapping pieces (~200-500 words each)
  
  Chunk 1: "Our return policy allows customers to return 
            items within 30 days with receipt."
  Chunk 2: "items within 30 days with receipt. Electronics 
            must be unopened..."
  Chunk 3: "Electronics must be unopened..."
  
  ↑ Note the overlap — prevents losing context at boundaries
          │
          ▼
  [2. EMBEDDING]
  Convert each chunk to a vector (array of numbers)
  
  Chunk 1 → [0.23, -0.87, 0.41, 0.09, ..., 0.55]  (384 or 768 numbers)
  Chunk 2 → [0.19, -0.91, 0.38, 0.11, ..., 0.60]
  Chunk 3 → [0.08, -0.45, 0.72, -0.23, ..., 0.31]
  
  Semantically similar text → numerically close vectors
          │
          ▼
  [3. STORE IN VECTOR DATABASE]
  ChromaDB, Pinecone, Weaviate, etc.
  Fast similarity search over millions of vectors

QUERY PHASE (happens on every user question):
──────────────────────────────────────────────

  User: "How long do I have to return something?"
          │
          ▼
  [4. EMBED THE QUESTION]
  "How long do I have to return something?" → [0.21, -0.88, 0.40, ...]
  (Same embedding model as indexing phase!)
          │
          ▼
  [5. SIMILARITY SEARCH]
  Compare question vector to all chunk vectors
  Find the top-K most similar chunks (e.g., K=3)
  
  Similarity scores:
  Chunk 1: 0.97 ← very similar (about return timeframe)
  Chunk 2: 0.89 ← similar
  Chunk 3: 0.41 ← not very relevant
          │
          ▼
  [6. INJECT INTO PROMPT]
  
  System: "Answer based ONLY on the provided context. 
           If the answer isn't in the context, say so."
  
  Context (retrieved):
  "Our return policy allows customers to return items 
   within 30 days with receipt. Electronics must be unopened."
  
  User: "How long do I have to return something?"
          │
          ▼
  [7. LLM GENERATES GROUNDED ANSWER]
  "According to the return policy, you have 30 days to 
   return items, as long as you have your receipt."
```

---

## 🔑 Key Concept: Embeddings

**Embeddings are the heart of RAG.** You must understand this.

An embedding model converts text into a list of numbers (a vector). The crucial property: **semantically similar text produces numerically similar vectors**.

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [
    "The dog chased the ball",
    "A puppy ran after a sphere",      # Similar meaning
    "The stock market crashed today",  # Unrelated
]

embeddings = model.encode(texts)

# Cosine similarity between text 0 and text 1: ~0.85 (very similar)
# Cosine similarity between text 0 and text 2: ~0.10 (unrelated)
```

**Cosine similarity** measures how "pointing in the same direction" two vectors are:
- 1.0 = identical meaning
- 0.0 = completely unrelated
- -1.0 = opposite meaning (rare in practice)

This is what allows RAG to find relevant chunks even when the user doesn't use the same exact words as the document.

---

## ⚙️ Chunking Strategy — This Matters More Than You Think

How you split documents dramatically affects RAG quality. Three common strategies:

### 1. Fixed Size (Simple but naive)
```python
def fixed_size_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap to avoid losing context
    return chunks
```
**Problem**: May cut sentences mid-way.

### 2. Sentence-Aware (Better)
```python
import re
def sentence_aware_chunks(text, max_chars=800, overlap_sentences=1):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Keep last sentence as overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_len = sum(len(s) for s in current_chunk)
        current_chunk.append(sentence)
        current_len += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
```

### 3. Semantic Chunking (Best, but complex)
Split at topic boundaries detected by embedding similarity drops. We won't implement this here — sentence-aware is sufficient for learning.

**Rule of thumb for chunk size**:
- Small chunks (100–200 words): high precision, may miss context
- Medium chunks (300–500 words): good balance, use this for most cases
- Large chunks (500–1000 words): more context, but may include irrelevant text

---

## 🏗️ Project: Chat with Your PDF

We'll build a full RAG pipeline: upload a PDF, index it, ask questions.

### Project Structure
```
genai_learning/
└── projects/
    └── rag_pdf_chat.py
```

### Full Code: `projects/rag_pdf_chat.py`

```python
"""
RAG PDF Chat – Ask questions about any PDF document.
Run with: streamlit run projects/rag_pdf_chat.py
"""

import streamlit as st
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re
import hashlib

load_dotenv()

# ─── Initialize models ────────────────────────────────────────
@st.cache_resource
def load_embedder():
    """Load embedding model once and cache it."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_chroma_client():
    """Create persistent ChromaDB client."""
    return chromadb.Client()

embedder = load_embedder()
chroma_client = get_chroma_client()
hf_client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# ─── Helper Functions ─────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes):
    """Extract all text from PDF bytes."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n[Page {page_num + 1}]\n{text}"
    return full_text

def chunk_text(text, chunk_size=400, overlap=50):
    """Split text into overlapping chunks, respecting sentence boundaries."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence.split())
        
        if current_length + sentence_len > chunk_size and current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text.strip())
            
            # Overlap: keep last few sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s.split()) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s.split())
            
            current_chunk_sentences = overlap_sentences
            current_length = overlap_len
        
        current_chunk_sentences.append(sentence)
        current_length += sentence_len
    
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences).strip())
    
    # Filter out very short chunks (e.g., single words, page headers)
    return [c for c in chunks if len(c.split()) > 10]

def index_document(chunks, collection_name):
    """Create embeddings and store in ChromaDB."""
    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # use cosine similarity
    )
    
    # Embed all chunks
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    
    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection

def retrieve_relevant_chunks(question, collection, top_k=3):
    """Find the most relevant chunks for a question."""
    question_embedding = embedder.encode([question]).tolist()
    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=min(top_k, collection.count())
    )
    
    chunks = results['documents'][0]
    distances = results['distances'][0]
    
    return chunks, distances

def generate_answer(question, context_chunks, chat_history):
    """Generate an answer using retrieved context."""
    context = "\n\n---\n\n".join(context_chunks)
    
    messages = [
        {
            "role": "system",
            "content": """You are a document assistant. Answer questions based ONLY on the provided context.

Rules:
- If the answer is in the context, provide it clearly and concisely.
- If the answer is NOT in the context, say "I couldn't find that information in the document."
- Never make up information not present in the context.
- Quote relevant parts of the context when helpful.
- Keep answers concise (2-4 sentences) unless more detail is needed."""
        },
        {
            "role": "user",
            "content": f"""Context from document:
{context}

Question: {question}"""
        }
    ]
    
    # Add relevant chat history (last 3 exchanges)
    if chat_history:
        history_messages = []
        for exchange in chat_history[-3:]:
            history_messages.append({"role": "user", "content": exchange["question"]})
            history_messages.append({"role": "assistant", "content": exchange["answer"]})
        # Insert history after system message
        messages = [messages[0]] + history_messages + [messages[1]]
    
    full_response = ""
    response = hf_client.chat_completion(
        messages=messages,
        model="deepseek-ai/DeepSeek-R1",
        max_tokens=500,
        temperature=0.1,  # low temp for factual answers
        stream=True
    )
    
    for chunk in response:
        if hasattr(chunk, 'choices') and chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
    
    return full_response

# ─── Streamlit UI ─────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with Your PDF")
st.markdown("_Upload a PDF, and ask questions about its content._")

# Session state for chat history and indexed document
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection" not in st.session_state:
    st.session_state.collection = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# ─── Sidebar: Upload + Index ──────────────────────────────────
with st.sidebar:
    st.header("📂 Document")
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        # Generate a collection name from filename
        collection_name = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:12]
        
        if st.button("📥 Index Document", type="primary"):
            with st.spinner("Reading PDF..."):
                pdf_bytes = uploaded_file.read()
                full_text = extract_text_from_pdf(pdf_bytes)
            
            with st.spinner("Splitting into chunks..."):
                chunks = chunk_text(full_text, chunk_size=400, overlap=50)
            
            with st.spinner(f"Creating embeddings for {len(chunks)} chunks..."):
                collection = index_document(chunks, collection_name)
                st.session_state.collection = collection
                st.session_state.chat_history = []
                st.session_state.doc_stats = {
                    "filename": uploaded_file.name,
                    "total_chars": len(full_text),
                    "num_chunks": len(chunks),
                    "avg_chunk_size": len(full_text) // max(len(chunks), 1)
                }
            
            st.success("✅ Document indexed!")
    
    if st.session_state.doc_stats:
        st.markdown("---")
        st.markdown("**Document Stats**")
        stats = st.session_state.doc_stats
        st.markdown(f"📄 **File**: {stats['filename']}")
        st.markdown(f"📝 **Characters**: {stats['total_chars']:,}")
        st.markdown(f"🧩 **Chunks**: {stats['num_chunks']}")
        st.markdown(f"📏 **Avg chunk**: ~{stats['avg_chunk_size']} chars")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []

# ─── Main Chat Interface ──────────────────────────────────────
if not st.session_state.collection:
    st.info("👈 Upload and index a PDF to get started.")
    
    # Demo with sample text
    st.markdown("---")
    st.markdown("**Or try with a sample text:**")
    sample_text = """
    The Python programming language was created by Guido van Rossum and first released in 1991.
    Python is known for its simple, readable syntax and is widely used in data science, 
    web development, and automation. Python uses indentation to define code blocks, 
    unlike other languages that use braces. The language supports multiple programming 
    paradigms including procedural, object-oriented, and functional programming.
    Python's package manager is called pip, and the main package repository is PyPI.
    """
    if st.button("Load Sample Text"):
        chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
        collection = index_document(chunks, "sample_demo")
        st.session_state.collection = collection
        st.session_state.doc_stats = {
            "filename": "sample_text.txt",
            "total_chars": len(sample_text),
            "num_chunks": len(chunks),
            "avg_chunk_size": len(sample_text) // max(len(chunks), 1)
        }
        st.rerun()

else:
    # Display chat history
    for exchange in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            with st.expander("📎 Retrieved context (what the AI read)"):
                for i, (chunk, dist) in enumerate(zip(exchange["chunks"], exchange["distances"])):
                    similarity = 1 - dist  # cosine: distance 0 = similarity 1
                    st.markdown(f"**Chunk {i+1}** (similarity: {similarity:.2%})")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.markdown("---")
    
    # Question input
    question = st.chat_input("Ask a question about the document...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                chunks, distances = retrieve_relevant_chunks(
                    question, 
                    st.session_state.collection,
                    top_k=3
                )
            
            answer_placeholder = st.empty()
            
            # Stream the answer
            full_answer = generate_answer(question, chunks, st.session_state.chat_history)
            answer_placeholder.markdown(full_answer)
            
            with st.expander("📎 Retrieved context"):
                for i, (chunk, dist) in enumerate(zip(chunks, distances)):
                    similarity = 1 - dist
                    st.markdown(f"**Chunk {i+1}** (similarity: {similarity:.2%})")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                    st.markdown("---")
        
        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": full_answer,
            "chunks": chunks,
            "distances": distances
        })
```

---

## Step 2: Run It

```bash
streamlit run projects/rag_pdf_chat.py
```

---

## Step 3: Test With a Real PDF

Try with any of these:
- A research paper PDF
- Your company's policy document
- A textbook chapter
- The Indian Constitution (PDF available freely online)
- Any long Wikipedia article saved as PDF

**Good test questions** that verify RAG is working:
- Ask for something specific that's in the document
- Ask for something NOT in the document (should say "not found")
- Ask a follow-up question to test conversation memory

---

## 🔬 Understanding Why Similarity Scores Matter

The app shows similarity percentages for retrieved chunks. Use these to debug:

| Similarity | What it means |
|------------|---------------|
| > 85% | Excellent match — highly relevant |
| 70–85% | Good match — probably relevant |
| 50–70% | Weak match — might be relevant |
| < 50% | Poor match — question may not be in document |

If all retrieved chunks show < 50% similarity, the AI is likely making up an answer. This is called **hallucination from insufficient context** — a key failure mode of RAG.

---

## 🧪 Challenges

1. **Add multi-document support**: Let users upload 2+ PDFs and index them all into one collection. Add metadata (filename, page number) to each chunk so the AI can cite its source.

2. **Improve chunking**: Modify the chunk function to split on paragraph breaks (`\n\n`) rather than sentences. Compare answer quality.

3. **Add a "show raw chunks" debug mode**: A toggle that shows the exact text passed to the AI vs what it answers. Use this to diagnose wrong answers.

4. **Add source highlighting**: When the AI answers, highlight which sentence in the original document was the source.

---

## ✅ What You Learned

- **Why RAG exists**: LLMs don't know your private data
- **Embeddings**: How text becomes numbers and enables semantic search
- **Chunking**: Why document splitting strategy matters
- **ChromaDB**: How to store and query vectors locally
- **Grounded generation**: How to prevent hallucinations using retrieved context

Next: [04a_function_calling_theory.md](04a_function_calling_theory.md) — give the AI tools to take actions.
