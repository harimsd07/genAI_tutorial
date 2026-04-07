"""
RAG PDF Chat – Ask questions about any PDF.
Run with: streamlit run projects/rag_pdf_chat.py

What this does:
  - Upload a PDF and index it into ChromaDB
  - Ask questions in a chat interface
  - Shows which chunks were retrieved (with similarity scores)
  - Maintains conversation context
"""

import streamlit as st
import fitz                          # pip install pymupdf
import chromadb
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os, re, hashlib

load_dotenv()

# ── Resource init (cached so they only load once) ─────────────
@st.cache_resource
def load_resources():
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    db         = chromadb.Client()
    hf         = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return embedder, db, hf

embedder, chroma, hf_client = load_resources()

# ── Helpers ───────────────────────────────────────────────────

def pdf_to_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    parts = []
    for i, page in enumerate(doc):
        t = page.get_text().strip()
        if t:
            parts.append(f"[Page {i+1}]\n{t}")
    return "\n\n".join(parts)

def chunk_text(text, size=400, overlap=50):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        wc = len(s.split())
        if cur_len + wc > size and cur:
            chunks.append(" ".join(cur).strip())
            tail, tlen = [], 0
            for x in reversed(cur):
                xw = len(x.split())
                if tlen + xw > overlap:
                    break
                tail.insert(0, x); tlen += xw
            cur, cur_len = tail, tlen
        cur.append(s); cur_len += wc
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if len(c.split()) > 10]

def index_pdf(pdf_bytes, filename):
    col_name = hashlib.md5(pdf_bytes).hexdigest()[:16]
    try:
        chroma.delete_collection(col_name)
    except Exception:
        pass
    col = chroma.create_collection(col_name, metadata={"hnsw:space": "cosine"})
    text   = pdf_to_text(pdf_bytes)
    chunks = chunk_text(text)
    embs   = embedder.encode(chunks, show_progress_bar=False).tolist()
    col.add(documents=chunks, embeddings=embs, ids=[f"c{i}" for i in range(len(chunks))])
    return col, len(chunks)

def retrieve(question, col, k=3):
    q_emb   = embedder.encode([question]).tolist()
    results = col.query(query_embeddings=q_emb, n_results=min(k, col.count()))
    chunks  = results["documents"][0]
    dists   = results["distances"][0]
    return chunks, [1 - d for d in dists]   # convert distance → similarity

def answer(question, chunks, history):
    context = "\n\n---\n\n".join(chunks)
    sys_msg = {
        "role": "system",
        "content": (
            "Answer using ONLY the provided context. "
            "If the answer is not there, say so. "
            "Be concise (2-4 sentences)."
        )
    }
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    msgs = [sys_msg]
    for ex in history[-3:]:
        msgs += [{"role": "user", "content": ex["q"]}, {"role": "assistant", "content": ex["a"]}]
    msgs.append(user_msg)

    out = ""
    for chunk in hf_client.chat_completion(messages=msgs, model="deepseek-ai/DeepSeek-R1",
                                           max_tokens=400, temperature=0.1, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            c = chunk.choices[0].delta.content
            if c:
                out += c
    return out

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with Your PDF")

if "history"    not in st.session_state: st.session_state.history    = []
if "collection" not in st.session_state: st.session_state.collection = None
if "doc_info"   not in st.session_state: st.session_state.doc_info   = {}

with st.sidebar:
    st.header("📂 Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("📥 Index", type="primary"):
        pdf_bytes = uploaded.read()
        with st.spinner("Indexing…"):
            col, n = index_pdf(pdf_bytes, uploaded.name)
        st.session_state.collection = col
        st.session_state.history    = []
        st.session_state.doc_info   = {"name": uploaded.name, "chunks": n}
        st.success(f"✅ {n} chunks indexed")
    if st.session_state.doc_info:
        st.markdown(f"**File**: {st.session_state.doc_info['name']}")
        st.markdown(f"**Chunks**: {st.session_state.doc_info['chunks']}")
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []

if not st.session_state.collection:
    st.info("👈 Upload and index a PDF to start chatting.")
    st.stop()

# Render history
for ex in st.session_state.history:
    with st.chat_message("user"):
        st.write(ex["q"])
    with st.chat_message("assistant"):
        st.write(ex["a"])
        with st.expander("📎 Retrieved context"):
            for txt, sim in zip(ex["chunks"], ex["sims"]):
                st.markdown(f"**Similarity: {sim:.0%}**")
                st.text(txt[:300] + ("…" if len(txt) > 300 else ""))
                st.markdown("---")

# Input
question = st.chat_input("Ask a question about the document…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            chunks, sims = retrieve(question, st.session_state.collection)
        resp = answer(question, chunks, st.session_state.history)
        st.markdown(resp)
        with st.expander("📎 Retrieved context"):
            for txt, sim in zip(chunks, sims):
                st.markdown(f"**Similarity: {sim:.0%}**")
                st.text(txt[:300] + ("…" if len(txt) > 300 else ""))
                st.markdown("---")
    st.session_state.history.append({"q": question, "a": resp, "chunks": chunks, "sims": sims})
