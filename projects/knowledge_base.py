"""
Multi-Document Knowledge Base with persistent ChromaDB storage.
Run with: streamlit run projects/knowledge_base.py

What this does:
  - Index multiple PDFs / text files into a persistent vector database
  - Ask questions across all documents with source attribution
  - Similarity scores show which chunks were used
  - Survives restarts — documents stay indexed
"""

import streamlit as st, chromadb, fitz, re, hashlib, os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
DB_PATH = "./knowledge_base_db"   # persistent folder next to projects/

@st.cache_resource
def init():
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    db  = chromadb.PersistentClient(path=DB_PATH)
    col = db.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})
    hf  = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    return emb, col, hf

embedder, collection, hf_client = init()

# ── Helpers ───────────────────────────────────────────────────

def extract(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        return "\n\n".join(f"[Page {i+1}] {p.get_text().strip()}"
                           for i, p in enumerate(doc) if p.get_text().strip())
    return file_bytes.decode("utf-8", errors="replace")

def chunk(text, size=350, overlap=50):
    sents   = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur_len + w > size and cur:
            chunks.append(" ".join(cur).strip())
            tail, tlen = [], 0
            for x in reversed(cur):
                xw = len(x.split())
                if tlen + xw > overlap: break
                tail.insert(0, x); tlen += xw
            cur, cur_len = tail, tlen
        cur.append(s); cur_len += w
    if cur: chunks.append(" ".join(cur).strip())
    return [c for c in chunks if len(c.split()) > 10]

def index_file(file_bytes, filename):
    doc_hash = hashlib.md5(file_bytes).hexdigest()
    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        return f"ℹ️ Already indexed ({len(existing['ids'])} chunks)"
    text   = extract(file_bytes, filename)
    chunks = chunk(text)
    if not chunks: return "❌ No text found"
    embs   = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids    = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metas  = [{"source": filename, "doc_hash": doc_hash} for _ in chunks]
    collection.add(documents=chunks, embeddings=embs, metadatas=metas, ids=ids)
    return f"✅ {len(chunks)} chunks indexed"

def search(query, top_k=4, source_filter=None):
    if collection.count() == 0: return []
    q_emb = embedder.encode([query]).tolist()
    kwargs = {"query_embeddings": q_emb, "n_results": min(top_k, collection.count())}
    if source_filter and source_filter != "All":
        kwargs["where"] = {"source": source_filter}
    res = collection.query(**kwargs)
    return [{"text": d, "source": m["source"], "sim": 1 - dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

def answer(question, retrieved):
    if not retrieved: return "No relevant content found."
    context = "\n\n---\n\n".join(
        f"[{r['source']} | {r['sim']:.0%}]\n{r['text']}" for r in retrieved
    )
    msgs = [
        {"role": "system", "content":
         "Answer using ONLY the provided context. "
         "End with 'Sources: filename(s)'. "
         "If not found, say so."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    out = ""
    for chunk in hf_client.chat_completion(messages=msgs, model="deepseek-ai/DeepSeek-R1",
                                           max_tokens=500, temperature=0.1, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            c = chunk.choices[0].delta.content
            if c: out += c
    return out

def get_sources():
    if collection.count() == 0: return []
    return sorted(set(m["source"] for m in collection.get()["metadatas"]))

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Knowledge Base", page_icon="🗃️", layout="wide")
st.title("🗃️ Multi-Document Knowledge Base")

left, right = st.columns([1, 2])

with left:
    st.subheader("📂 Documents")
    uploads = st.file_uploader("Upload PDFs or text files", type=["pdf","txt","md"],
                                accept_multiple_files=True)
    if uploads and st.button("📥 Index All", type="primary"):
        for f in uploads:
            msg = index_file(f.read(), f.name)
            (st.success if msg.startswith("✅") else st.info)(f"{f.name}: {msg}")
        st.rerun()

    st.markdown("---")
    sources = get_sources()
    if not sources:
        st.info("No documents yet.")
        if st.button("Load sample docs"):
            for name, text in [
                ("python_guide.txt", "Python is a high-level interpreted language created by Guido van Rossum in 1991. It is widely used in data science, web development, and automation. Python uses indentation to define code blocks. The package manager is pip. Popular frameworks include Django, Flask, NumPy, and Pandas."),
                ("ml_intro.txt", "Machine learning is a subset of AI that enables computers to learn from data. Types: supervised learning uses labeled data, unsupervised learning finds patterns without labels, reinforcement learning rewards correct actions. Deep learning uses multi-layer neural networks. Common algorithms: linear regression, decision trees, SVMs."),
            ]:
                msg = index_file(text.encode(), name)
                st.write(f"{name}: {msg}")
            st.rerun()
    else:
        for src in sources:
            n = len(collection.get(where={"source": src})["ids"])
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"📄 **{src}** _{n} chunks_")
            if c2.button("🗑️", key=f"rm_{src}"):
                collection.delete(where={"source": src})
                st.rerun()
        st.markdown(f"**Total chunks:** {collection.count()}")

with right:
    st.subheader("💬 Ask Questions")
    if collection.count() == 0:
        st.info("👈 Index some documents first.")
        st.stop()

    src_opts = ["All"] + get_sources()
    selected = st.selectbox("Search in:", src_opts)
    top_k    = st.slider("Chunks to retrieve", 1, 8, 3)

    if "kb_hist" not in st.session_state: st.session_state.kb_hist = []
    for ex in st.session_state.kb_hist:
        with st.chat_message("user"):      st.write(ex["q"])
        with st.chat_message("assistant"):
            st.write(ex["a"])
            with st.expander("📎 Retrieved chunks"):
                for r in ex["retrieved"]:
                    st.markdown(f"**{r['source']}** | {r['sim']:.0%}")
                    st.text(r["text"][:200] + "…"); st.markdown("---")

    q = st.chat_input("Ask anything about your documents…")
    if q:
        with st.chat_message("user"): st.write(q)
        with st.chat_message("assistant"):
            retrieved = search(q, top_k, selected if selected != "All" else None)
            ans       = answer(q, retrieved)
            st.markdown(ans)
            with st.expander(f"📎 {len(retrieved)} chunks retrieved"):
                for r in retrieved:
                    st.markdown(f"**{r['source']}** | {r['sim']:.0%}")
                    st.text(r["text"][:250] + "…"); st.markdown("---")
        st.session_state.kb_hist.append({"q": q, "a": ans, "retrieved": retrieved})
