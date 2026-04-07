"""
Evaluation Harness – Score and compare AI outputs.
Run with: streamlit run projects/evaluate.py

Tabs:
  1. Single test    – evaluate one generated answer
  2. Batch test     – run 5+ questions and get aggregate stats
  3. A/B comparison – compare two system prompts on the same question
"""

import streamlit as st, json, re, pandas as pd
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ── Core evaluation functions ─────────────────────────────────

def semantic_similarity(a: str, b: str) -> float:
    ea = embedder.encode(a, convert_to_tensor=True)
    eb = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())

def llm_judge(question: str, generated: str, golden: str = None) -> dict:
    prompt = f"""Evaluate this AI answer. Return ONLY JSON, no other text.

Question: {question}
Answer: {generated}
{"Reference: " + golden if golden else ""}

JSON: {{"accuracy": 0-10, "clarity": 0-10, "overall": 0-10, "reasoning": "one sentence"}}"""
    try:
        r = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-ai/DeepSeek-R1", max_tokens=200, temperature=0, stream=False
        )
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content, flags=re.DOTALL).strip()
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {"overall": 0, "error": "parse failed"}
    except Exception as e:
        return {"overall": 0, "error": str(e)}

def check_groundedness(answer: str, context: str) -> dict:
    prompt = f"""Is this answer fully supported by the context?
Context: {context}
Answer: {answer}
Return ONLY JSON: {{"grounded": true/false, "score": 0-10, "unsupported": ["list unsupported claims"]}}"""
    try:
        r = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-ai/DeepSeek-R1", max_tokens=200, temperature=0, stream=False
        )
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content, flags=re.DOTALL).strip()
        m    = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group()) if m else {"grounded": False, "score": 0}
    except Exception as e:
        return {"grounded": False, "score": 0, "error": str(e)}

def generate_answer(question: str, system: str = "") -> str:
    msgs = []
    if system: msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": question})
    try:
        r = client.chat_completion(messages=msgs, model="deepseek-ai/DeepSeek-R1",
                                   max_tokens=400, temperature=0.3, stream=False)
        return r.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Evaluator", page_icon="📊", layout="wide")
st.title("📊 AI Evaluation Harness")

tab1, tab2, tab3 = st.tabs(["Single Test", "Batch Test", "A/B Compare"])

with tab1:
    st.header("Evaluate a Single Response")
    c1, c2 = st.columns(2)
    with c1:
        q1  = st.text_area("Question", "What is the capital of France?", key="q1")
        gen = st.text_area("AI Answer", "Paris is the capital of France.", key="gen")
    with c2:
        gld = st.text_area("Golden Answer (optional)", "The capital of France is Paris.", key="gld")
        ctx = st.text_area("Context (optional, for groundedness)", "", key="ctx")
    if st.button("Evaluate", type="primary"):
        with st.spinner("Evaluating…"):
            sim    = semantic_similarity(gen, gld) if gld else None
            scores = llm_judge(q1, gen, gld or None)
            ground = check_groundedness(gen, ctx) if ctx else None
        a, b, c_ = st.columns(3)
        if sim   is not None: a.metric("Semantic Similarity", f"{sim:.0%}")
        b.metric("LLM Overall Score", f"{scores.get('overall', 0)}/10")
        if ground: c_.metric("Grounded", "✅ Yes" if ground.get("grounded") else "❌ No")
        st.markdown("**LLM Judge detail:**"); st.json(scores)
        if ground: st.markdown("**Groundedness:**"); st.json(ground)

with tab2:
    st.header("Batch Evaluation")
    sys_prompt = st.text_area("System Prompt under test", "You are a helpful assistant.", key="bsp")
    default = json.dumps([
        {"question": "What is the capital of France?", "golden_answer": "Paris"},
        {"question": "What is 2 + 2?",                "golden_answer": "4"},
        {"question": "Who wrote Romeo and Juliet?",   "golden_answer": "William Shakespeare"},
        {"question": "What is Python used for?",      "golden_answer": "Python is used for web development, data science, automation, and AI/ML."},
    ], indent=2)
    cases_json = st.text_area("Test Cases (JSON)", default, height=200)
    if st.button("Run Batch", type="primary"):
        try:
            cases = json.loads(cases_json)
        except Exception:
            st.error("Invalid JSON."); st.stop()
        results, prog, status = [], st.progress(0), st.empty()
        for i, case in enumerate(cases):
            status.text(f"Testing {i+1}/{len(cases)}: {case['question'][:50]}…")
            ans    = generate_answer(case["question"], sys_prompt)
            golden = case.get("golden_answer", "")
            sim    = semantic_similarity(ans, golden) if golden else None
            scores = llm_judge(case["question"], ans, golden or None)
            results.append({"question": case["question"], "generated": ans,
                            "golden": golden, "similarity": sim, "llm_scores": scores})
            prog.progress((i+1)/len(cases))
        status.text("✅ Done!")
        sims   = [r["similarity"] for r in results if r["similarity"] is not None]
        llm_sc = [r["llm_scores"].get("overall", 0) for r in results]
        a, b, c_ = st.columns(3)
        if sims:   a.metric("Avg Similarity", f"{sum(sims)/len(sims):.0%}")
        if llm_sc: b.metric("Avg LLM Score",  f"{sum(llm_sc)/len(llm_sc):.1f}/10")
        c_.metric("Questions", len(results))
        rows = [{"Q": r["question"][:60], "Generated": r["generated"][:80],
                 "Similarity": f"{r['similarity']:.0%}" if r["similarity"] else "N/A",
                 "LLM Score": r["llm_scores"].get("overall", "N/A")} for r in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("⬇️ Download Results", json.dumps(results, indent=2),
                           "eval_results.json", "application/json")

with tab3:
    st.header("A/B System Comparison")
    c1, c2 = st.columns(2)
    with c1: sys_a = st.text_area("System A", "You are a helpful assistant. Be concise.", key="sa")
    with c2: sys_b = st.text_area("System B", "You are a helpful assistant. Be detailed and use examples.", key="sb")
    cq  = st.text_area("Test question", "Explain what machine learning is.")
    cgl = st.text_area("Golden answer (optional)", "")
    if st.button("Compare", type="primary"):
        with st.spinner("Running both…"):
            ans_a  = generate_answer(cq, sys_a)
            ans_b  = generate_answer(cq, sys_b)
            scr_a  = llm_judge(cq, ans_a, cgl or None)
            scr_b  = llm_judge(cq, ans_b, cgl or None)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**System A:**"); st.markdown(ans_a)
            st.json(scr_a)
        with c2:
            st.markdown("**System B:**"); st.markdown(ans_b)
            st.json(scr_b)
        oa, ob = scr_a.get("overall", 0), scr_b.get("overall", 0)
        if   oa > ob: st.success(f"🏆 System A wins! ({oa} vs {ob})")
        elif ob > oa: st.success(f"🏆 System B wins! ({ob} vs {oa})")
        else:         st.info("It's a tie!")
