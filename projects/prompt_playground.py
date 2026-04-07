"""
Prompt Playground – Interactive prompt engineering tool.
Run with: streamlit run projects/prompt_playground.py

What this does:
  - Side-by-side A/B comparison of two prompts
  - Adjustable temperature and max tokens per prompt
  - System prompt + few-shot example support
  - Session history saved to JSON
"""

import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

st.set_page_config(page_title="Prompt Playground", page_icon="🎮", layout="wide")
st.title("🎮 Prompt Playground")
st.markdown("_Experiment with prompts. Compare outputs. Learn what works._")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.header("⚙️ Model")
MODELS = {
    "DeepSeek-R1 (Reasoning)":    "deepseek-ai/DeepSeek-R1",
    "Llama 3.1 8B (Instruction)":  "meta-llama/Llama-3.1-8B-Instruct",
    "Zephyr 7B (General Chat)":    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen 2.5 7B (Multilingual)":  "Qwen/Qwen2.5-7B-Instruct",
}
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
model_id = MODELS[model_name]

st.sidebar.markdown("### 🌡️ Temperature Guide")
st.sidebar.markdown("""
| Range | Effect |
|-------|--------|
| 0.0–0.2 | Factual, deterministic |
| 0.3–0.5 | Balanced |
| 0.6–0.8 | Creative |
| 0.9–1.0 | Experimental |
""")

# ── System Prompt ─────────────────────────────────────────────
st.subheader("1️⃣ System Prompt")
system = st.text_area(
    "System Prompt",
    "You are a helpful assistant that answers clearly and concisely.",
    height=70,
    label_visibility="collapsed",
)

# ── Few-Shot Examples ─────────────────────────────────────────
st.subheader("2️⃣ Few-Shot Examples (optional)")
with st.expander("Add examples — format: 'User: ...' / 'Assistant: ...'"):
    few_shot = st.text_area("Examples", "", height=100, label_visibility="collapsed")

# ── A/B Comparison ────────────────────────────────────────────
st.subheader("3️⃣ A/B Comparison")
c1, c2 = st.columns(2)
with c1:
    prompt_a  = st.text_area("Prompt A", "Explain machine learning.", height=90, key="pa")
    temp_a    = st.slider("Temp A",  0.0, 1.0, 0.7, key="ta")
    tokens_a  = st.slider("Max tokens A", 50, 1000, 300, key="ma")
with c2:
    prompt_b  = st.text_area("Prompt B", "Explain machine learning using a simple everyday analogy, then give the technical definition.", height=90, key="pb")
    temp_b    = st.slider("Temp B",  0.0, 1.0, 0.3, key="tb")
    tokens_b  = st.slider("Max tokens B", 50, 1000, 300, key="mb")


def build_messages(system_txt, few_shot_txt, user_txt):
    msgs = []
    if system_txt.strip():
        msgs.append({"role": "system", "content": system_txt})
    if few_shot_txt.strip():
        for line in few_shot_txt.strip().splitlines():
            if line.startswith("User:"):
                msgs.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith("Assistant:"):
                msgs.append({"role": "assistant", "content": line[10:].strip()})
    msgs.append({"role": "user", "content": user_txt})
    return msgs


def call_model(msgs, temp, max_tok):
    full = ""
    for chunk in client.chat_completion(messages=msgs, model=model_id, temperature=temp, max_tokens=max_tok, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            c = chunk.choices[0].delta.content
            if c:
                full += c
    return full


if st.button("🚀 Generate Both", type="primary", use_container_width=True):
    if "history" not in st.session_state:
        st.session_state.history = []
    out_a = out_b = ""
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Output A:**")
        try:
            with st.spinner("Generating A..."):
                out_a = call_model(build_messages(system, few_shot, prompt_a), temp_a, tokens_a)
            st.markdown(out_a)
        except Exception as e:
            st.error(str(e))
    with r2:
        st.markdown("**Output B:**")
        try:
            with st.spinner("Generating B..."):
                out_b = call_model(build_messages(system, few_shot, prompt_b), temp_b, tokens_b)
            st.markdown(out_b)
        except Exception as e:
            st.error(str(e))
    st.session_state.history.append({
        "ts": datetime.now().strftime("%H:%M:%S"), "model": model_name,
        "system": system, "pa": prompt_a, "oa": out_a, "ta": temp_a,
        "pb": prompt_b, "ob": out_b, "tb": temp_b,
    })

# ── Single Streaming Prompt ───────────────────────────────────
st.markdown("---")
st.subheader("4️⃣ Single Prompt (streaming)")
single = st.text_area("Prompt", "Write a haiku about Python programming.", height=70, key="sp")
s_temp = st.slider("Temperature", 0.0, 1.0, 0.7, key="st")
s_max  = st.slider("Max tokens",  50, 1000, 300, key="sm")

if st.button("Generate", key="gen_s"):
    box = st.empty()
    full = ""
    try:
        for chunk in client.chat_completion(
            messages=build_messages(system, few_shot, single),
            model=model_id, temperature=s_temp, max_tokens=s_max, stream=True
        ):
            if hasattr(chunk, "choices") and chunk.choices:
                c = chunk.choices[0].delta.content
                if c:
                    full += c
                    box.markdown(full + "▌")
        box.markdown(full)
    except Exception as e:
        st.error(str(e))

# ── History ───────────────────────────────────────────────────
if st.session_state.get("history"):
    st.markdown("---")
    st.subheader("📜 Session History")
    if st.button("💾 Save History"):
        with open("prompt_history.json", "w") as f:
            json.dump(st.session_state.history, f, indent=2)
        st.success("Saved to prompt_history.json")
    for entry in reversed(st.session_state.history):
        with st.expander(f"[{entry['ts']}] {entry['pa'][:60]}..."):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Prompt A** (temp={entry['ta']}): {entry['pa']}")
                st.markdown(f"**Output A:** {entry['oa']}")
            with c2:
                st.markdown(f"**Prompt B** (temp={entry['tb']}): {entry['pb']}")
                st.markdown(f"**Output B:** {entry['ob']}")
