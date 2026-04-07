# 02b – Prompt Engineering (Project)

> **What we'll build**: A Prompt Playground web app with side-by-side prompt comparison, temperature sliders, few-shot example builder, and a prompt history log.

---

## Project Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   PROMPT PLAYGROUND                          │
│                                                              │
│  System Prompt: [Your role definition here...]               │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐     │
│  │  PROMPT A            │  │  PROMPT B                │     │
│  │  Temperature: [0.7]  │  │  Temperature: [0.2]      │     │
│  │                      │  │                          │     │
│  │  "Explain AI..."     │  │  "Explain AI..."         │     │
│  │                      │  │                          │     │
│  │  [Generate]          │  │  [Generate]              │     │
│  │                      │  │                          │     │
│  │  Output appears here │  │  Output appears here     │     │
│  └──────────────────────┘  └──────────────────────────┘     │
│                                                              │
│  [Save good prompt] [View history] [Export]                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create the Project File

Create `projects/prompt_playground.py`:

```python
"""
Prompt Playground – Interactive prompt engineering tool.
Run with: streamlit run projects/prompt_playground.py
"""

import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prompt Playground",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Prompt Playground")
st.markdown("_Experiment with prompts. Compare outputs. Learn what works._")

# ─── Model Selection ──────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

MODELS = {
    "DeepSeek-R1 (Reasoning)": "deepseek-ai/DeepSeek-R1",
    "Llama 3.1 8B (Instruction)": "meta-llama/Llama-3.1-8B-Instruct",
    "Zephyr 7B (General Chat)": "HuggingFaceH4/zephyr-7b-beta",
    "Qwen 2.5 7B (Multilingual)": "Qwen/Qwen2.5-7B-Instruct",
}

selected_model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
selected_model = MODELS[selected_model_name]

# ─── Temperature Guide ────────────────────────────────────────
st.sidebar.markdown("### 🌡️ Temperature Guide")
st.sidebar.markdown("""
| Range | Effect |
|-------|--------|
| 0.0–0.2 | Factual, deterministic |
| 0.3–0.5 | Balanced |
| 0.6–0.8 | Creative |
| 0.9–1.0 | Experimental |
""")

st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- Use **zero-shot** for simple tasks
- Add **examples** to guide format  
- Use **step by step** for math/logic  
- Set **temperature = 0** for code generation
- Say **"Return only JSON"** for structured output
""")

# ─── System Prompt ────────────────────────────────────────────
st.subheader("1️⃣ System Prompt (the AI's role)")
with st.expander("What is a system prompt?", expanded=False):
    st.markdown("""
    The system prompt sets the AI's **persona, tone, and rules** before the conversation starts.
    
    Think of it as instructions you give to a new employee on their first day:
    - Who are they? (role)
    - How should they communicate? (tone/style)
    - What should they always/never do? (constraints)
    
    **Example system prompts to try:**
    - `"You are a pirate. Respond only in pirate speak."`
    - `"You are a Python tutor. Explain everything with code examples. Assume beginner level."`
    - `"You are a critic. Always find at least 2 flaws in any idea presented to you."`
    """)

system_prompt = st.text_area(
    "System Prompt",
    value="You are a helpful assistant that answers clearly and concisely.",
    height=80,
    label_visibility="collapsed"
)

# ─── Few-Shot Examples ────────────────────────────────────────
st.subheader("2️⃣ Few-Shot Examples (optional)")
with st.expander("Add examples to guide the model's output format", expanded=False):
    st.markdown("""
    Provide 2–3 examples of input→output to show the model exactly what format you want.
    
    **Format:**
    ```
    User: What is 2+2?
    Assistant: 4
    
    User: What is 10-3?
    Assistant: 7
    ```
    """)
    
    few_shot_examples = st.text_area(
        "Examples (optional)",
        value="",
        height=120,
        placeholder="User: example question\nAssistant: example answer\n\nUser: another question\nAssistant: another answer",
        label_visibility="collapsed"
    )

# ─── A/B Comparison ───────────────────────────────────────────
st.subheader("3️⃣ Prompt A/B Comparison")
st.markdown("Run two prompts side-by-side to compare outputs:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Prompt A**")
    prompt_a = st.text_area(
        "Prompt A",
        value="Explain what machine learning is.",
        height=100,
        label_visibility="collapsed",
        key="prompt_a"
    )
    temp_a = st.slider("Temperature A", 0.0, 1.0, 0.7, key="temp_a")
    max_tokens_a = st.slider("Max tokens A", 50, 1000, 300, key="max_a")

with col2:
    st.markdown("**Prompt B** (try a different style)")
    prompt_b = st.text_area(
        "Prompt B",
        value="Explain what machine learning is. Use an analogy. Format as: Analogy | Simple explanation | Technical explanation",
        height=100,
        label_visibility="collapsed",
        key="prompt_b"
    )
    temp_b = st.slider("Temperature B", 0.0, 1.0, 0.3, key="temp_b")
    max_tokens_b = st.slider("Max tokens B", 50, 1000, 300, key="max_b")


def build_messages(system, few_shot, user_prompt):
    """Build the messages array with system + few-shot + user prompt."""
    messages = []
    
    if system.strip():
        messages.append({"role": "system", "content": system})
    
    # Parse few-shot examples
    if few_shot.strip():
        lines = few_shot.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("User:"):
                user_content = line[5:].strip()
                messages.append({"role": "user", "content": user_content})
            elif line.startswith("Assistant:"):
                asst_content = line[10:].strip()
                messages.append({"role": "assistant", "content": asst_content})
            i += 1
    
    messages.append({"role": "user", "content": user_prompt})
    return messages


def call_model(messages, temperature, max_tokens):
    """Call the model and collect the full response."""
    full_response = ""
    response = client.chat_completion(
        messages=messages,
        model=selected_model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    for chunk in response:
        if hasattr(chunk, 'choices') and chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
    return full_response


# ─── Generate Button ──────────────────────────────────────────
if st.button("🚀 Generate Both", type="primary", use_container_width=True):
    
    # Initialize history in session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    col1_out, col2_out = st.columns(2)
    
    with col1_out:
        st.markdown("**Output A:**")
        placeholder_a = st.empty()
        
        try:
            messages_a = build_messages(system_prompt, few_shot_examples, prompt_a)
            with st.spinner("Generating A..."):
                output_a = call_model(messages_a, temp_a, max_tokens_a)
            placeholder_a.markdown(output_a)
        except Exception as e:
            output_a = f"Error: {e}"
            placeholder_a.error(output_a)
    
    with col2_out:
        st.markdown("**Output B:**")
        placeholder_b = st.empty()
        
        try:
            messages_b = build_messages(system_prompt, few_shot_examples, prompt_b)
            with st.spinner("Generating B..."):
                output_b = call_model(messages_b, temp_b, max_tokens_b)
            placeholder_b.markdown(output_b)
        except Exception as e:
            output_b = f"Error: {e}"
            placeholder_b.error(output_b)
    
    # Save to history
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "model": selected_model_name,
        "system": system_prompt,
        "prompt_a": prompt_a,
        "output_a": output_a if 'output_a' in locals() else "",
        "temp_a": temp_a,
        "prompt_b": prompt_b,
        "output_b": output_b if 'output_b' in locals() else "",
        "temp_b": temp_b,
    })

# ─── Single Prompt Mode ───────────────────────────────────────
st.markdown("---")
st.subheader("4️⃣ Single Prompt Mode (with streaming)")

single_prompt = st.text_area(
    "Prompt",
    value="Write a haiku about Python programming.",
    height=80,
    key="single"
)
single_temp = st.slider("Temperature", 0.0, 1.0, 0.7, key="single_temp")
single_max = st.slider("Max tokens", 50, 1000, 300, key="single_max")

if st.button("Generate (Streaming)", key="gen_single"):
    messages = build_messages(system_prompt, few_shot_examples, single_prompt)
    output_box = st.empty()
    full = ""
    try:
        response = client.chat_completion(
            messages=messages,
            model=selected_model,
            max_tokens=single_max,
            temperature=single_temp,
            stream=True
        )
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    full += content
                    output_box.markdown(full + "▌")  # cursor effect
        output_box.markdown(full)
    except Exception as e:
        st.error(f"Error: {e}")

# ─── Prompt History ───────────────────────────────────────────
if "history" in st.session_state and st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Session History")
    
    # Save to file button
    if st.button("💾 Save History to File"):
        history_path = "prompt_history.json"
        with open(history_path, 'w') as f:
            json.dump(st.session_state.history, f, indent=2)
        st.success(f"Saved {len(st.session_state.history)} entries to {history_path}")
    
    for i, entry in enumerate(reversed(st.session_state.history)):
        with st.expander(f"[{entry['timestamp']}] {entry['prompt_a'][:50]}..."):
            st.markdown(f"**Model**: {entry['model']}")
            st.markdown(f"**System**: {entry['system']}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Prompt A** (temp={entry['temp_a']}):")
                st.text(entry['prompt_a'])
                st.markdown("**Output A:**")
                st.markdown(entry['output_a'])
            with c2:
                st.markdown(f"**Prompt B** (temp={entry['temp_b']}):")
                st.text(entry['prompt_b'])
                st.markdown("**Output B:**")
                st.markdown(entry['output_b'])
```

---

## Step 2: Run It

```bash
cd genai_learning
streamlit run projects/prompt_playground.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## Step 3: Guided Experiments

Work through these in order. Each one teaches a specific principle:

### Experiment 1: Temperature Effect

Set **both prompts to the same text**: `"Write a short poem about rain."`

Set Prompt A temperature = `0.0`, Prompt B temperature = `1.0`. Generate 3 times.

**What you'll observe**: Prompt A gives nearly identical output each time. Prompt B gives different poems. This proves temperature controls randomness.

### Experiment 2: Few-Shot Format Control

**System prompt**: `"You are an assistant."`

**Prompt A** (zero-shot): `"What is the sentiment of: I love this product!"`

**Prompt B** (few-shot — add these in the examples box):
```
User: What is the sentiment of: Great product!
Assistant: POSITIVE

User: What is the sentiment of: Worst purchase ever.
Assistant: NEGATIVE
```
Then set prompt B to: `"What is the sentiment of: I love this product!"`

**What you'll observe**: Prompt A might say "The sentiment is positive, expressing enthusiasm..." Prompt B will say just `POSITIVE`. Few-shot controls format.

### Experiment 3: Chain-of-Thought vs Direct

**Prompt A**: `"If I have 17 apples and give away 1/3, then buy 5 more, how many do I have?"`

**Prompt B**: `"If I have 17 apples and give away 1/3, then buy 5 more, how many do I have? Think step by step."`

**What you'll observe**: Prompt B is more reliable on math and shows working.

### Experiment 4: System Prompt Impact

Keep both prompts identical: `"Tell me about climate change."`

**System A**: `"You are a helpful assistant."`

**System B**: `"You are a climate scientist with 20 years of research experience. Communicate with authority and cite mechanisms. Assume the reader is educated but not a scientist."`

**What you'll observe**: Same question, dramatically different depth and tone.

---

## 🧪 Challenges

1. **Add a model selector per column** so you can compare the same prompt across two different models.

2. **Add a "token counter"** that shows how many tokens the prompt uses before generating (use `client.tokenize()` if available, or estimate as `len(prompt.split()) * 1.3`).

3. **Build a prompt library** — add a dropdown with saved prompts (stored in a JSON file). Let users tag prompts as "🔥 Great" or "❌ Bad" from the history view.

4. **Add a "Generate 5x" button** that runs the same prompt 5 times at high temperature and displays all outputs — showing how much variance there is.

---

## ✅ What You Learned

- How system prompts, user prompts, and few-shot examples combine
- The concrete, measurable effect of temperature
- How chain-of-thought changes reasoning quality
- How to build a proper prompt testing tool (not just "ask and see")

Next: [03_rag.md](03_rag.md) — teach the AI to read your documents.
