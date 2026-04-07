"""
Multi-Tool AI Agent with ReAct loop.
Run with: streamlit run projects/agent.py

Tools available:
  - get_weather(city)        → live weather from wttr.in
  - calculate(expression)    → safe math evaluation
  - get_current_datetime()   → current date and time
  - search_wikipedia(topic)  → Wikipedia summary
  - get_news(topic)          → mock news headlines
"""

import json, re, math, requests, streamlit as st
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# ── Tool functions ────────────────────────────────────────────

def get_weather(city: str) -> str:
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return r.text.strip() if r.status_code == 200 else f"Unavailable for {city}"
    except Exception as e:
        return f"Error: {e}"

def calculate(expression: str) -> str:
    try:
        safe = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        safe.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, safe)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

def get_current_datetime() -> str:
    return datetime.now().strftime("Today is %A, %B %d, %Y. Time: %I:%M %p")

def search_wikipedia(topic: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("extract", "")[:500]
        return f"No article found for '{topic}'"
    except Exception as e:
        return f"Error: {e}"

def get_news(topic: str = "technology") -> str:
    db = {
        "technology": ["New open-source AI model beats GPT-4 on coding benchmarks",
                       "India's tech startup ecosystem hits record funding"],
        "business":   ["Sensex reaches new all-time high",
                       "RBI holds interest rates steady"],
        "sports":     ["India wins test series against Australia 3-1",
                       "Neeraj Chopra breaks javelin world record"],
    }
    for key, headlines in db.items():
        if key in topic.lower():
            return "\n".join(f"• {h}" for h in headlines)
    return "\n".join(f"• {h}" for h in db["technology"])

TOOLS = {
    "get_weather":          {"fn": get_weather,           "desc": "Get current weather for a city.", "params": {"city": "city name"}},
    "calculate":            {"fn": calculate,             "desc": "Evaluate a math expression.",     "params": {"expression": "e.g. '15 * 8 + 100'"}},
    "get_current_datetime": {"fn": get_current_datetime,  "desc": "Get current date and time.",      "params": {}},
    "search_wikipedia":     {"fn": search_wikipedia,      "desc": "Get a Wikipedia summary.",        "params": {"topic": "topic string"}},
    "get_news":             {"fn": get_news,              "desc": "Get news headlines.",             "params": {"topic": "technology | business | sports"}},
}

# ── Agent logic ───────────────────────────────────────────────

SYSTEM = """You are a helpful assistant with tools.

{tools}

To call a tool respond ONLY with JSON (no other text):
{{"tool": "name", "arguments": {{"param": "value"}}}}

To give a final answer respond ONLY with JSON:
{{"response": "your answer"}}
"""

def tool_descriptions():
    lines = []
    for name, t in TOOLS.items():
        lines.append(f"  {name}: {t['desc']} params={t['params']}")
    return "\n".join(lines)

def parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"response": text}

def run_agent(query: str, max_iter: int = 6):
    steps = []
    msgs  = [
        {"role": "system", "content": SYSTEM.format(tools=tool_descriptions())},
        {"role": "user",   "content": query},
    ]
    for _ in range(max_iter):
        raw = ""
        for chunk in client.chat_completion(messages=msgs, model="deepseek-ai/DeepSeek-R1",
                                            max_tokens=300, temperature=0.1, stream=True):
            if hasattr(chunk, "choices") and chunk.choices:
                c = chunk.choices[0].delta.content
                if c:
                    raw += c
        decision = parse(raw)

        if "tool" in decision:
            name = decision["tool"]
            args = decision.get("arguments", {})
            fn   = TOOLS.get(name, {}).get("fn")
            result = fn(**args) if fn else f"Unknown tool: {name}"
            steps.append({"type": "tool", "tool": name, "args": args, "result": result})
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({"role": "user", "content": f"Tool result for {name}: {result}\n\nNow give your final answer."})
        elif "response" in decision:
            steps.append({"type": "final", "text": decision["response"]})
            return decision["response"], steps
        else:
            return raw, steps
    return "Could not complete within step limit.", steps

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 Multi-Tool AI Agent")
st.markdown("_Ask anything — I'll decide which tools to use._")

with st.sidebar:
    st.header("🛠️ Tools")
    for name, t in TOOLS.items():
        with st.expander(f"`{name}`"):
            st.markdown(t["desc"])
            if t["params"]:
                for p, d in t["params"].items():
                    st.markdown(f"- `{p}`: {d}")
    st.markdown("**Try:**")
    examples = [
        "What's the weather in Chennai?",
        "What is 15% of 3500?",
        "What day is today?",
        "Tell me about the Taj Mahal",
        "Latest technology news?",
        "Weather in Mumbai and Delhi, which is hotter?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.prefill = ex

if "agent_history" not in st.session_state: st.session_state.agent_history = []

for ex in st.session_state.agent_history:
    with st.chat_message("user"):      st.write(ex["q"])
    with st.chat_message("assistant"):
        st.write(ex["a"])
        if ex["steps"]:
            with st.expander(f"🔍 {len(ex['steps'])} steps"):
                for s in ex["steps"]:
                    if s["type"] == "tool":
                        st.code(f"Tool: {s['tool']}\nArgs: {s['args']}\nResult: {s['result']}")

prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask anything…")
actual  = query or prefill

if actual:
    with st.chat_message("user"): st.write(actual)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, steps = run_agent(actual)
        st.write(answer)
        if any(s["type"] == "tool" for s in steps):
            with st.expander(f"🔍 {len(steps)} steps"):
                for s in steps:
                    if s["type"] == "tool":
                        st.code(f"Tool: {s['tool']}\nArgs: {s['args']}\nResult: {s['result']}")
    st.session_state.agent_history.append({"q": actual, "a": answer, "steps": steps})
