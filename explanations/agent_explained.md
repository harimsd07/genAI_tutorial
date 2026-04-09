# 📖 Code Explanation: `agent.py`

> **Purpose**: Every line of `projects/agent.py` explained — the tools, the agent loop, and the Streamlit UI — using plain language and real-world analogies.

---

## 🧱 Big Picture

```
An AI model by itself can only generate text.
It can TALK about the weather, but it can't CHECK the weather.
It can TALK about math, but it can't reliably DO the math.
Function calling changes this.

Think of the agent as a manager at a company:
  - The manager (LLM) receives a task from a client
  - The manager doesn't do every task personally
  - Instead, the manager identifies which employee (tool) should handle it
  - The manager assigns the task and reads the employee's report
  - Finally, the manager composes a coherent reply to the client

The LLM never actually executes code.
It just outputs INSTRUCTIONS in JSON format.
Your Python code is the one that actually runs the tools.
```

---

## 📦 Section 1: Imports (Lines 13–20)

```python
import json, re, math, requests, streamlit as st
from datetime import datetime
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
```

Previously explained: `re`, `streamlit`, `InferenceClient`, `load_dotenv`, `os`, `datetime`.

New ones here:

---

```python
import json
```

**What `json` is**: Python's built-in library for working with JSON (JavaScript Object Notation) — a text format for structured data.

**Why it's critical for agents**: When the LLM decides to call a tool, it outputs a JSON string like `{"tool": "get_weather", "arguments": {"city": "Mumbai"}}`. We use `json.loads()` to parse this string into a Python dictionary our code can work with.

**Analogy**: JSON is a universal form that every system agrees on. When the AI fills out the form `{"tool": "get_weather", "city": "Mumbai"}`, our Python code reads the form and knows exactly what to do. `json.loads()` is the form reader.

---

```python
import math
```

**What `math` is**: Python's built-in module providing mathematical functions — `math.sqrt()`, `math.pi`, `math.ceil()`, `math.floor()`, trigonometry functions, logarithms, etc.

**Why we import it**: Our `calculate` tool needs to safely evaluate math expressions. We give it access to `math` functions so users can type things like `sqrt(144)` or `pi * 5**2`.

**Analogy**: `math` is a scientific calculator built into Python. Instead of implementing square root yourself, you use the pre-built one. `math.sqrt(144)` returns `12.0`.

---

```python
import requests
```

**What `requests` is**: A popular third-party Python library for making HTTP requests — fetching web pages, calling APIs, downloading files. It's not built into Python (you install it with `pip install requests`).

**Why it's popular**: Python does have a built-in `urllib` for HTTP requests, but it's verbose and complex. `requests` provides a much simpler, more readable interface for the same operations.

**How we use it**: `requests.get(url)` fetches the content at a URL. We use it to get weather data from `wttr.in` and Wikipedia summaries.

**Analogy**: `requests` is like a delivery driver for your program. You give it an address (URL) and it goes out, fetches the content, and brings it back to you as a Python object you can work with.

---

## 🛠️ Section 2: Tool Functions (Lines 22–65)

### `get_weather()` (Lines 24–29)

```python
def get_weather(city: str) -> str:
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return r.text.strip() if r.status_code == 200 else f"Unavailable for {city}"
    except Exception as e:
        return f"Error: {e}"
```

**`city: str`** and **`-> str`**: These are Python **type hints** — annotations that tell readers (and tools) what types the function expects and returns. `city: str` means "city should be a string." `-> str` means "this function returns a string." Python doesn't enforce these, but they improve code clarity.

**`f"https://wttr.in/{city}?format=3"`**: Builds the URL string. `wttr.in` is a free, no-API-key-required weather service. `?format=3` tells it to return a compact single-line response like `Mumbai: ⛅️  +32°C`.

**`requests.get(..., timeout=5)`**: Makes an HTTP GET request to the URL. `timeout=5` means "if the server doesn't respond within 5 seconds, stop waiting and raise a `Timeout` exception." Without a timeout, a slow server could freeze your app indefinitely.

**Analogy**: `timeout=5` is like calling a restaurant to place a delivery order and saying "if nobody answers within 5 rings, I'm calling somewhere else." It prevents your program from waiting forever.

**`r.status_code`**: HTTP responses always include a numeric status code indicating success or failure:
- 200 = OK (request succeeded)
- 404 = Not Found
- 500 = Server Error
- 429 = Rate Limit Exceeded

**`r.text`**: The response body as a string (the actual content returned by the server).

**`r.text.strip()`**: Removes any trailing newlines from the weather service's response.

**Ternary expression**: `r.text.strip() if r.status_code == 200 else f"Unavailable for {city}"` means: "if the request succeeded, return the weather text; otherwise return an error message." Same as:
```python
if r.status_code == 200:
    return r.text.strip()
else:
    return f"Unavailable for {city}"
```

---

### `calculate()` (Lines 31–38)

```python
def calculate(expression: str) -> str:
    try:
        safe = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        safe.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, safe)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"
```

**`eval(expression, ...)`**: Python's built-in `eval()` function takes a string and executes it as Python code, returning the result. `eval("2 + 2")` returns `4`.

**⚠️ The danger of `eval()`**: Without restrictions, `eval()` can execute *any* Python code. If a user types `eval("import os; os.system('rm -rf /')")`, it would delete files! This is why we use a sandboxed version.

**`math.__dict__`**: Every Python module stores its contents in a `__dict__` dictionary. `math.__dict__` is a dictionary of all functions in the `math` module: `{"sqrt": <function>, "pi": 3.14..., "sin": <function>, ...}`.

**`{k: v for k, v in math.__dict__.items() if not k.startswith("__")}`**: A dictionary comprehension — creates a new dict by filtering out keys that start with `"__"`. Keys starting with `"__"` are Python's internal magic attributes (`__name__`, `__doc__`, etc.) — we don't need those, only the actual math functions.

**`safe.update({...})`**: Adds a few more safe functions to our allowed namespace: `abs` (absolute value), `round`, `min`, `max`. These are Python built-ins, not in `math`.

**`eval(expression, {"__builtins__": {}}, safe)`**: The three-argument form of `eval`:
- Arg 1: the expression to evaluate
- Arg 2: global namespace — `{"__builtins__": {}}` = empty globals, blocking ALL dangerous built-ins like `import`, `open`, `exec`
- Arg 3: local namespace — `safe` = only our allowed math functions

**Combined effect**: `eval` can now use math functions (from `safe`) but cannot `import` modules, call `open()`, access `__builtins__`, or do anything dangerous.

**Analogy**: A normal `eval` is like giving someone the master key to your entire house. The sandboxed `eval` is like giving them a key that only opens the math lab — they can do calculations there, but can't enter any other room.

---

### `get_current_datetime()` (Lines 40–41)

```python
def get_current_datetime() -> str:
    return datetime.now().strftime("Today is %A, %B %d, %Y. Time: %I:%M %p")
```

**`datetime.now()`**: Returns the current local date and time as a `datetime` object.

**`.strftime("Today is %A, %B %d, %Y. Time: %I:%M %p")`**: Format codes:
- `%A` = full weekday name (e.g., "Wednesday")
- `%B` = full month name (e.g., "January")
- `%d` = day of month with leading zero (e.g., "05")
- `%Y` = 4-digit year (e.g., "2024")
- `%I` = hour in 12-hour format (e.g., "03")
- `%M` = minutes (e.g., "45")
- `%p` = AM/PM

Result: `"Today is Wednesday, January 05, 2024. Time: 03:45 PM"`

---

### `search_wikipedia()` (Lines 43–51)

```python
def search_wikipedia(topic: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json().get("extract", "")[:500]
        return f"No article found for '{topic}'"
    except Exception as e:
        return f"Error: {e}"
```

**`topic.replace(' ', '_')`**: Wikipedia URLs use underscores instead of spaces. `"Taj Mahal"` → `"Taj_Mahal"`.

**Wikipedia REST API**: Wikipedia provides a free REST API that requires no authentication. The `/page/summary/{title}` endpoint returns a JSON object with the article's summary.

**`r.json()`**: Parses the response body as JSON and returns a Python dictionary. This is equivalent to `json.loads(r.text)`.

**`.get("extract", "")`**: Dictionary `.get(key, default)` — returns the value for `"extract"` key if it exists, otherwise returns `""`. Safer than `d["extract"]` which would throw `KeyError` if the key doesn't exist.

**`[:500]`**: Limits the Wikipedia summary to 500 characters. Full Wikipedia summaries can be very long — we only need a brief overview to give the AI context.

---

### `get_news()` (Lines 53–65)

```python
def get_news(topic: str = "technology") -> str:
    db = {
        "technology": [...],
        "business":   [...],
        "sports":     [...],
    }
    for key, headlines in db.items():
        if key in topic.lower():
            return "\n".join(f"• {h}" for h in headlines)
    return "\n".join(f"• {h}" for h in db["technology"])
```

**`topic: str = "technology"`**: Default parameter — if `get_news()` is called without arguments, `topic` defaults to `"technology"`.

**`topic.lower()`**: Converts the topic string to lowercase. This makes matching case-insensitive: "Technology", "TECHNOLOGY", and "technology" all match.

**`if key in topic.lower()`**: Checks if the dict key (e.g., "technology") appears anywhere in the topic string. So a query like "latest technology news" would match the "technology" key.

**`"\n".join(f"• {h}" for h in headlines)`**: A **generator expression** (like a list comprehension but lazy — doesn't create the list, just generates items on demand). For each headline `h`, creates `"• headline text"`. Then `"\n".join(...)` puts newlines between them.

Result:
```
• New open-source AI model beats GPT-4 on coding benchmarks
• India's tech startup ecosystem hits record funding
```

**Fallback `return`**: If no topic matches, defaults to technology news. Prevents the function from returning `None`.

---

## 📋 Section 3: Tool Registry (Lines 67–73)

```python
TOOLS = {
    "get_weather": {
        "fn": get_weather,
        "desc": "Get current weather for a city.",
        "params": {"city": "city name"}
    },
    ...
}
```

**What this is**: A dictionary that maps each tool name to:
- `"fn"` — the actual Python function to call
- `"desc"` — a description written FOR THE LLM (helps it decide when to use this tool)
- `"params"` — what arguments the function needs

**Why a registry?**: Instead of a giant `if/elif` chain to decide which function to call, we look up the function in this dictionary: `TOOLS["get_weather"]["fn"]` gives us the `get_weather` function directly.

**Analogy**: `TOOLS` is like a company's employee directory. Each entry has the person's name (key), job description (`"desc"`), and phone extension (`"fn"` — how to actually reach them). When you need something done, you look up who handles it, read their job description to confirm, then call their extension.

**`"fn": get_weather`**: We're storing a reference to the function itself — NOT calling it. `get_weather` (without parentheses) is the function object. `get_weather()` (with parentheses) would call it. This is an important Python distinction.

---

## 🤖 Section 4: The Agent Loop (Lines 76–140)

### The System Prompt

```python
SYSTEM = """You are a helpful assistant with tools.

{tools}

To call a tool respond ONLY with JSON (no other text):
{{"tool": "name", "arguments": {{"param": "value"}}}}

To give a final answer respond ONLY with JSON:
{{"response": "your answer"}}
"""
```

**`{tools}`**: A placeholder. We'll fill this in with the actual tool descriptions using `.format()` before sending to the API.

**`{{"tool": "name"}}`**: In Python f-strings and `.format()` strings, a literal `{` is written as `{{` (and `}` as `}}`). Otherwise Python would try to interpret `{tool}` as a variable placeholder.

**Why the SYSTEM prompt is so rigid**: We instruct the LLM to respond with ONLY JSON. This is critical — if the LLM adds any prose ("Sure! I'll help you with that. Here's the JSON: {...}"), our `json.loads()` would fail. We're essentially programming the LLM's output format.

**Analogy**: The SYSTEM prompt is like a form the AI must fill out. The form has a strict format — "Fill out ONLY this form, no extra writing." If the AI adds commentary outside the form, our parser can't process it.

---

```python
def tool_descriptions():
    lines = []
    for name, t in TOOLS.items():
        lines.append(f"  {name}: {t['desc']} params={t['params']}")
    return "\n".join(lines)
```

**Purpose**: Creates a human-readable (and LLM-readable) description of all available tools. This gets injected into `SYSTEM` via `.format(tools=tool_descriptions())`.

**Output looks like**:
```
  get_weather: Get current weather for a city. params={'city': 'city name'}
  calculate: Evaluate a math expression. params={'expression': "e.g. '15 * 8 + 100'"}
  ...
```

---

### The JSON Parser

```python
def parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {"response": text}
```

**`re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)`**: Removes thinking blocks.

**Why**: DeepSeek-R1 is a reasoning model that outputs its internal thinking in `<think>...</think>` tags before the actual response. We need to strip these out before parsing the JSON.

**`r"<think>.*?</think>"`** explained:
- `<think>` — matches literally
- `.*?` — matches any character (`.`) zero or more times (`*`) non-greedily (`?`)
- `</think>` — matches literally
- Non-greedy (`?`) means "match as LITTLE as possible" — stops at the FIRST `</think>`, not the last

**`flags=re.DOTALL`**: By default, `.` in regex doesn't match newlines. `re.DOTALL` makes `.` match everything including newlines. Thinking blocks can span multiple lines.

**Analogy**: The `<think>...</think>` stripping is like removing Post-It notes from a report before presenting it. The thinking is the author's scratchpad (Post-Its). The JSON is the final report. We remove the scratchpad before processing.

---

**`re.search(r"\{.*\}", text, re.DOTALL)`**: Searches for JSON-like content anywhere in the text.
- `\{` — matches a literal `{`
- `.*` — matches anything (greedily)
- `\}` — matches a literal `}`
- Returns a **match object** if found, or `None` if not

**`m.group()`**: If a match was found, `.group()` returns the matched string (the JSON text we found).

**`json.loads(m.group())`**: `json.loads()` (load string) — parses a JSON string into a Python dictionary.

**Fallback**: `return {"response": text}` — if we can't find or parse JSON, treat the entire response as a plain text answer.

**Analogy**: `parse()` is like a secretary who opens an envelope from the AI. She first removes any sticky notes (thinking blocks). Then she looks for the official form (`\{.*\}` — the JSON). If she finds it, she files it properly (`json.loads`). If not, she puts the whole letter in the "general response" folder.

---

### The Main Agent Loop

```python
def run_agent(query: str, max_iter: int = 6):
    steps = []
    msgs  = [
        {"role": "system", "content": SYSTEM.format(tools=tool_descriptions())},
        {"role": "user",   "content": query},
    ]
```

**`SYSTEM.format(tools=tool_descriptions())`**: Fills the `{tools}` placeholder in our SYSTEM string with the actual tool descriptions. Python's `str.format()` replaces `{placeholder}` with the given values.

**`steps = []`**: Keeps track of everything the agent did — which tools it called, what they returned. This is displayed in the UI's "Agent steps" expander.

---

```python
    for _ in range(max_iter):
```

**`for _ in range(max_iter):`**: Loops up to `max_iter` (6) times. The `_` is a convention for "I don't care about the loop variable" — we only need the iteration count, not the actual index.

**Why limit iterations?**: Without a limit, a buggy or confused LLM could loop forever — calling tools endlessly without ever giving a final answer. `max_iter=6` ensures the agent terminates even in pathological cases.

**Analogy**: `max_iter` is like a restaurant saying "a table can stay for a maximum of 2 hours." Without this rule, a table could be occupied forever. With it, there's always an endpoint.

---

```python
        raw = ""
        for chunk in client.chat_completion(messages=msgs, model="deepseek-ai/DeepSeek-R1",
                                            max_tokens=300, temperature=0.1, stream=True):
            if hasattr(chunk, "choices") and chunk.choices:
                c = chunk.choices[0].delta.content
                if c:
                    raw += c
        decision = parse(raw)
```

**`temperature=0.1`**: Very low temperature for structured output (JSON). We want the LLM to reliably output valid JSON — high temperature might cause it to output garbled JSON or add prose.

**`raw`**: Accumulates the complete LLM response (after all streaming chunks are assembled).

**`decision = parse(raw)`**: Parses the raw text into a Python dictionary. Either `{"tool": ..., "arguments": ...}` or `{"response": ...}`.

---

```python
        if "tool" in decision:
            name = decision["tool"]
            args = decision.get("arguments", {})
            fn   = TOOLS.get(name, {}).get("fn")
            result = fn(**args) if fn else f"Unknown tool: {name}"
```

**`decision.get("arguments", {})`**: If the LLM's JSON has no `"arguments"` key (some tools have no parameters), use `{}` as default.

**`TOOLS.get(name, {}).get("fn")`**: Two chained `.get()` calls:
1. `TOOLS.get(name, {})` — look up the tool by name; if not found, return `{}` (empty dict) instead of crashing
2. `.get("fn")` — get the function from the tool dict; returns `None` if the tool wasn't found

**`fn(**args) if fn else f"Unknown tool: {name}"`**: 
- If `fn` is a function (not `None`): call it with `**args`
- If `fn` is `None` (unknown tool): return an error string

**`**args`**: Double asterisk unpacks a dictionary as keyword arguments. If `args = {"city": "Mumbai"}`, then `fn(**args)` is the same as `fn(city="Mumbai")`. This is how we call any tool function with the arguments the LLM provided without knowing in advance how many arguments there are.

**Analogy**: `**args` is like handing someone an envelope labeled with their name and saying "open it when you need your instructions." Inside is the full set of instructions customized for that function. Each function gets exactly the arguments it needs.

---

```python
            msgs.append({"role": "assistant", "content": raw})
            msgs.append({
                "role": "user",
                "content": f"Tool result for {name}: {result}\n\nNow give your final answer."
            })
```

**After a tool call, we add two messages**:
1. The LLM's tool call request (as "assistant" role) — so the LLM remembers what it asked for
2. The tool result (as "user" role) — injected as if the user is giving the LLM the information

**Why add the result as a "user" message?**: The API conversation format only has "user" and "assistant" roles (plus "system"). Tool results don't have a native role, so we inject them as "user" messages.

**Analogy**: Imagine a manager (LLM) writing a memo to their assistant (our code): "Please check the weather in Mumbai" (first appended message). The assistant goes and checks, then comes back: "Mumbai: 32°C, partly cloudy" (second appended message). Now the manager can write the final report based on this information.

---

```python
        elif "response" in decision:
            steps.append({"type": "final", "text": decision["response"]})
            return decision["response"], steps
```

**When the LLM gives a direct response** (no tool needed): extract the text and return it immediately. The loop exits early — we don't need more iterations.

---

```python
    return "Could not complete within step limit.", steps
```

**Fallback after the loop**: If we used all `max_iter` iterations without getting a `"response"` decision, return a polite failure message. This handles the case where the LLM keeps trying to call tools without ever giving a final answer.

---

## 🖥️ Section 5: Streamlit UI (Lines 143–189)

```python
with st.sidebar:
    for name, t in TOOLS.items():
        with st.expander(f"`{name}`"):
            st.markdown(t["desc"])
            if t["params"]:
                for p, d in t["params"].items():
                    st.markdown(f"- `{p}`: {d}")
```

**`f"\`{name}\`"`**: Backticks in Markdown render text in code font (monospace). So `` `get_weather` `` renders as `get_weather`.

**Nested `for` loop**: Outer loop over tools, inner loop over each tool's parameters. For each parameter, we display `- parameter_name: description` as a markdown list item.

---

```python
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state.prefill = ex
```

**Clickable example buttons**: When clicked, stores the example text in `session_state.prefill`. The main area reads this and pre-fills the chat input.

**`key=ex`**: Since all buttons are created in a loop, each needs a unique key. Using the example text itself (`ex`) as the key is safe because all example texts are unique strings.

---

```python
prefill = st.session_state.pop("prefill", "")
query   = st.chat_input("Ask anything…")
actual  = query or prefill
```

**`st.session_state.pop("prefill", "")`**: Like `dict.pop()` — removes the key from session state and returns its value. If `"prefill"` doesn't exist, returns `""`. Critically, this also **deletes** the prefill value so it's only used once (not re-applied on every re-run).

**`query or prefill`**: In Python, `a or b` returns `a` if `a` is truthy, otherwise returns `b`. So `actual` is the typed query if one exists, otherwise the prefilled text. If both are empty strings, `actual` is `""` (falsy), and the `if actual:` block won't run.

---

```python
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
```

**`any(s["type"] == "tool" for s in steps)`**: A generator expression inside `any()`. `any(condition for item in iterable)` returns `True` if the condition is true for at least ONE item. Here: "are there any steps that are tool calls?" Only show the expander if the agent actually called tools.

**`st.code(f"...")`**: Renders text in a monospace code block with syntax highlighting support. Used here to display tool call details clearly.

---

## 🔑 New Concepts in This File

| Concept | What it is | Analogy |
|---------|-----------|---------|
| `requests.get(url)` | Make an HTTP GET request | A delivery driver fetching a URL |
| `r.status_code` | HTTP response code (200=OK) | Delivery confirmation number |
| `r.json()` | Parse JSON response body | Form reader that understands JSON |
| `timeout=5` | Don't wait more than 5 seconds | "If no answer in 5 rings, hang up" |
| `eval(expr, globals, locals)` | Run string as Python code, sandboxed | A calculator room with restricted access |
| `**args` (double unpack) | Unpack dict as keyword arguments | Personalized instruction envelope |
| `type hints` (`: str`, `-> str`) | Document expected types | Labels on file folders |
| `SYSTEM.format(tools=...)` | Fill placeholders in a string | Filling in a form template |
| `any(condition for x in iterable)` | True if condition is true for any item | "Is there at least one red ball in the bag?" |
| `dict.pop(key, default)` | Remove and return a value | Taking a note off a corkboard |
| `math.__dict__` | All contents of a module as a dict | The table of contents of a library |

Next: `evaluate_explained.md`
