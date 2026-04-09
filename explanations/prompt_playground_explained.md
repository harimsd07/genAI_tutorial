# 📖 Code Explanation: `prompt_playground.py`

> **Purpose of this file**: This document explains every single line of `projects/prompt_playground.py` — what it does, why it exists, and how to think about it using real-world analogies. Read this alongside the actual code file.

---

## 🧱 The Big Picture Before We Start

Before reading a single line, understand the whole application in one mental image:

```
Imagine you are a scientist testing two fertilizer formulas (Prompt A and Prompt B)
on two identical plants (same AI model) under the same conditions.
You change ONE thing at a time (the prompt, or the temperature),
observe both plants' growth (AI outputs) side by side,
and write notes in your lab book (session history).

That is EXACTLY what this app does — but for AI prompts.
```

The app has four sections:
1. **System Prompt** — the AI's "job description"
2. **Few-Shot Examples** — demonstration examples for the AI
3. **A/B Comparison** — two prompts running at the same time
4. **History** — a saved record of every experiment

---

## 📦 Section 1: The Imports (Lines 1–17)

```python
"""
Prompt Playground – Interactive prompt engineering tool.
Run with: streamlit run projects/prompt_playground.py
...
"""
```

**What this is**: A docstring — a string that acts as documentation for the entire file. Python doesn't execute this; it's purely for humans reading the code.

**Analogy**: This is like the label on the front of a filing folder — it tells you what's inside before you open it.

---

```python
import streamlit as st
```

**What `streamlit` is**: A Python library that lets you build interactive web applications using nothing but Python — no HTML, no JavaScript, no CSS needed. You describe what you want on the page, and Streamlit handles the rest.

**How it works**: Every time a user interacts with the app (clicks a button, moves a slider), Streamlit re-runs your entire Python script from top to bottom and refreshes the page with the new output.

**Analogy**: Streamlit is like a very smart waiter at a restaurant. You (the programmer) hand him a recipe card (your Python script). Every time a customer changes their order (clicks something), the waiter re-reads the recipe card from the beginning and brings the updated dish. You never talk to the kitchen directly — the waiter (Streamlit) handles all that.

**`as st`**: This creates a short alias. Instead of writing `streamlit.title(...)` everywhere, we write `st.title(...)`. It's just for convenience.

---

```python
from huggingface_hub import InferenceClient
```

**What `huggingface_hub` is**: The official Python SDK (Software Development Kit) for Hugging Face. A library that handles all the complexity of calling AI models over the internet — authentication, request formatting, response parsing, error handling.

**What `InferenceClient` is**: The specific class (blueprint for an object) inside `huggingface_hub` that we need. It represents "a connection to the Hugging Face inference API." Think of it as a phone — once you set it up with your account details, you can call any AI model.

**Analogy**: `huggingface_hub` is like the postal service company (FedEx). `InferenceClient` is your specific FedEx account — you set it up once with your credentials, and then you can send packages (prompts) to any address (AI model) and receive deliveries (responses).

**`from X import Y`**: Instead of importing the entire `huggingface_hub` package, we only import the specific piece we need (`InferenceClient`). This is like going to a hardware store and only buying the one screwdriver you need, rather than buying the entire toolkit.

---

```python
from dotenv import load_dotenv
```

**What `dotenv` is**: A library that reads a special file called `.env` (dot-env) and loads the contents into your program's environment variables. 

**Why we need it**: Your API key (your secret password for Hugging Face) should never be written directly in your code. If you share your code or push it to GitHub, everyone can see your key and steal it. Instead, you put the key in a `.env` file (which you never share), and `dotenv` reads it into your program at runtime.

**Analogy**: Imagine you have a locker at the gym. Your code is a printed note that says "Go to locker number 42 and get the gym membership card." The actual membership card (your API key) is in the locker (the `.env` file). The printed note (your code) can be shown to anyone — it doesn't contain the secret. `dotenv` is the person who walks to the locker and gets the card when your program starts.

---

```python
import os
```

**What `os` is**: Python's built-in module for interacting with the operating system. "OS" stands for Operating System. It gives you tools to read environment variables, create folders, check if files exist, etc.

**Why we need it here**: After `load_dotenv()` loads our API key into the environment, we use `os.getenv("HUGGINGFACEHUB_API_TOKEN")` to actually read it.

**Analogy**: `os` is like your operating system's secretary. It knows where everything is stored on your computer, and you can ask it to fetch things. `os.getenv()` is like saying "Secretary, please find the document labeled HUGGINGFACEHUB_API_TOKEN from the environment cabinet."

---

```python
import json
```

**What `json` is**: Python's built-in module for working with JSON (JavaScript Object Notation) — a text format for representing structured data (dictionaries, lists, numbers, strings).

**Why we need it**: When we save the prompt history to a file, we save it as JSON because JSON is a universal format that any program or person can read easily.

**Analogy**: JSON is like a standardized form that every organization agrees to use. If you fill out a customer form in JSON format, any company's computer system can read it — there's no special translator needed.

---

```python
from datetime import datetime
```

**What `datetime` is**: Python's built-in module for working with dates and times. `datetime` is the specific class inside it that represents a specific point in time (like "2024-01-15 at 3:45 PM").

**Why we need it**: When saving history, we want to timestamp each entry so you know when you ran each experiment.

**Analogy**: `datetime` is like the clock and calendar on your wall. When you take a photo and want to remember when it was taken, you look at the clock. Here, `datetime.now()` takes the current time reading and stamps it onto each history entry.

---

## 🔑 Section 2: Setup and Initialization (Lines 19–20)

```python
load_dotenv()
```

**What this line does**: Reads the `.env` file in your project folder and loads all the `KEY=VALUE` pairs inside it into your system's environment variables. After this line runs, Python can access your API token.

**Analogy**: This is like telling the gym secretary "Go to the locker room now and fetch my membership card." After this line, the card is in your pocket (environment).

**Important**: This must be called BEFORE you try to read any environment variables. If you call `os.getenv()` before `load_dotenv()`, the key won't be found yet.

---

```python
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
```

Let's break this into three parts:

**`os.getenv("HUGGINGFACEHUB_API_TOKEN")`**: Goes to the environment (loaded by `load_dotenv()`) and retrieves the value stored under the key `"HUGGINGFACEHUB_API_TOKEN"`. This returns a string like `"hf_abcdefg123456..."`.

**`InferenceClient(token=...)`**: Creates a new `InferenceClient` object, configured with your API token. This object now represents "a live authenticated connection to Hugging Face."

**`client = ...`**: Stores this connection in a variable called `client`. Every time we want to call an AI model later, we'll use `client.chat_completion(...)`.

**Analogy**: 
- `os.getenv("HUGGINGFACEHUB_API_TOKEN")` = Getting your membership card from the locker
- `InferenceClient(token=...)` = Walking into the gym and showing your card at the entrance
- `client = ...` = You are now "checked in" and can use the equipment (call AI models)

---

## 🖥️ Section 3: Page Configuration (Lines 22–24)

```python
st.set_page_config(page_title="Prompt Playground", page_icon="🎮", layout="wide")
```

**What this does**: Configures the browser tab appearance and page layout. This must be the FIRST Streamlit command in any script — calling it later causes an error.

**`page_title="Prompt Playground"`**: Sets the text shown in the browser tab (the small text you see at the top of your browser).

**`page_icon="🎮"`**: Sets the favicon — the small icon shown in the browser tab next to the title.

**`layout="wide"`**: Makes the app use the full width of the browser window. The alternative is `"centered"`, which puts content in a narrow column in the middle.

**Analogy**: This is like calling a restaurant to make a reservation and specifying: "We want the table by the window (wide layout), and please put 'Smith Party' on the nameplate (page_title), and a small rose on the table (page_icon)." You set the scene before guests arrive.

---

```python
st.title("🎮 Prompt Playground")
st.markdown("_Experiment with prompts. Compare outputs. Learn what works._")
```

**`st.title(...)`**: Renders a large heading (H1 in HTML terms) on the page.

**`st.markdown(...)`**: Renders text that supports Markdown formatting. The underscores `_text_` make the text *italic*. Markdown is a lightweight markup language where `**text**` = **bold**, `_text_` = *italic*, `# text` = heading, etc.

**Analogy**: These two lines are like putting a big sign above your shop ("🎮 Prompt Playground") and a small tagline below it ("_Experiment with prompts..._"). Every customer sees this when they walk in.

---

## ⚙️ Section 4: Sidebar Model Selector (Lines 26–45)

```python
st.sidebar.header("⚙️ Model")
```

**What the sidebar is**: A narrow panel on the left side of the app. In Streamlit, anything you put in `st.sidebar` appears there instead of the main area. It's used for controls and settings that aren't the main content.

**`st.sidebar.header(...)`**: Adds a heading inside the sidebar.

**Analogy**: The sidebar is like the control panel on a mixing board in a recording studio. The main area of the app is the stage where music plays. The sidebar is where you adjust settings (volume, EQ, reverb) without cluttering the stage.

---

```python
MODELS = {
    "DeepSeek-R1 (Reasoning)":    "deepseek-ai/DeepSeek-R1",
    "Llama 3.1 8B (Instruction)":  "meta-llama/Llama-3.1-8B-Instruct",
    "Zephyr 7B (General Chat)":    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen 2.5 7B (Multilingual)":  "Qwen/Qwen2.5-7B-Instruct",
}
```

**What this is**: A Python **dictionary** — a data structure that stores key-value pairs. The keys are the human-readable names you'll see in the dropdown (like "DeepSeek-R1 (Reasoning)"), and the values are the actual model IDs that Hugging Face uses internally.

**Why separate human names from model IDs?**: Model IDs like `"deepseek-ai/DeepSeek-R1"` are ugly and confusing. We show the user a nice name, but when we call the API, we use the real ID. This also means we can change the model ID in one place (here) if it changes, without touching any other code.

**`ALL_CAPS` variable name**: By convention in Python, `MODELS` being all-caps signals to other programmers "this is a constant — don't change this value during the program." Python doesn't enforce this, but it's a widely respected signal.

**Analogy**: A restaurant menu lists "Chicken Tikka Masala" (the human-readable key), but the kitchen order slip says "KTM-003" (the internal ID). This dictionary is that translation table.

---

```python
model_name = st.sidebar.selectbox("Model", list(MODELS.keys()))
```

**`list(MODELS.keys())`**: `.keys()` returns all the keys of the dictionary — the human-readable names. `list(...)` converts them into a regular Python list, which Streamlit's selectbox requires.

Result: `["DeepSeek-R1 (Reasoning)", "Llama 3.1 8B (Instruction)", "Zephyr 7B (General Chat)", "Qwen 2.5 7B (Multilingual)"]`

**`st.sidebar.selectbox("Model", [...])`**: Creates a dropdown menu in the sidebar with the label "Model" and the given list of options. Returns the currently selected option as a string.

**`model_name = ...`**: Stores the user's selection (e.g., `"DeepSeek-R1 (Reasoning)"`) in a variable.

**Analogy**: `st.sidebar.selectbox` is the dropdown on a flight booking site where you choose your destination. `model_name` is whatever city you clicked on.

---

```python
model_id = MODELS[model_name]
```

**What this does**: Uses `model_name` (the human-readable label the user selected) as a key to look up the corresponding model ID in the `MODELS` dictionary.

If `model_name = "DeepSeek-R1 (Reasoning)"`, then `MODELS[model_name]` returns `"deepseek-ai/DeepSeek-R1"`.

**Analogy**: You selected "Chicken Tikka Masala" from the menu (model_name). The kitchen system looks it up in the translation table and gets "KTM-003" (model_id). The API call uses "KTM-003" — the model never hears the human-readable name.

---

## 📝 Section 5: System Prompt Input (Lines 47–54)

```python
st.subheader("1️⃣ System Prompt")
system = st.text_area(
    "System Prompt",
    "You are a helpful assistant that answers clearly and concisely.",
    height=70,
    label_visibility="collapsed",
)
```

**`st.subheader(...)`**: Renders a medium-sized heading (H2). Smaller than `st.title`, larger than normal text. Used to label sections of the page.

**`st.text_area(...)`**: Creates a multi-line text input box. Arguments:
- `"System Prompt"` — the internal label (hidden here due to `label_visibility="collapsed"`)
- The second positional argument (`"You are a helpful assistant..."`) — the default value that appears in the box when the app first loads
- `height=70` — how tall the box is in pixels
- `label_visibility="collapsed"` — hides the label text above the box (since we already have the subheader above it)

**Return value**: Whatever text is currently in the box. `system` holds this text.

**Analogy**: `st.text_area` is like a physical sticky note. It comes pre-filled with a suggestion ("You are a helpful assistant..."), but the user can erase and rewrite it. Whatever is written on the sticky note at any given moment is stored in `system`.

---

## 🎯 Section 6: Few-Shot Examples (Lines 56–59)

```python
with st.expander("Add examples — format: 'User: ...' / 'Assistant: ...'"):
    few_shot = st.text_area("Examples", "", height=100, label_visibility="collapsed")
```

**`st.expander(...)`**: Creates a collapsible section — it shows a clickable header, and when you click it, a box expands to reveal content inside. The string argument is the clickable header text.

**`with st.expander(...):` (the `with` keyword)**: In Python, `with` is used for context managers — things that have a setup and teardown. Here it means: "Everything indented inside this `with` block will be placed inside the expander widget."

**`few_shot = st.text_area("Examples", "", height=100, ...)`**: Creates a text area inside the expander. Default value is `""` (empty string) — the box starts blank.

**Analogy**: `st.expander` is like a drawer in a desk. Most of the time it's closed (keeping the interface clean). When you need to add examples, you open the drawer. Everything inside the drawer only becomes visible when the drawer is open.

---

## ↔️ Section 7: A/B Comparison Setup (Lines 61–71)

```python
st.subheader("3️⃣ A/B Comparison")
c1, c2 = st.columns(2)
```

**`st.columns(2)`**: Divides the page into 2 equal columns side by side. Returns a tuple of column objects — one per column. We immediately unpack them into `c1` and `c2`.

**Analogy**: `st.columns(2)` is like splitting a sheet of paper down the middle with a vertical line. Whatever you draw on the left half goes in `c1`, and whatever you draw on the right half goes in `c2`.

**`c1, c2 = st.columns(2)`**: Python tuple unpacking — `st.columns(2)` returns `(column_object_1, column_object_2)`, and we assign the first to `c1` and the second to `c2` in one line. It's equivalent to:
```python
cols = st.columns(2)
c1 = cols[0]
c2 = cols[1]
```

---

```python
with c1:
    prompt_a  = st.text_area("Prompt A", "Explain machine learning.", height=90, key="pa")
    temp_a    = st.slider("Temp A",  0.0, 1.0, 0.7, key="ta")
    tokens_a  = st.slider("Max tokens A", 50, 1000, 300, key="ma")
```

**`with c1:`**: Everything indented here appears in the left column.

**`st.slider("Temp A", 0.0, 1.0, 0.7, key="ta")`**: Creates a draggable slider.
- `"Temp A"` — label shown above the slider
- `0.0` — minimum value
- `1.0` — maximum value  
- `0.7` — **default value** (where the slider starts)
- `key="ta"` — a unique identifier for this widget (see below)

**Why `key=`?**: When Streamlit has multiple widgets of the same type (like two sliders both labeled "Temperature"), it gets confused about which is which unless you give each a unique `key`. Every widget that appears more than once needs a unique key.

**Analogy**: The `key` is like giving each widget an employee ID number. Even if two employees have the same job title ("slider"), they have different IDs ("ta" vs "tb"), so the system can tell them apart.

---

```python
with c2:
    prompt_b  = st.text_area("Prompt B", "Explain machine learning using a simple everyday analogy, then give the technical definition.", height=90, key="pb")
    temp_b    = st.slider("Temp B",  0.0, 1.0, 0.3, key="tb")
    tokens_b  = st.slider("Max tokens B", 50, 1000, 300, key="mb")
```

Identical structure to column 1, but with different default values:
- `temp_b` starts at `0.3` (more conservative) vs `temp_a` at `0.7` (more creative)
- This pre-sets a meaningful comparison: same question, different temperatures

---

## 🏗️ Section 8: The `build_messages` Function (Lines 74–85)

```python
def build_messages(system_txt, few_shot_txt, user_txt):
    msgs = []
```

**`def build_messages(...):`**: Defines a function called `build_messages` that takes three arguments.

**`msgs = []`**: Creates an empty list. We'll build the messages list inside this function and return it at the end.

**What this function's purpose is**: The Hugging Face API expects messages in a very specific format — a list of dictionaries, each with a `"role"` and `"content"` key. This function assembles that list from the three pieces of input.

**Analogy**: Think of building a sandwich. `system_txt` is the bread (goes first, sets the foundation). `few_shot_txt` is the lettuce and tomatoes (optional filling in the middle). `user_txt` is the final layer. This function assembles them in the right order.

---

```python
    if system_txt.strip():
        msgs.append({"role": "system", "content": system_txt})
```

**`system_txt.strip()`**: Removes leading and trailing whitespace from the string. `.strip()` on `"   hello   "` returns `"hello"`.

**`if system_txt.strip():`**: If the stripped text is non-empty (i.e., the user actually typed something), add it to messages. An empty string `""` is "falsy" in Python — the if block won't run if the user left the system prompt blank.

**`msgs.append({...})`**: Adds a new dictionary to the `msgs` list. The dictionary has two keys:
- `"role": "system"` — tells the AI this is a system instruction, not a user message
- `"content": system_txt` — the actual text

**Analogy**: When you hire a new employee (the AI), you give them an employee handbook (system message) before their first customer interaction. This line says: "If there's a handbook, put it at the start of the conversation."

---

```python
    if few_shot_txt.strip():
        for line in few_shot_txt.strip().splitlines():
            if line.startswith("User:"):
                msgs.append({"role": "user", "content": line[5:].strip()})
            elif line.startswith("Assistant:"):
                msgs.append({"role": "assistant", "content": line[10:].strip()})
```

**`few_shot_txt.strip().splitlines()`**: Splits the multi-line text into a list of individual lines.

For example, if the text is:
```
User: What is 2+2?
Assistant: 4
User: What is 5+3?
Assistant: 8
```
Then `splitlines()` gives:
```python
["User: What is 2+2?", "Assistant: 4", "User: What is 5+3?", "Assistant: 8"]
```

**`for line in ...:`**: Iterates over each line one at a time.

**`if line.startswith("User:")`**: Checks if the line begins with "User:". `startswith()` is a string method that returns `True` or `False`.

**`line[5:].strip()`**: `line[5:]` slices the string starting from index 5 (skipping the first 5 characters: `"User:"`). `.strip()` removes any leading space. So `"User: What is 2+2?"` becomes `"What is 2+2?"`.

**`line[10:].strip()`**: Same idea for "Assistant:" which is 10 characters long.

**Analogy**: Imagine parsing a transcript of a conversation. Each line is labeled with who said it. This function reads each labeled line and converts it into the proper API format — like a human transcriptionist converting audio timestamps into a structured spreadsheet.

---

```python
    msgs.append({"role": "user", "content": user_txt})
    return msgs
```

**Final user message**: This always gets added last — it's the actual question the user is asking right now.

**`return msgs`**: The function hands back the completed list of message dictionaries to whoever called it.

---

## 📡 Section 9: The `call_model` Function (Lines 88–95)

```python
def call_model(msgs, temp, max_tok):
    full = ""
    for chunk in client.chat_completion(messages=msgs, model=model_id, temperature=temp, max_tokens=max_tok, stream=True):
        if hasattr(chunk, "choices") and chunk.choices:
            c = chunk.choices[0].delta.content
            if c:
                full += c
    return full
```

**`full = ""`**: Start with an empty string. We'll build the complete response by adding pieces to this string.

**`client.chat_completion(...)`**: Makes the actual API call to Hugging Face. Arguments:
- `messages=msgs` — the conversation history we built with `build_messages()`
- `model=model_id` — which AI model to call (the ID we looked up earlier)
- `temperature=temp` — how random the output is (0.0–1.0)
- `max_tokens=max_tok` — maximum length of the response
- `stream=True` — instead of waiting for the full response, send it back in pieces as it's generated

**`stream=True` and what it means**: When you set `stream=True`, the API sends the response word-by-word (or token-by-token) as the model generates it, instead of waiting until the entire response is finished. This is how ChatGPT shows text appearing gradually.

**Analogy**: `stream=True` is like watching a live cricket match on TV vs watching a recorded replay. Without streaming, you'd wait until the match ends and then watch the whole thing at once. With streaming, you watch each ball as it happens in real time.

**`for chunk in client.chat_completion(..., stream=True):`**: When streaming, `chat_completion()` becomes a **generator** — an object you can iterate over. Each iteration gives you the next "chunk" (a small piece of the response).

---

**`if hasattr(chunk, "choices") and chunk.choices:`**

**`hasattr(chunk, "choices")`**: Python's `hasattr()` function checks if an object has a specific attribute. Not all streaming chunks contain choices — some are metadata chunks sent at the beginning or end of the stream. This check prevents errors when accessing `.choices` on those metadata chunks.

**`and chunk.choices`**: Even if the attribute exists, it might be an empty list. The second condition ensures it has content.

**`hasattr()` explained deeper**: When streaming, Hugging Face sends different types of objects. Most are content chunks (have `.choices`), but some are start/end signals (don't have `.choices`). Without this check, Python would throw an `AttributeError` on the metadata chunks.

**Analogy**: Imagine a conveyor belt delivering packages. Most packages contain actual products (content chunks), but occasionally empty boxes and separator cards come down the belt (metadata chunks). `hasattr()` is the quality inspector who checks each item before processing it — only actual packages go to the customer.

---

```python
            c = chunk.choices[0].delta.content
```

**`chunk.choices`**: A list of "choices" (the API supports generating multiple completions at once, though we always use index 0 since we only need one response).

**`chunk.choices[0]`**: Gets the first (and only) choice.

**`.delta`**: In streaming mode, each chunk contains a "delta" — the *new* content added in this chunk, not the entire response so far.

**`.content`**: The actual text string in this delta. Could be a single character, a word, or several words depending on how the model generates.

**Structure summary**: `chunk` → `.choices[0]` → `.delta` → `.content` → the actual text

**Analogy**: Imagine receiving a telegram that arrives letter by letter. `chunk` is the current delivery. `choices[0]` is the main message (there's only one). `delta` is the newest letter just arrived. `content` is the actual letter character. We build the full message by collecting all the letters.

---

```python
            if c:
                full += c
    return full
```

**`if c:`**: Checks that `c` (the content) is not `None` or an empty string. Some streaming chunks have `content=None` as a signal that streaming has ended. This prevents `None` being appended to our string (which would cause a TypeError).

**`full += c`**: Appends the new content to our growing response string. `+=` with strings means "add to the end." So if `full = "The weather"` and `c = " is sunny"`, then after `full += c`, `full = "The weather is sunny"`.

**`return full`**: Return the complete assembled response once all chunks are processed.

---

## 🚀 Section 10: The Generate Button and Outputs (Lines 98–123)

```python
if st.button("🚀 Generate Both", type="primary", use_container_width=True):
```

**`st.button(...)`**: Creates a clickable button. Returns `True` when clicked, `False` otherwise.

**`type="primary"`**: Makes the button visually prominent (filled/colored background). The alternative is `type="secondary"` (outlined button).

**`use_container_width=True`**: Makes the button stretch to fill the full width of its container.

**The `if` statement**: Everything indented inside this `if` block only runs when the button is clicked. Since Streamlit re-runs the script on every interaction, the button returns `True` only on the click that triggered the re-run.

**Analogy**: `if st.button(...):` is like a vending machine button. The button is always visible, but the code inside the `if` block (dispensing a snack) only executes when you actually press it.

---

```python
    if "history" not in st.session_state:
        st.session_state.history = []
```

**`st.session_state`**: This is one of Streamlit's most important concepts. Because Streamlit re-runs your entire script on every interaction, normal Python variables are reset to their initial values every run. `st.session_state` is a special dictionary that **persists between re-runs** — it keeps its values as long as the browser tab is open.

**`if "history" not in st.session_state:`**: Checks if the key `"history"` already exists in session state. On the very first run of the app, it won't exist yet, so we initialize it to an empty list.

**`st.session_state.history = []`**: Creates the history list in session state. From now on, `st.session_state.history` will keep all previous experiments even as the script re-runs.

**Analogy**: Normal Python variables are like writing on a whiteboard that gets erased after every class (every Streamlit re-run). `st.session_state` is a notebook that you take with you — the writing stays even between classes.

---

```python
    out_a = out_b = ""
```

**What this does**: Initializes both output variables to empty strings in one line. It's equivalent to:
```python
out_a = ""
out_b = ""
```

**Why initialize them?**: Later, when we save to history, we reference `out_a` and `out_b`. If an error occurred before they were assigned, Python would throw a `NameError`. Initializing them to `""` first means they always exist, even if the API call fails.

---

```python
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Output A:**")
        try:
            with st.spinner("Generating A..."):
                out_a = call_model(build_messages(system, few_shot, prompt_a), temp_a, tokens_a)
            st.markdown(out_a)
        except Exception as e:
            st.error(str(e))
```

**`r1, r2 = st.columns(2)`**: Creates two columns for displaying the outputs side by side (output A on left, output B on right).

**`with st.spinner("Generating A..."):`**: Shows a loading animation with a message while the code inside runs. Without this, the user would see a frozen screen during the API call (which can take 5-30 seconds) and might think the app has crashed.

**`try: ... except Exception as e:`**: Error handling. The `try` block contains code that might fail (API calls can fail for many reasons: rate limits, model unavailable, network issues). If anything inside `try` throws an error, execution immediately jumps to `except`, which catches the error and runs the `except` block instead of crashing the entire app.

**`Exception as e`**: `Exception` is Python's base class for all errors. `as e` gives the caught exception a name (`e`) so we can use it.

**`str(e)`**: Converts the exception object to a human-readable error message string.

**`st.error(str(e))`**: Displays the error message in a red error box in the UI.

**Analogy**: `try/except` is like a safety net under a trapeze artist. The artist (your code) attempts the difficult move (API call). If they fall (an error occurs), the net (except block) catches them and the show continues. Without the net, one mistake would end the entire performance (crash the app).

**`build_messages(system, few_shot, prompt_a)` as an argument**: We're calling `build_messages()` directly as an argument to `call_model()`. Python evaluates inner function calls first. So this is equivalent to:
```python
messages = build_messages(system, few_shot, prompt_a)
out_a = call_model(messages, temp_a, tokens_a)
```
We just combined it into one line for brevity.

---

```python
    st.session_state.history.append({
        "ts": datetime.now().strftime("%H:%M:%S"), "model": model_name,
        "system": system, "pa": prompt_a, "oa": out_a, "ta": temp_a,
        "pb": prompt_b, "ob": out_b, "tb": temp_b,
    })
```

**`st.session_state.history.append({...})`**: Adds a new dictionary to the history list. Each dictionary represents one complete experiment run.

**`datetime.now()`**: Gets the current date and time as a `datetime` object.

**`.strftime("%H:%M:%S")`**: `strftime` = "string format time." Converts the datetime object to a string using a format code:
- `%H` = hours (24-hour, e.g., 14)
- `%M` = minutes (e.g., 35)
- `%S` = seconds (e.g., 07)
- Result: "14:35:07"

**The dictionary keys**: `"ts"` (timestamp), `"model"`, `"system"`, `"pa"` (prompt A), `"oa"` (output A), `"ta"` (temperature A), etc. Short names because this is internal storage.

---

## 💬 Section 11: Single Streaming Prompt (Lines 125–147)

```python
if st.button("Generate", key="gen_s"):
    box = st.empty()
    full = ""
```

**`st.empty()`**: Creates an empty placeholder on the page. A placeholder is a reserved space that you can fill and re-fill with content. This is the key to implementing the "streaming text appearing" effect.

**Analogy**: `st.empty()` is like reserving a blank billboard. Right now it shows nothing. But you can update what's displayed on it at any time without changing the billboards around it.

---

```python
    try:
        for chunk in client.chat_completion(..., stream=True):
            if hasattr(chunk, "choices") and chunk.choices:
                c = chunk.choices[0].delta.content
                if c:
                    full += c
                    box.markdown(full + "▌")
        box.markdown(full)
```

**`box.markdown(full + "▌")`**: On every chunk, we update the placeholder with the accumulated text so far, plus a `▌` cursor character at the end. This creates the visual effect of text being typed in real time.

**`box.markdown(full)` after the loop**: Once streaming is complete, we update the placeholder one final time with just `full` (no cursor) to remove the blinking cursor character.

**Analogy**: The `box` is like a digital scoreboard at a cricket match. Every time a run is scored (new chunk arrives), you update the scoreboard with the new total. The `▌` cursor is like the "live" indicator light on the scoreboard, showing the match is still in progress. When the match ends, the light turns off.

---

## 📜 Section 12: Session History Display (Lines 149–165)

```python
if st.session_state.get("history"):
```

**`st.session_state.get("history")`**: `.get()` on a dictionary returns `None` if the key doesn't exist (instead of throwing a `KeyError`). `None` is falsy, so this condition is `False` when history is empty or doesn't exist yet — the history section won't be shown until at least one experiment has been run.

**Why not `if st.session_state.history:`?**: This would throw a `KeyError` if "history" hasn't been created yet (before the first button click).

---

```python
    if st.button("💾 Save History"):
        with open("prompt_history.json", "w") as f:
            json.dump(st.session_state.history, f, indent=2)
        st.success("Saved to prompt_history.json")
```

**`open("prompt_history.json", "w")`**: Opens (or creates) a file named `prompt_history.json` in **write mode** (`"w"`). If the file already exists, write mode overwrites it.

**`as f`**: Gives the opened file object the name `f`. This is Python's context manager pattern — when the `with` block ends, the file is automatically closed (even if an error occurs), preventing data corruption.

**`json.dump(st.session_state.history, f, indent=2)`**: 
- `json.dump()` converts a Python object to JSON text and writes it to a file
- First argument: the data to serialize (`st.session_state.history`)
- Second argument: the file to write to (`f`)
- `indent=2`: makes the JSON human-readable with 2-space indentation instead of one long line

**Analogy**: `json.dump()` is like a translator who takes your Python dictionary (written in "Python language") and writes it out in "JSON language" onto a paper file. The `indent=2` option is like asking the translator to use nice formatting with paragraphs instead of cramming everything onto one line.

---

```python
    for entry in reversed(st.session_state.history):
        with st.expander(f"[{entry['ts']}] {entry['pa'][:60]}..."):
```

**`reversed(...)`**: Returns an iterator that goes through the list in reverse order (newest first). We want the most recent experiment at the top.

**`f"[{entry['ts']}] {entry['pa'][:60]}..."`**: An f-string (formatted string literal). The curly braces `{}` are placeholders where Python inserts values:
- `entry['ts']` — the timestamp stored in this entry
- `entry['pa'][:60]` — the first 60 characters of Prompt A (slicing with `[:60]` to prevent huge expander headers)
- `"..."` — a literal ellipsis to indicate the text was cut off

Result looks like: `"[14:35:07] Explain machine learning using a simple everyday..."`

---

This completes the full line-by-line explanation of `prompt_playground.py`.

## 🔑 Key Concepts Recap

| Concept | What it is | Analogy |
|---------|-----------|---------|
| `import X` | Load a library | Picking up a toolbox |
| `st.*` | Streamlit UI element | Adding a widget to a web page |
| `st.session_state` | Persistent storage between re-runs | A notebook you carry between classes |
| `st.columns(N)` | Side-by-side layout | Splitting a page into N columns |
| `stream=True` | Get response token-by-token | Live cricket match vs recorded replay |
| `try/except` | Error handling | Safety net for acrobats |
| `@st.cache_resource` | Load expensive things once | Hiring a specialist who stays on call |
| `hasattr(obj, attr)` | Check if attribute exists | Checking if a package has contents before opening |
| `json.dump()` | Save Python data to a JSON file | Translating and printing a document |

Next explanation: `rag_pdf_chat_explained.md`
