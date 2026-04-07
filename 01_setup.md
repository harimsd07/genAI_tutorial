# 01 – Environment Setup

> **Goal**: Understand what Hugging Face is, get your free API key, install all tools, and make your first successful AI API call. By the end of this file you'll have a working development environment and understand *why* each piece exists.

---

## 🧠 Background: What is Hugging Face?

Before we touch any code, understand what you're signing up for.

**Hugging Face** is the GitHub of AI models. It hosts over 500,000 open-source AI models, datasets, and demo apps. Think of it as a giant library where researchers upload their trained models for anyone to use.

```
Traditional software:       AI with Hugging Face:
──────────────────          ───────────────────────
You write all the code  →   You use a pre-trained model (months of training = free)
You define all rules    →   Model learned patterns from billions of examples
You update manually     →   New models released constantly
```

**Why use Hugging Face instead of OpenAI/Anthropic?**

| Feature | Hugging Face (Free tier) | OpenAI GPT-4 |
|---------|--------------------------|--------------|
| Cost | Free (rate-limited) | Pay per token |
| Models available | 500,000+ | A few (GPT-3.5, GPT-4...) |
| Open source | Yes | No |
| Privacy | Models can run locally | Data sent to OpenAI |
| Learning value | High (see the internals) | Lower (black box) |

The free tier has rate limits (you can only call the API ~100 times/day), but it's completely sufficient for learning.

---

## Step 1: Create a Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** — use email or GitHub
3. Verify your email address

---

## Step 2: Get Your API Token

Your API token is like a password that proves you are who you say you are when calling the API.

1. Click your profile picture (top right) → **Settings**
2. Left sidebar → **Access Tokens**
3. Click **New token**
4. Name it `learning` → Type: **read** (read-only is enough for inference)
5. Copy the token (starts with `hf_`)
6. **Never share this token** — treat it like a password

> ⚠️ **Security rule**: Never write your actual token directly in code. We'll use a `.env` file instead.

---

## Step 3: Understand What "Provider" Means

When you call the Hugging Face API, your request doesn't always go directly to Hugging Face's servers. It goes to **inference providers** — third-party companies that host and run models:

```
Your code ──► Hugging Face API router ──► Provider (e.g., featherless-ai, nebius)
                                               │
                                               ▼
                                         Actual GPU running the model
                                               │
                                               ▼
                                         Response returned to you
```

**Why this matters**: Different providers support different models and different task types. If a model isn't available on any provider you have enabled, you'll get a "not supported" error. This is normal — just try a different model.

---

## Step 4: Install Python

**Windows**: Download from [python.org](https://python.org) → check "Add to PATH" during install.

**Mac**: 
```bash
brew install python   # if you have Homebrew
# or download from python.org
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update && sudo apt install python3 python3-pip -y
```

Verify installation (should be 3.8 or higher):
```bash
python --version      # Windows
python3 --version     # Mac/Linux
```

---

## Step 5: Create Project Folder and Virtual Environment

A **virtual environment** keeps your project's Python packages isolated from your system Python. This prevents version conflicts across projects.

```bash
# Create folder
mkdir genai_learning
cd genai_learning

# Create virtual environment
python -m venv venv         # Windows
python3 -m venv venv        # Mac/Linux

# Activate it
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# You should see (venv) at the start of your terminal prompt
```

> 💡 **Always activate your virtual environment** before working on this project. If you open a new terminal, re-run the activate command.

---

## Step 6: Install Required Libraries

```bash
pip install huggingface_hub streamlit chromadb sentence-transformers \
            pymupdf requests python-dotenv transformers datasets accelerate \
            Pillow torch
```

This will take 3–5 minutes. Here's what each package does:

| Package | Purpose |
|---------|---------|
| `huggingface_hub` | Main SDK for calling Hugging Face APIs |
| `streamlit` | Build web apps with pure Python (no HTML/JS needed) |
| `chromadb` | Local vector database for storing embeddings (used in RAG) |
| `sentence-transformers` | Local embedding model (converts text → numbers) |
| `pymupdf` | Read and extract text from PDF files |
| `requests` | Make HTTP API calls (used for weather API) |
| `python-dotenv` | Load secrets from `.env` file |
| `transformers` | Hugging Face's core model library (used for fine-tuning) |
| `datasets` | Load and process datasets (used for fine-tuning) |
| `accelerate` | Speed up training on GPU/CPU (used for fine-tuning) |
| `Pillow` | Image processing (used for multi-modal) |
| `torch` | PyTorch deep learning framework (underlying engine) |

---

## Step 7: Store Your API Token Safely

**Why `.env` files?** Your API token should never be hardcoded in Python files. If you accidentally push your code to GitHub, a hardcoded token gets exposed publicly and bots scrape GitHub 24/7 looking for leaked API keys.

Create a file named `.env` inside `genai_learning/`:

**Mac/Linux:**
```bash
echo 'HUGGINGFACEHUB_API_TOKEN=hf_your_token_here' > .env
```

**Windows (PowerShell):**
```powershell
'HUGGINGFACEHUB_API_TOKEN=hf_your_token_here' | Out-File -FilePath .env -Encoding UTF8
```

**Or simply create the file manually** with a text editor:
```
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

Also create a `.gitignore` file to prevent accidentally committing your secrets:
```bash
echo '.env' > .gitignore
echo 'venv/' >> .gitignore
echo '__pycache__/' >> .gitignore
```

---

## Step 8: Test Your First API Call

Create a file `test.py` and paste this code:

```python
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load the .env file → puts HUGGINGFACEHUB_API_TOKEN into environment
load_dotenv()

# Create a client with your token
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

print("Testing connection to Hugging Face API...")

try:
    response = client.chat_completion(
        messages=[{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        model="deepseek-ai/DeepSeek-R1",
        max_tokens=100,
        stream=True
    )
    
    print("✅ Connection successful! Model response:")
    print("─" * 40)
    for chunk in response:
        if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
    print("\n" + "─" * 40)

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check your token in .env is correct")
    print("2. Make sure you're connected to the internet")
    print("3. The model might be temporarily unavailable — try again in a minute")
```

Run it:
```bash
python test.py
```

Expected output:
```
Testing connection to Hugging Face API...
✅ Connection successful! Model response:
────────────────────────────────────────
The capital of France is Paris.
────────────────────────────────────────
```

---

## Step 9: Understanding the API Response Structure

This is important — you'll see this pattern in every file:

```python
# The response object from chat_completion with stream=True
# is a generator that yields "chunks" as the model generates text.

for chunk in response:
    # A chunk looks like this:
    # ChatCompletionStreamOutput(
    #     choices=[
    #         Choice(
    #             delta=ChoiceDelta(
    #                 content="The",    ← the new text in this chunk
    #                 role="assistant"
    #             ),
    #             finish_reason=None,
    #             index=0
    #         )
    #     ],
    #     ...
    # )
    
    # Safety check: some chunks (the last one) have finish_reason set
    # and may have empty choices or None content. Always check before accessing.
    if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
        content = chunk.choices[0].delta.content
        if content:  # Only print if there's actual text
            print(content, end="", flush=True)
```

**Why streaming?** Without streaming, you'd wait until the entire response is generated before seeing anything — which could be 10-30 seconds. With `stream=True`, tokens appear as they're generated, just like ChatGPT.

---

## Step 10: Run the Full Diagnostic Script

Create `diagnostics.py` to test everything is working:

```python
"""
Diagnostic script — runs tests for all capabilities used in this tutorial.
Run this whenever something isn't working.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def check(name, fn):
    try:
        result = fn()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False

print("=" * 50)
print("GENAI LEARNING ENVIRONMENT DIAGNOSTICS")
print("=" * 50)

# 1. Python version
print("\n[1] Python")
v = sys.version_info
check("Python version", lambda: f"{v.major}.{v.minor}.{v.micro} {'✓ OK' if v.major==3 and v.minor>=8 else '⚠ Need 3.8+'}")

# 2. Environment variables
print("\n[2] Environment Variables")
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
check("HF token loaded", lambda: f"{'hf_...' + token[-4:] if token else 'NOT FOUND'}")

# 3. Package imports
print("\n[3] Packages")
packages = [
    ("huggingface_hub", "from huggingface_hub import InferenceClient"),
    ("streamlit", "import streamlit"),
    ("chromadb", "import chromadb"),
    ("sentence_transformers", "from sentence_transformers import SentenceTransformer"),
    ("pymupdf (fitz)", "import fitz"),
    ("python-dotenv", "from dotenv import load_dotenv"),
    ("torch", "import torch"),
]
for name, imp in packages:
    check(name, lambda i=imp: exec(i) or "imported OK")

# 4. API connection
print("\n[4] API Connection")
from huggingface_hub import InferenceClient
client = InferenceClient(token=token)
def test_api():
    r = client.chat_completion(
        messages=[{"role": "user", "content": "Say 'pong'"}],
        model="deepseek-ai/DeepSeek-R1",
        max_tokens=10, stream=False
    )
    return "connected, response received"
check("Hugging Face API", test_api)

# 5. Local embedding model
print("\n[5] Local Embedding Model")
def test_embeddings():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer('all-MiniLM-L6-v2')
    emb = m.encode("test")
    return f"embedding shape: {emb.shape}"
check("SentenceTransformer", test_embeddings)

print("\n" + "=" * 50)
print("If all ✅: You're ready to go!")
print("If any ❌: Follow the error message above.")
print("=" * 50)
```

---

## Common Setup Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Package not installed | Run `pip install <package_name>` |
| `Token is invalid` | Wrong token in `.env` | Double-check the token starts with `hf_` and has no spaces |
| `Model not supported` | Provider doesn't host this model | Use `deepseek-ai/DeepSeek-R1` instead |
| `Rate limit exceeded` | Too many calls in short time | Wait 60 seconds and try again |
| `404 Not Found` | Using `text_generation` with wrong model | Use `chat_completion` instead |
| `Connection error` | No internet | Check your network connection |

---

## ✅ You're Ready!

You now have:
- A Hugging Face account with an API token
- Python virtual environment with all packages
- A tested API connection
- Understanding of how providers, tokens, and streaming work

**Before continuing**, make sure you ran `test.py` and saw a successful response.

Next: [02a_prompt_eng.md](02a_prompt_eng.md) – learn how to talk to AI effectively.
