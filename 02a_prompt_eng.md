# 02a – Prompt Engineering (Theory)

> **Core idea**: The AI model's weights are frozen — you can't change them. But you *can* change what you feed it. Prompt engineering is the art of crafting that input to reliably get the output you want.

---

## 🧠 What Exactly Is a Prompt?

A **prompt** is everything the model receives as input before generating a response. This includes:

```
┌────────────────────────────────────────────────────────┐
│                      FULL PROMPT                        │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ SYSTEM MESSAGE (optional)                        │  │
│  │ "You are a helpful assistant that responds       │  │
│  │  in bullet points. Be concise."                  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ CONVERSATION HISTORY (optional)                  │  │
│  │ User: "What is Python?"                          │  │
│  │ Assistant: "Python is a programming language..." │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ CURRENT USER MESSAGE                             │  │
│  │ "Now tell me about JavaScript."                  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

**The model sees all of this as one long text string.** The "roles" (system, user, assistant) are special tokens that help the model understand who said what.

---

## 📐 The Anatomy of a Great Prompt

A high-quality prompt has up to 6 components. Not all are always needed, but knowing them lets you choose:

| Component | What it does | Example |
|-----------|-------------|---------|
| **Role** | Sets the AI's persona and expertise | `"You are a senior Python engineer with 10 years of experience"` |
| **Task** | Describes precisely what to do | `"Review this code for bugs and security issues"` |
| **Context** | Background information the model needs | `"This is production code running on Python 3.9"` |
| **Format** | How the output should be structured | `"Respond in a numbered list. Each item must have: issue, severity, fix"` |
| **Examples** | Demonstrate the pattern you want | `"Example: Issue: SQL injection. Severity: Critical. Fix: Use parameterized queries."` |
| **Constraints** | Limits and rules | `"Be concise. Maximum 3 sentences per issue. No preamble."` |

---

## 🎯 The 5 Core Techniques

### 1. Zero-Shot Prompting

You ask the model to do something with no examples. Works well for tasks the model was heavily trained on.

```python
# Zero-shot
prompt = "Classify the sentiment of this review as Positive, Negative, or Neutral: 'The pizza arrived cold and the service was rude.'"
# Output: Negative
```

**When to use**: Simple, well-defined tasks. Classification, translation, summarization.

**Weakness**: For unusual output formats or edge cases, the model may guess incorrectly.

---

### 2. Few-Shot Prompting

You give 2–5 examples of input → output pairs *inside the prompt itself*. The model learns the pattern from the examples and applies it to your new input.

```python
prompt = """
Classify sentiment as: Positive, Negative, or Neutral.

Review: "I love this product! Works perfectly."
Sentiment: Positive

Review: "It broke on the first day. Terrible quality."
Sentiment: Negative

Review: "The package arrived on time."
Sentiment: Neutral

Review: "The pizza arrived cold and the service was rude."
Sentiment:"""
# Output: Negative  ← follows the pattern precisely
```

**Why it works**: The model doesn't truly "learn" from examples — it pattern-matches from its training. When you show it examples, you're activating the most relevant patterns in its weights.

**When to use**: Unusual output formats, domain-specific classification, structured outputs, when zero-shot gives wrong format.

**Rule of thumb**: 3 examples is usually optimal. More than 5 rarely helps and wastes tokens.

---

### 3. Chain-of-Thought (CoT) Prompting

You instruct the model to reason step-by-step before giving a final answer. This dramatically improves performance on math, logic, and multi-step reasoning.

```python
# Without CoT — often wrong on complex math
prompt = "If a train leaves Station A at 9am going 60km/h, and another train leaves Station B (240km away) at 10am going 80km/h, at what time do they meet?"
# Model might guess: "11:30am"  ← could be wrong

# With CoT — reliable
prompt = """
If a train leaves Station A at 9am going 60km/h, and another train leaves 
Station B (240km away) at 10am going 80km/h, at what time do they meet?

Think through this step by step:
"""
# Model will show:
# Step 1: By 10am, Train A has traveled 60km. Remaining gap = 180km.
# Step 2: Combined closing speed = 60 + 80 = 140 km/h
# Step 3: Time to close 180km = 180/140 = 1.29 hours = ~1h17m
# Step 4: 10am + 1h17m = 11:17am
# Answer: 11:17am
```

**Why it works**: Forcing the model to write intermediate steps prevents it from jumping to an intuitive but wrong answer. The intermediate text also becomes "context" for the final answer.

**Variants**:
- `"Let's think step by step."` — simple trigger
- `"Think through this carefully before answering."` — gentle
- `"First, identify what you know. Then, what you need to find. Then solve."` — structured

---

### 4. Role Prompting (System Prompt Design)

You set a persona for the model in the system message. This primes the model to draw from specific knowledge and adopt a specific communication style.

```python
# Generic system prompt — mediocre results
system = "You are a helpful assistant."

# Role-specific system prompt — much better results
system = """You are Dr. Sarah Chen, a cardiologist with 15 years of clinical experience.
When explaining medical concepts:
- Use clear, non-jargon language first, then introduce medical terms in parentheses
- Always clarify when something requires professional in-person evaluation
- Structure explanations as: simple explanation → technical detail → practical implication"""
```

**What to include in a system prompt**:
1. Who/what the model is (role)
2. Tone and communication style
3. What to always do / never do
4. Output format expectations

---

### 5. Structured Output Prompting

You constrain the model to output in a specific format (JSON, XML, markdown table, etc.) so your code can reliably parse it.

```python
prompt = """
Extract the key information from this job posting and return ONLY valid JSON.
No explanation, no preamble, no markdown code fences. Just the JSON object.

Job Posting:
"We're hiring a Senior Python Engineer in Bangalore. 
Salary: 25–35 LPA. Requirements: 5+ years Python, FastAPI, PostgreSQL. 
Apply by December 31."

Return this exact structure:
{
  "title": "",
  "location": "",
  "salary_min": 0,
  "salary_max": 0,
  "currency": "",
  "requirements": [],
  "deadline": ""
}
"""
```

**Critical trick**: Say "Return ONLY valid JSON" and "No explanation, no preamble." Without this, the model often wraps the JSON in explanation text that breaks `json.loads()`.

---

## 🌡️ Temperature and Other Parameters — Explained Deeply

### Temperature

Temperature controls the **randomness of token selection**. Understanding this prevents countless frustrating hours.

When a model generates the next token, it calculates a probability for every token in its vocabulary. For example:

```
After "The capital of France is", the probabilities might be:
  "Paris"    → 97.3%
  "located"  → 1.2%
  "a"        → 0.8%
  ...
```

**Temperature = 0.0**: Always pick the highest probability token. Outputs are deterministic and repetitive.

**Temperature = 1.0**: Use probabilities exactly as calculated. Some randomness.

**Temperature > 1.0**: Flatten the distribution (make unlikely tokens more likely). Very random, often incoherent.

```
Temperature effect on distribution:
                                     
Low temp (0.1)    High temp (1.5)    
  ████             ████             
  ████             ████             
  ████     vs      ████             
  ████             ████             
  ████    ██       ████  ████████   
Paris  Other     Paris  Other words
(very dominant)  (more spread out)  
```

**Practical guide**:

| Use case | Temperature |
|----------|-------------|
| Factual Q&A, structured output, code | 0.0 – 0.2 |
| Summarization, extraction | 0.2 – 0.5 |
| General chatbot, explanations | 0.5 – 0.7 |
| Creative writing, brainstorming | 0.7 – 1.0 |
| Poetry, experimental | 1.0 – 1.3 |

### Top-P (Nucleus Sampling)

An alternative to temperature. Instead of scaling probabilities, you only sample from the top tokens whose probabilities sum to P.

- `top_p=0.9` → Only consider tokens whose cumulative probability reaches 90%
- Practically similar to temperature but with a hard cutoff for very unlikely tokens

**Rule of thumb**: Use temperature OR top_p, not both at the same time.

### Max Tokens

The maximum number of tokens the model can generate. One token ≈ 0.75 words in English.

```
100 tokens  ≈  75 words  (short paragraph)
500 tokens  ≈  375 words (one page)
2000 tokens ≈  1500 words (several paragraphs)
```

**Important for reasoning models** (like DeepSeek-R1): These models generate a hidden "thinking" block before the final answer. Set `max_tokens` to at least 1000 or you'll cut off the reasoning mid-way.

---

## 🚫 Common Prompt Engineering Mistakes

### Mistake 1: Vague instructions

```python
# ❌ Bad
"Write something about climate change"

# ✅ Good
"Write a 3-paragraph explanation of how rising ocean temperatures affect 
coral reef ecosystems. Audience: high school students. Include one specific 
example of a reef that has been affected. End with one positive conservation effort."
```

### Mistake 2: Contradictory constraints

```python
# ❌ Bad — contradicts itself
"Give me a comprehensive, detailed, thorough overview in 2 sentences."

# ✅ Good — clear tradeoff
"Give me a 2-sentence summary of the key points. Then separately list 5 details."
```

### Mistake 3: Assuming the model knows your context

```python
# ❌ Bad — "it" is ambiguous
"Make it better"

# ✅ Good — specific reference
"Rewrite the third paragraph of my email to sound more professional and direct."
```

### Mistake 4: Forgetting to prevent preamble

```python
# ❌ Bad — model will add "Sure! Here's the translation: ..."
"Translate to Spanish: Hello world"

# ✅ Good
"Translate to Spanish. Return only the translated text, nothing else: Hello world"
```

### Mistake 5: Not specifying format for structured tasks

```python
# ❌ Bad — will get prose
"List the pros and cons of remote work"

# ✅ Good — will get parseable output
"List the pros and cons of remote work as a JSON object:
{'pros': [...], 'cons': [...]}"
```

---

## 🔁 The Prompt Refinement Loop

Prompt engineering is iterative. Every prompt is a hypothesis:

```
Write prompt
     │
     ▼
Run it 3-5 times          ← sample multiple runs (especially with high temperature)
     │
     ▼
Analyze failures           ← what went wrong?
     │
     ├── Wrong format? → Add explicit format instructions
     ├── Wrong tone?   → Adjust role/system prompt
     ├── Too vague?    → Add specific constraints
     ├── Too verbose?  → Add "Be concise. Max X sentences."
     └── Wrong facts?  → Consider RAG (the model can't know your data)
     │
     ▼
Refine and repeat
```

---

## 🛠️ Prompt Templates: A Library

Keep reusable prompt templates. Here are the most useful ones:

### Summarization Template
```
You are a concise summarizer. 
Summarize the following [CONTENT TYPE] for [AUDIENCE].
Format: [BULLET POINTS / ONE PARAGRAPH / TLDR + DETAILS]
Length: [SHORT/MEDIUM/DETAILED]

Content:
[PASTE CONTENT HERE]
```

### Code Review Template
```
You are a senior [LANGUAGE] engineer. Review this code for:
1. Bugs and logic errors
2. Security vulnerabilities  
3. Performance issues
4. Code style/readability

For each issue found, provide:
- Issue: (what's wrong)
- Severity: (Critical/High/Medium/Low)
- Fix: (specific code change)

Code:
[PASTE CODE HERE]
```

### Data Extraction Template
```
Extract [WHAT TO EXTRACT] from the text below.
Return ONLY a JSON object with these keys: [KEY1, KEY2, KEY3]
If a field is not found, use null.
No explanation, no markdown.

Text:
[PASTE TEXT HERE]
```

---

## 📊 Quick Reference Summary

| Technique | Best for | Key tip |
|-----------|----------|---------|
| Zero-shot | Simple well-defined tasks | Be very specific about format |
| Few-shot | Unusual formats, classification | 2–5 examples is usually enough |
| Chain-of-thought | Math, logic, multi-step | Add "step by step" or "think through this" |
| Role prompting | Domain expertise, tone | Be specific about the persona's background |
| Structured output | Programmatic parsing | Say "ONLY return JSON" explicitly |

---

Continue to [02b_prompt_project.md](02b_prompt_project.md) — build a Prompt Playground with side-by-side comparison and A/B testing.
