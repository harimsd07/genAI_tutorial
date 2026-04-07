# 04a – Function Calling & Agents (Theory)

> **Core idea**: Normally an LLM can only *output text*. Function calling lets it output *structured commands* that your code executes — giving AI the ability to act in the world.

---

## 🧠 The Problem With Text-Only AI

A pure language model can tell you "The weather in Mumbai is 32°C" — but only if it already knows that from training. It can't actually check the weather right now.

Similarly, it can't:
- Send an email
- Look up the current stock price
- Write a file to disk
- Search the web
- Book a calendar event

**Function calling bridges this gap.**

---

## 📐 How Function Calling Works — Step by Step

```
User: "What's the weather in Mumbai today?"
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  LLM receives:                                          │
│  - User message                                         │
│  - Description of available tools (functions)           │
│                                                         │
│  LLM thinks: "I need weather data. I have a             │
│  get_weather tool that takes a city name."              │
│                                                         │
│  LLM outputs (NOT a text response):                     │
│  {                                                      │
│    "tool": "get_weather",                               │
│    "arguments": { "city": "Mumbai" }                    │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  YOUR CODE executes:                                    │
│  result = get_weather("Mumbai")                         │
│  → "32°C, partly cloudy, humidity 78%"                  │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Your code sends back to LLM:                           │
│  - Original conversation                                │
│  - Tool call it made                                    │
│  - Tool result: "32°C, partly cloudy, humidity 78%"     │
│                                                         │
│  LLM now generates final response:                      │
│  "The weather in Mumbai today is 32°C and partly        │
│   cloudy, with 78% humidity. It's a warm day — bring    │
│   water if you're going out!"                           │
└─────────────────────────────────────────────────────────┘
```

**Key insight**: The LLM never actually executes code. It outputs a request for a tool to be called. Your code is the executor. You control what tools exist and what they do.

---

## 🤖 What is an AI Agent?

A **basic LLM** is a single turn: prompt in → response out.

An **AI Agent** is an LLM in a loop, able to take multiple actions before returning a final answer:

```
User: "Find today's top AI news, summarize the 3 most important stories, 
       and calculate how many days since the oldest story was published."

Agent loop:
─────────────────────────────────────────────────────────
Step 1: Plan
  → "I need to search for AI news, then do some date math"

Step 2: Call search_web("AI news today")
  → Gets 10 news articles back

Step 3: Call search_web("latest AI research papers")
  → Gets more results

Step 4: Reason about all results
  → Identifies top 3 stories

Step 5: Call calculate("days_since('2024-01-15')")
  → Gets "45 days"

Step 6: Compose final answer
  → Returns formatted response to user
─────────────────────────────────────────────────────────
```

**The difference**:
- Basic LLM: one shot, output depends only on prompt
- Agent: iterative, uses tools, adapts based on results

---

## 🏗️ The ReAct Pattern (Reason + Act)

The most common agent architecture. Each step has three phases:

```
Thought: "I need to check the weather first, then decide if outdoor 
          recommendations make sense."
Action: get_weather(city="Bangalore")
Observation: "28°C, sunny"

Thought: "It's nice weather. Now I'll search for outdoor activities."
Action: search_places(query="parks in Bangalore", type="outdoor")
Observation: [Cubbon Park, Lalbagh Gardens, ...]

Thought: "I have enough information to give a good recommendation."
Action: FINISH
Final Answer: "Great weather for outdoor activities! Consider visiting 
              Cubbon Park or Lalbagh Gardens..."
```

ReAct forces the model to explicitly state its reasoning before each action, making it easier to debug when something goes wrong.

---

## ⚠️ When Function Calling Goes Wrong

**Hallucinated tool calls**: The model might invent function arguments that don't exist.
**Tool call loops**: The model keeps calling tools without terminating.
**Wrong tool selection**: The model picks the wrong function.

**How to prevent**:
1. Write very clear tool descriptions
2. Set a max iteration limit in your agent loop
3. Validate arguments before executing

---

## Tool Description Best Practices

The model decides which tool to call based on your description. Write them like good documentation:

```python
# ❌ Poor description — model will call this incorrectly
{"name": "get_data", "description": "Gets some data"}

# ✅ Good description — model knows exactly when to use this
{
    "name": "get_weather",
    "description": "Get the current weather conditions for a specific city. "
                   "Use this when the user asks about weather, temperature, "
                   "or whether they need an umbrella. Returns temperature in "
                   "Celsius and a weather description.",
    "parameters": {
        "city": {
            "type": "string",
            "description": "City name, e.g. 'Mumbai', 'Delhi', 'Bangalore'",
            "required": True
        }
    }
}
```

---

Continue to [04b_function_calling_project.md](04b_function_calling_project.md) — build a multi-tool agent.
