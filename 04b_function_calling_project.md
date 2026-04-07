# 04b – Function Calling & Agents (Project)

> **What we'll build**: A multi-tool AI agent with weather, news, calculator, and web search tools — using a proper ReAct loop with iteration limits and error handling.

---

## Project: Multi-Tool AI Agent

### Create `projects/agent.py`

```python
"""
Multi-Tool AI Agent with ReAct loop.
Run with: python projects/agent.py
Or with UI: streamlit run projects/agent.py
"""

import json
import re
import requests
import math
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

# ═══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# Each tool has: a Python function, a description for the LLM
# ═══════════════════════════════════════════════════════════════

def get_weather(city: str) -> str:
    """Get current weather for a city using wttr.in (free, no API key)."""
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text.strip()
        return f"Weather data unavailable for {city}"
    except Exception as e:
        return f"Error fetching weather: {e}"

def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Allow only safe math operations
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        
        # Block dangerous builtins
        safe_expression = re.sub(r'[^0-9+\-*/().%,sqrt\s]', '', expression)
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

def get_current_datetime() -> str:
    """Return the current date and time."""
    now = datetime.now()
    return now.strftime("Today is %A, %B %d, %Y. Current time: %I:%M %p")

def search_wikipedia(topic: str) -> str:
    """Search Wikipedia for a brief summary of a topic."""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            # Return first 500 characters
            return extract[:500] + ("..." if len(extract) > 500 else "")
        return f"No Wikipedia article found for '{topic}'"
    except Exception as e:
        return f"Search error: {e}"

def get_mock_news(topic: str = "technology") -> str:
    """
    Returns mock news headlines. 
    Replace this with a real news API (e.g., GNews, NewsAPI) for production.
    """
    mock_news = {
        "technology": [
            "New open-source AI model beats GPT-4 on coding benchmarks",
            "Google announces quantum computing breakthrough",
            "India's tech startup ecosystem reaches record funding in Q4"
        ],
        "business": [
            "Sensex reaches new all-time high as FIIs return to Indian markets",
            "RBI holds interest rates steady amid inflation concerns",
            "Infosys and TCS report strong quarterly earnings"
        ],
        "sports": [
            "India wins test series against Australia 3-1",
            "Neeraj Chopra breaks his own javelin world record",
            "Chennai Super Kings qualify for IPL playoffs"
        ]
    }
    
    # Find matching topic
    for key in mock_news:
        if key in topic.lower():
            headlines = mock_news[key]
            return "\n".join([f"• {h}" for h in headlines])
    
    # Default
    return "\n".join([f"• {h}" for h in mock_news["technology"]])

# ═══════════════════════════════════════════════════════════════
# TOOL REGISTRY
# Maps tool names to functions and descriptions
# ═══════════════════════════════════════════════════════════════

TOOLS = {
    "get_weather": {
        "function": get_weather,
        "description": "Get current weather for a city. Use when user asks about weather, temperature, or if they need an umbrella.",
        "parameters": {"city": "Name of the city (string)"}
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression. Use for any arithmetic, percentages, or math calculations.",
        "parameters": {"expression": "Math expression as string, e.g. '15 * 8 + 100' or 'sqrt(144)'"}
    },
    "get_current_datetime": {
        "function": get_current_datetime,
        "description": "Get the current date and time. Use when user asks what time it is, what today's date is, or what day of the week it is.",
        "parameters": {}
    },
    "search_wikipedia": {
        "function": search_wikipedia,
        "description": "Search Wikipedia for factual information about a topic, person, place, or concept.",
        "parameters": {"topic": "Topic to search for (string)"}
    },
    "get_news": {
        "function": get_mock_news,
        "description": "Get latest news headlines. Use when user asks about current news, recent events, or what's happening.",
        "parameters": {"topic": "News topic: 'technology', 'business', 'sports' (string)"}
    }
}

def build_tools_description():
    """Build a text description of all tools for the LLM prompt."""
    desc = "You have access to these tools:\n\n"
    for name, info in TOOLS.items():
        desc += f"Tool: {name}\n"
        desc += f"  Description: {info['description']}\n"
        if info['parameters']:
            desc += f"  Parameters: {info['parameters']}\n"
        desc += "\n"
    return desc

# ═══════════════════════════════════════════════════════════════
# AGENT LOOP
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

{tools_description}

To use a tool, respond with ONLY a JSON object (no explanation):
{{"tool": "tool_name", "arguments": {{"param1": "value1"}}}}

If you don't need a tool, respond with:
{{"response": "your final answer here"}}

Important:
- Use tools when you need real-time data (weather, news, current time)
- Use calculate for any math
- For general knowledge questions, answer directly without tools
- After getting a tool result, formulate a clear, friendly final response
"""

def execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool and return its result."""
    if tool_name not in TOOLS:
        return f"Unknown tool: {tool_name}. Available: {list(TOOLS.keys())}"
    
    tool_fn = TOOLS[tool_name]["function"]
    
    try:
        if arguments:
            result = tool_fn(**arguments)
        else:
            result = tool_fn()
        return str(result)
    except TypeError as e:
        return f"Wrong arguments for {tool_name}: {e}"
    except Exception as e:
        return f"Tool execution error: {e}"

def parse_llm_output(text: str) -> dict:
    """Parse the LLM's JSON output, handling common formatting issues."""
    # Strip thinking blocks (DeepSeek-R1 sometimes outputs <think>...</think>)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback: treat entire response as a text response
    return {"response": text}

def run_agent(user_query: str, max_iterations: int = 5) -> tuple[str, list]:
    """
    Run the agent loop.
    Returns: (final_answer, list_of_steps)
    """
    steps = []  # Track what the agent did
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(tools_description=build_tools_description())
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
    
    for iteration in range(max_iterations):
        # Get LLM decision
        full_response = ""
        response = client.chat_completion(
            messages=messages,
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=300,
            temperature=0.1,  # low temp for reliable JSON
            stream=True
        )
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
        
        # Parse the LLM output
        decision = parse_llm_output(full_response)
        
        # Case 1: LLM wants to call a tool
        if "tool" in decision:
            tool_name = decision["tool"]
            arguments = decision.get("arguments", {})
            
            step = {
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments
            }
            
            # Execute the tool
            tool_result = execute_tool(tool_name, arguments)
            step["result"] = tool_result
            steps.append(step)
            
            # Add tool call and result to conversation history
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}: {tool_result}\n\nNow provide your final answer to the user."
            })
        
        # Case 2: LLM has a final response
        elif "response" in decision:
            final_answer = decision["response"]
            steps.append({"type": "final_answer", "text": final_answer})
            return final_answer, steps
        
        # Safety: if we can't parse, treat as final response
        else:
            return full_response, steps
    
    # If we hit max iterations without a final answer
    return "I wasn't able to complete that task within the allowed steps.", steps


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

def main_ui():
    st.set_page_config(page_title="AI Agent", page_icon="🤖", layout="wide")
    st.title("🤖 Multi-Tool AI Agent")
    st.markdown("_Ask me anything — I can check weather, do math, look up facts, and more._")
    
    # Sidebar: Available tools
    with st.sidebar:
        st.header("🛠️ Available Tools")
        for name, info in TOOLS.items():
            with st.expander(f"`{name}`"):
                st.markdown(info["description"])
                if info["parameters"]:
                    st.markdown("**Parameters:**")
                    for param, desc in info["parameters"].items():
                        st.markdown(f"- `{param}`: {desc}")
        
        st.markdown("---")
        st.markdown("**Try asking:**")
        examples = [
            "What's the weather in Chennai?",
            "What is 15% of 3500?",
            "What day is today?",
            "Tell me about the Python programming language",
            "What's in the technology news today?",
            "What's the weather in Mumbai and also what is 100 * 1.18?",
        ]
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.prefill = ex
    
    # Chat interface
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    
    # Display history
    for exchange in st.session_state.agent_history:
        with st.chat_message("user"):
            st.write(exchange["query"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            if exchange["steps"]:
                with st.expander(f"🔍 Agent steps ({len(exchange['steps'])} steps)"):
                    for i, step in enumerate(exchange["steps"]):
                        if step["type"] == "tool_call":
                            st.markdown(f"**Step {i+1}: Tool Call**")
                            st.code(f"Tool: {step['tool']}\nArgs: {step['arguments']}\nResult: {step['result']}")
                        elif step["type"] == "final_answer":
                            st.markdown(f"**Step {i+1}: Final Answer**")
    
    # Input
    prefill = st.session_state.get("prefill", "")
    query = st.chat_input("Ask anything...")
    
    if query or prefill:
        actual_query = query or prefill
        st.session_state.prefill = ""
        
        with st.chat_message("user"):
            st.write(actual_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                answer, steps = run_agent(actual_query)
            
            st.write(answer)
            
            if len(steps) > 1:  # More than just the final answer
                with st.expander(f"🔍 Agent steps ({len(steps)} steps)"):
                    for i, step in enumerate(steps):
                        if step["type"] == "tool_call":
                            st.markdown(f"**Step {i+1}: Tool Call**")
                            st.code(f"Tool: {step['tool']}\nArgs: {step['arguments']}\nResult: {step['result']}")
        
        st.session_state.agent_history.append({
            "query": actual_query,
            "answer": answer,
            "steps": steps
        })

if __name__ == "__main__":
    main_ui()
```

---

## Step 2: Run It

```bash
streamlit run projects/agent.py
```

---

## Step 3: Test These Queries

| Query | What the agent should do |
|-------|--------------------------|
| `"What's the weather in Mumbai?"` | Call `get_weather("Mumbai")` |
| `"What is 23% of 15000?"` | Call `calculate("0.23 * 15000")` |
| `"What day is today?"` | Call `get_current_datetime()` |
| `"Tell me about the Taj Mahal"` | Call `search_wikipedia("Taj Mahal")` |
| `"Weather in Delhi and also what's 100 + 200?"` | Call both tools in sequence |

---

## Step 4: Run as Command-Line Agent (No UI)

Add this to the bottom of `agent.py` and run with `python projects/agent.py`:

```python
# Command-line mode (comment out main_ui() call and use this instead)
if __name__ == "__main__" and not "streamlit" in str(os.environ.get("PATH", "")):
    print("🤖 AI Agent — type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        
        answer, steps = run_agent(user_input)
        
        # Show steps
        for step in steps:
            if step["type"] == "tool_call":
                print(f"  [Tool: {step['tool']}({step['arguments']})] → {step['result']}")
        
        print(f"Agent: {answer}\n")
```

---

## 🧪 Challenges

1. **Add a `write_file(filename, content)` tool** that saves content to disk. Ask the agent: "Write a poem about AI and save it to poem.txt". Observe the agent create the file.

2. **Add memory between sessions**: Save `agent_history` to a JSON file and reload it on startup. The agent will remember previous conversations.

3. **Add a `code_executor(python_code)` tool** using `subprocess` that safely runs Python code snippets in a sandboxed way. Only allow `print()` statements for output.

4. **Multi-step planning test**: Ask "Check the weather in 3 Indian cities (Mumbai, Delhi, Bangalore) and tell me which is the hottest." The agent should call the weather tool 3 times.

---

## ✅ What You Learned

- How function calling converts text output to structured tool commands
- How to build an agent loop with ReAct pattern
- How to write effective tool descriptions
- How to handle tool errors and loop limits
- The difference between a basic LLM call and an agentic system

Next: [05a_finetuning_theory.md](05a_finetuning_theory.md) — change the model's personality.
