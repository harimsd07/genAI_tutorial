# 06 – Evaluation

> **Core insight**: You cannot improve what you don't measure. Without systematic evaluation, you're guessing whether your prompts, RAG pipeline, or fine-tuned model is actually better.

---

## 🧠 Why Evaluation Is Hard

Evaluating AI is genuinely difficult for several reasons:

1. **No single right answer**: "Explain photosynthesis" has many valid responses
2. **Dimensions matter differently**: Is brevity more important than accuracy? Depends on the use case.
3. **Human evaluation is expensive**: You can't read 10,000 AI responses manually
4. **LLMs can confidently be wrong**: High fluency ≠ correctness

---

## 📐 The Evaluation Framework

A complete evaluation suite uses multiple complementary metrics:

```
┌───────────────────────────────────────────────────────────┐
│                 EVALUATION DIMENSIONS                      │
│                                                            │
│  1. CORRECTNESS         Does the answer match facts?       │
│     → Semantic similarity to gold answer                   │
│     → LLM-as-judge scoring                                 │
│                                                            │
│  2. GROUNDEDNESS        Is it from the source? (RAG only)  │
│     → Is the answer supported by retrieved context?        │
│                                                            │
│  3. RELEVANCE           Does it answer what was asked?     │
│     → Does the response address the actual question?       │
│                                                            │
│  4. COMPLETENESS        Does it cover all aspects?         │
│     → Are important parts missing?                         │
│                                                            │
│  5. FAITHFULNESS        Are there hallucinations?          │
│     → Did the AI make up information?                      │
└───────────────────────────────────────────────────────────┘
```

---

## 🔬 Method 1: Semantic Similarity

**When to use**: When you have a "golden answer" (correct reference answer).

**How it works**: Convert both the generated answer and the golden answer to embeddings. Compare how similar the vectors are.

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

generated = "Paris is the capital city of France"
golden    = "The capital of France is Paris"

# Both answers mean the same thing → high similarity
emb_gen    = model.encode(generated)
emb_golden = model.encode(golden)

similarity = util.cos_sim(emb_gen, emb_golden).item()
# Result: ~0.95 (very similar)
```

**Threshold guide**:
- > 0.90: Essentially the same meaning
- 0.75–0.90: Very similar, minor differences
- 0.50–0.75: Partially related
- < 0.50: Likely wrong answer

**Limitation**: A very fluent, confident wrong answer can sometimes score high. Use alongside LLM-as-judge.

---

## 🔬 Method 2: LLM-as-Judge

**When to use**: When there's no single correct answer, or you want nuanced evaluation.

**How it works**: You ask a second LLM to score the first LLM's output.

```python
def llm_judge(question, generated_answer, golden_answer=None, criteria=None):
    if criteria is None:
        criteria = ["accuracy", "clarity", "completeness"]
    
    if golden_answer:
        prompt = f"""You are an expert evaluator. Score the following answer.

Question: {question}
Generated Answer: {generated_answer}
Reference Answer: {golden_answer}

Score on these criteria (0-10 each):
{chr(10).join(f"- {c}" for c in criteria)}

Return a JSON object:
{{"accuracy": N, "clarity": N, "completeness": N, "reasoning": "brief explanation"}}

Return ONLY the JSON, nothing else."""
    else:
        prompt = f"""You are an expert evaluator. Score this answer.

Question: {question}
Answer: {generated_answer}

Score on: accuracy (0-10), clarity (0-10), helpfulness (0-10)
Return ONLY JSON: {{"accuracy": N, "clarity": N, "helpfulness": N, "reasoning": "..."}}"""
    
    # Call LLM to evaluate
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="deepseek-ai/DeepSeek-R1",
        max_tokens=200,
        temperature=0  # deterministic evaluation
    )
    
    text = response.choices[0].message.content
    
    import json, re
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"error": "Could not parse response", "raw": text}
```

---

## 🏗️ Project: Evaluation Harness

A complete tool for evaluating any AI system with batch testing and reporting.

### Create `projects/evaluate.py`

```python
"""
Evaluation Harness – Systematically test and score AI outputs.
Run with: streamlit run projects/evaluate.py
"""

import streamlit as st
import json
import re
import csv
import io
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

# ─── Evaluation Functions ─────────────────────────────────────

def semantic_similarity(answer_a: str, answer_b: str) -> float:
    """Return cosine similarity [0, 1] between two texts."""
    emb_a = embedder.encode(answer_a, convert_to_tensor=True)
    emb_b = embedder.encode(answer_b, convert_to_tensor=True)
    return float(util.cos_sim(emb_a, emb_b).item())

def llm_judge_score(question: str, generated: str, golden: str = None) -> dict:
    """Ask LLM to rate the answer. Returns dict with scores."""
    if golden:
        prompt = f"""Evaluate this AI answer. Return ONLY JSON, no explanation outside JSON.

Question: {question}
AI Answer: {generated}
Reference Answer: {golden}

JSON format:
{{"accuracy": 0-10, "clarity": 0-10, "completeness": 0-10, "overall": 0-10, "reasoning": "one sentence"}}"""
    else:
        prompt = f"""Evaluate this AI answer. Return ONLY JSON, no explanation outside JSON.

Question: {question}  
AI Answer: {generated}

JSON format:
{{"accuracy": 0-10, "clarity": 0-10, "helpfulness": 0-10, "overall": 0-10, "reasoning": "one sentence"}}"""
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=200,
            temperature=0,
            stream=False
        )
        text = response.choices[0].message.content
        
        # Strip thinking blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "Parse failed", "overall": 0}
    except Exception as e:
        return {"error": str(e), "overall": 0}

def check_hallucination(answer: str, context: str) -> dict:
    """Check if an answer is grounded in the provided context."""
    prompt = f"""Check if this answer is fully supported by the context.

Context: {context}
Answer: {answer}

Return ONLY JSON:
{{"grounded": true/false, "score": 0-10, "unsupported_claims": ["list any claims not in context"]}}"""
    
    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=200,
            temperature=0,
            stream=False
        )
        text = response.choices[0].message.content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"grounded": False, "score": 0}
    except Exception as e:
        return {"error": str(e)}

def generate_ai_answer(question: str, system_prompt: str = "", context: str = "") -> str:
    """Generate an answer from the AI system being evaluated."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    user_content = question
    if context:
        user_content = f"Context: {context}\n\nQuestion: {question}"
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        response = client.chat_completion(
            messages=messages,
            model="deepseek-ai/DeepSeek-R1",
            max_tokens=500,
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def run_full_evaluation(test_case: dict, system_prompt: str = "") -> dict:
    """
    Run all evaluation metrics on a single test case.
    
    test_case format:
    {
        "question": "...",
        "golden_answer": "...",  (optional)
        "context": "...",  (optional, for RAG evaluation)
    }
    """
    question = test_case["question"]
    golden = test_case.get("golden_answer", "")
    context = test_case.get("context", "")
    
    # Generate the answer
    generated = generate_ai_answer(question, system_prompt, context)
    
    result = {
        "question": question,
        "generated": generated,
        "golden": golden,
    }
    
    # Semantic similarity (if golden answer exists)
    if golden:
        result["semantic_similarity"] = semantic_similarity(generated, golden)
    
    # LLM judge
    result["llm_scores"] = llm_judge_score(question, generated, golden or None)
    
    # Hallucination check (if context exists)
    if context:
        result["hallucination_check"] = check_hallucination(generated, context)
    
    return result

# ─── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="AI Evaluator", page_icon="📊", layout="wide")
st.title("📊 AI Evaluation Harness")
st.markdown("_Systematically test and score AI outputs._")

tab1, tab2, tab3 = st.tabs(["Single Test", "Batch Test", "Compare Two Systems"])

# ─── Tab 1: Single Test ───────────────────────────────────────
with tab1:
    st.header("Evaluate a Single Response")
    
    col1, col2 = st.columns(2)
    
    with col1:
        question = st.text_area("Question", "What is the capital of France?")
        generated = st.text_area("AI's Answer (to evaluate)", "Paris is the capital of France.")
    
    with col2:
        golden = st.text_area("Golden Answer (optional)", "The capital of France is Paris.")
        context = st.text_area("Context (optional, for RAG)", "")
    
    if st.button("Evaluate", type="primary"):
        with st.spinner("Running evaluation..."):
            sim = semantic_similarity(generated, golden) if golden else None
            scores = llm_judge_score(question, generated, golden or None)
            halluc = check_hallucination(generated, context) if context else None
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if sim is not None:
                color = "green" if sim > 0.8 else "orange" if sim > 0.5 else "red"
                st.metric("Semantic Similarity", f"{sim:.2%}")
        
        with col_b:
            overall = scores.get("overall", 0)
            st.metric("LLM Overall Score", f"{overall}/10")
        
        with col_c:
            if halluc:
                grounded = halluc.get("grounded", False)
                st.metric("Grounded in Context", "✅ Yes" if grounded else "❌ No")
        
        st.markdown("**LLM Judge Details:**")
        st.json(scores)
        
        if halluc:
            st.markdown("**Hallucination Check:**")
            st.json(halluc)

# ─── Tab 2: Batch Test ────────────────────────────────────────
with tab2:
    st.header("Batch Evaluation")
    st.markdown("Test multiple questions at once and get aggregate statistics.")
    
    system_prompt = st.text_area(
        "System Prompt for AI under test",
        "You are a helpful assistant."
    )
    
    st.markdown("**Test cases (JSON format):**")
    default_tests = json.dumps([
        {"question": "What is the capital of France?", "golden_answer": "Paris"},
        {"question": "What is 2 + 2?", "golden_answer": "4"},
        {"question": "Who wrote Romeo and Juliet?", "golden_answer": "William Shakespeare"},
        {"question": "What is the speed of light?", "golden_answer": "299,792,458 meters per second"},
        {"question": "What is Python used for?", "golden_answer": "Python is used for web development, data science, automation, AI/ML, and general scripting."},
    ], indent=2)
    
    test_cases_json = st.text_area("Test Cases", default_tests, height=200)
    
    if st.button("Run Batch Evaluation", type="primary"):
        try:
            test_cases = json.loads(test_cases_json)
        except:
            st.error("Invalid JSON. Check your test cases format.")
            st.stop()
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, test_case in enumerate(test_cases):
            status.text(f"Evaluating question {i+1}/{len(test_cases)}: {test_case['question'][:50]}...")
            result = run_full_evaluation(test_case, system_prompt)
            results.append(result)
            progress.progress((i + 1) / len(test_cases))
        
        status.text("✅ Evaluation complete!")
        
        # Summary statistics
        st.markdown("---")
        st.subheader("📈 Results Summary")
        
        sim_scores = [r.get("semantic_similarity", 0) for r in results if "semantic_similarity" in r]
        llm_scores = [r["llm_scores"].get("overall", 0) for r in results if "llm_scores" in r]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if sim_scores:
                st.metric("Avg Semantic Similarity", f"{sum(sim_scores)/len(sim_scores):.2%}")
        with col2:
            if llm_scores:
                st.metric("Avg LLM Score", f"{sum(llm_scores)/len(llm_scores):.1f}/10")
        with col3:
            st.metric("Questions Tested", len(results))
        
        # Detailed results table
        st.markdown("**Detailed Results:**")
        rows = []
        for r in results:
            rows.append({
                "Question": r["question"][:60] + "...",
                "Generated": r["generated"][:80] + "...",
                "Similarity": f"{r.get('semantic_similarity', 'N/A'):.2%}" if isinstance(r.get('semantic_similarity'), float) else "N/A",
                "LLM Score": r.get("llm_scores", {}).get("overall", "N/A"),
            })
        
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        
        # Download results
        results_json = json.dumps(results, indent=2)
        st.download_button(
            "⬇️ Download Full Results (JSON)",
            data=results_json,
            file_name="evaluation_results.json",
            mime="application/json"
        )

# ─── Tab 3: Compare Two Systems ───────────────────────────────
with tab3:
    st.header("A/B System Comparison")
    st.markdown("Compare two different prompts/systems on the same questions.")
    
    col1, col2 = st.columns(2)
    with col1:
        system_a = st.text_area("System A prompt", "You are a helpful assistant. Be concise.", key="sysA")
    with col2:
        system_b = st.text_area("System B prompt", "You are a helpful assistant. Be detailed and thorough, providing examples.", key="sysB")
    
    compare_question = st.text_area("Test Question", "Explain what machine learning is.")
    compare_golden = st.text_area("Golden Answer (optional)", "")
    
    if st.button("Compare", type="primary"):
        with st.spinner("Running both systems..."):
            answer_a = generate_ai_answer(compare_question, system_a)
            answer_b = generate_ai_answer(compare_question, system_b)
            
            score_a = llm_judge_score(compare_question, answer_a, compare_golden or None)
            score_b = llm_judge_score(compare_question, answer_b, compare_golden or None)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System A Output:**")
            st.markdown(answer_a)
            st.markdown("**Scores:**")
            st.json(score_a)
        
        with col2:
            st.markdown("**System B Output:**")
            st.markdown(answer_b)
            st.markdown("**Scores:**")
            st.json(score_b)
        
        # Winner
        overall_a = score_a.get("overall", 0)
        overall_b = score_b.get("overall", 0)
        
        if overall_a > overall_b:
            st.success(f"🏆 System A wins! ({overall_a} vs {overall_b})")
        elif overall_b > overall_a:
            st.success(f"🏆 System B wins! ({overall_b} vs {overall_a})")
        else:
            st.info("It's a tie!")
```

---

## Step 2: Run It

```bash
streamlit run projects/evaluate.py
```

---

## 🧪 Challenges

1. **Evaluate your RAG pipeline**: Take 10 questions with known answers from a PDF you indexed. Run the RAG system and evaluate each answer. What's the average score?

2. **Find the breaking point**: Design test cases that *should* make the AI fail — ambiguous questions, questions outside its knowledge, trick questions. What failure modes do you see?

3. **Prompt optimization loop**: Use the batch evaluator to score Prompt A. Modify the system prompt. Score Prompt B. Repeat until score improves. Track scores in a spreadsheet.

---

## ✅ What You Learned

- Why subjective AI evaluation needs multiple metrics
- How semantic similarity works for factual questions
- How to use LLM-as-judge for nuanced scoring
- How to detect hallucinations by checking answer groundedness
- How to run systematic A/B tests on prompts

Next: [07_multimodal.md](07_multimodal.md) — vision + language.
