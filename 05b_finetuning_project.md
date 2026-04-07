# 05b – Fine-Tuning (Project)

> **What we'll build**: Fine-tune a small language model to speak like a specific persona, then compare the base model vs fine-tuned model side by side.

---

## Setup: Google Colab (Free GPU)

We use Google Colab because it provides a free GPU — fine-tuning even small models requires more compute than a typical laptop CPU.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. New notebook → Runtime → Change runtime type → **T4 GPU**
3. Copy each code cell below into Colab cells

---

## Cell 1: Install Libraries

```python
!pip install transformers datasets accelerate peft bitsandbytes -q

import os
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
```

---

## Cell 2: Prepare Your Dataset

We'll fine-tune a model to be a "Socratic tutor" — always asking questions instead of giving answers directly.

```python
from datasets import Dataset

# ── Fine-tuning dataset ───────────────────────────────────────
# Format: list of {"instruction": ..., "response": ...} dicts
# The model will learn to always respond in this style

training_examples = [
    {
        "instruction": "What is machine learning?",
        "response": "Interesting question! Before I explain, let me ask you: when you recognize a friend's face in a crowd, are you applying a rule someone gave you, or are you recognizing a pattern you learned? What do you think that tells us about the difference between rule-based programming and learning from experience?"
    },
    {
        "instruction": "How does Python work?",
        "response": "Great topic! Let me turn that around: when you write a recipe, who executes it — the recipe itself, or a person reading it? How might that idea apply to the relationship between Python code and the Python interpreter?"
    },
    {
        "instruction": "Explain recursion.",
        "response": "Before I explain, consider this: have you ever looked up a word in a dictionary, only to find the definition uses another word you don't know? And then looked that word up too? What does that process feel like, and how might it relate to a function that calls itself?"
    },
    {
        "instruction": "What is a database?",
        "response": "Let's think about it from your experience: if you had to store the phone numbers of 1000 people, how would you organize them so you could find anyone quickly? What problems would that system have as it grew? That might tell us exactly why databases were invented."
    },
    {
        "instruction": "What is an API?",
        "response": "Here's a question to start: when you use a TV remote, do you need to understand the electronics inside the TV to change the channel? What does that simplification give you? How might that principle apply to software systems talking to each other?"
    },
    {
        "instruction": "How does the internet work?",
        "response": "Let me ask you first: if you wanted to send a letter to someone on the other side of the world, how would you ensure it arrives? What systems would need to exist? Now, what if instead of a single letter, you could only send postcard-sized pieces? How would you reassemble them?"
    },
    {
        "instruction": "What is object-oriented programming?",
        "response": "Consider this: if you were designing a video game with different types of characters — warriors, mages, archers — what properties and behaviors would they all share? What would be unique to each? How might you organize your code to avoid writing the same things over and over?"
    },
    {
        "instruction": "What is a neural network?",
        "response": "Before diving in: your brain processes light hitting your retina and somehow produces the experience of seeing a red apple. Do you think this involves someone having programmed every possible image of a red apple, or something else? What alternative might there be?"
    },
]

# Format for instruction fine-tuning
def format_example(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}

### End"""
    }

dataset = Dataset.from_list([format_example(e) for e in training_examples])
print(f"Dataset size: {len(dataset)} examples")
print("\nSample formatted example:")
print(dataset[0]['text'])
```

---

## Cell 3: Load the Base Model with LoRA

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

MODEL_NAME = "microsoft/phi-2"   # Small, fast, good quality
# Alternative: "microsoft/Phi-3-mini-4k-instruct" (better but larger)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config (QLoRA) — fits in free Colab GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load base model
print("Loading model (this takes 1-2 minutes)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                        # Rank: higher = more capacity but more memory
    lora_alpha=32,               # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Expected output: trainable params: ~800K (0.07% of all params)
```

---

## Cell 4: Test the BASE Model (Before Fine-Tuning)

```python
def generate_response(model, tokenizer, prompt, max_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

test_prompt = """### Instruction:
What is machine learning?

### Response:"""

print("BASE MODEL response:")
print("─" * 50)
base_response = generate_response(model, tokenizer, test_prompt)
print(base_response)
print("─" * 50)
# Will give a direct explanation, NOT Socratic style
```

---

## Cell 5: Fine-Tune

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Tokenize dataset
def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Training configuration
training_args = TrainingArguments(
    output_dir="./socratic_tutor",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    warmup_ratio=0.1,
)

# Data collator handles padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (not masked)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")
```

---

## Cell 6: Test the FINE-TUNED Model and Compare

```python
# Test fine-tuned model
test_questions = [
    "What is machine learning?",
    "How does recursion work?",
    "What is a variable in programming?",
]

print("=" * 60)
print("COMPARISON: Base Model vs Fine-Tuned Model")
print("=" * 60)

for question in test_questions:
    prompt = f"""### Instruction:
{question}

### Response:"""
    
    print(f"\n❓ Question: {question}")
    print("\n📖 Fine-Tuned Model (Socratic style):")
    ft_response = generate_response(model, tokenizer, prompt)
    print(ft_response[:400])
    print("─" * 60)
```

---

## Cell 7: Save and Download Your Model

```python
# Save the LoRA adapters (small: ~5-20MB)
model.save_pretrained("./socratic_tutor_final")
tokenizer.save_pretrained("./socratic_tutor_final")

print("Model saved! Download from Colab sidebar → Files → socratic_tutor_final/")
print("\nFiles saved:")
import os
for f in os.listdir("./socratic_tutor_final"):
    size = os.path.getsize(f"./socratic_tutor_final/{f}") / 1024
    print(f"  {f}: {size:.1f} KB")
```

---

## Cell 8: Use Your Saved Model Later

```python
# To reload your fine-tuned model later:
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, ...)
ft_model = PeftModel.from_pretrained(base, "./socratic_tutor_final")

# Or merge LoRA weights into base model permanently:
merged_model = ft_model.merge_and_unload()
merged_model.save_pretrained("./socratic_tutor_merged")
```

---

## 📊 Understanding Training Loss

Watch the loss during training:

```
Epoch 1: loss=2.4  ← model is mostly guessing
Epoch 3: loss=1.8  ← starting to learn patterns
Epoch 5: loss=1.2  ← good learning
Epoch 8: loss=0.8  ← fitting well
Epoch 10: loss=0.5 ← check if overfitting

Signs of overfitting:
  Train loss: 0.3  ↓ (keeps going down)
  Eval loss:  1.8  ↑ (going UP = overfitting)
```

If eval loss rises while train loss falls: reduce epochs or add more training data.

---

## 🧪 Challenges

1. **Custom persona**: Replace the Socratic tutor examples with your own persona — make a model that responds like a fitness coach, a pirate, or a formal legal assistant.

2. **More data = better quality**: Expand the dataset to 50+ examples. Note how the output quality improves.

3. **Compare LoRA ranks**: Train two versions with `r=4` and `r=64`. Compare output quality and training time.

4. **Dataset from real text**: Fine-tune on actual text files using this approach:
```python
with open("your_style_text.txt") as f:
    raw_text = f.read()
# Split into chunks and use as training data
```

---

## ✅ What You Learned

- The difference between pre-training and fine-tuning
- How LoRA reduces compute requirements by 10-100x
- How to prepare a fine-tuning dataset with proper formatting
- How to detect and prevent overfitting
- When fine-tuning is better vs worse than RAG/prompting

Next: [06_evaluation.md](06_evaluation.md) — measure whether your AI is actually good.
