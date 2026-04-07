# 07 – Multi-Modal AI

> **Core idea**: Most LLMs only process text. Multi-modal models can process images, audio, and video alongside text — enabling entirely new categories of applications.

---

## 🧠 What "Multi-Modal" Really Means

The word "modal" refers to a type or channel of data:
- Text modal: words, sentences, documents
- Visual modal: photos, diagrams, videos
- Audio modal: speech, music, sound
- Structured modal: tables, code, JSON

A **multi-modal model** has been trained on pairs of different modalities together — e.g., millions of (image, caption) pairs — so it learned to connect meaning across them.

```
How vision-language models are trained:
─────────────────────────────────────────
Training data example:
  Image: [photo of a cat sitting on a couch]
  Caption: "A tabby cat resting on a brown sofa"

After seeing millions of these pairs, the model learns:
  "cat" ↔ [visual features of cats]
  "sofa" ↔ [visual features of sofas]
  
Now given just an image, it can generate a caption.
Given just a caption, it can identify what should be in an image.
```

---

## 🗂️ The Four Multi-Modal Capabilities

| Capability | Input | Output | Example use |
|------------|-------|--------|-------------|
| **Image Captioning** | Image | Text | "Describe what's in this photo" |
| **Visual Q&A (VQA)** | Image + Question | Answer | "What color is the car in this photo?" |
| **Text-to-Image** | Text prompt | Image | "A futuristic city at sunset" |
| **Vision-Language Models** | Image + Text | Text | "Read this handwritten note" / OCR |

---

## 🔬 How Image Captioning Works Internally

```
Input image
    │
    ▼
Image Encoder (e.g., Vision Transformer / ViT)
  → Splits image into patches (e.g., 16x16 pixels each)
  → Converts each patch to a vector
  → Produces a sequence of visual embeddings
    │
    ▼
Cross-Attention Layer
  → Image embeddings attend to text embeddings (and vice versa)
  → Model learns "which part of the image matches which word"
    │
    ▼
Text Decoder (language model)
  → Generates caption token by token
  → Each token attends to relevant image regions
    │
    ▼
Output: "A tabby cat sitting on a brown sofa near a window"
```

---

## 🏗️ Project: Multi-Modal Playground

### Create `projects/multimodal_app.py`

```python
"""
Multi-Modal AI Playground.
Run with: streamlit run projects/multimodal_app.py
"""

import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64
import requests

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

st.set_page_config(page_title="Multi-Modal AI", page_icon="🖼️", layout="wide")
st.title("🖼️ Multi-Modal AI Playground")

def pil_to_bytes(img: Image.Image, format="JPEG") -> bytes:
    """Convert PIL image to bytes."""
    buf = io.BytesIO()
    img = img.convert("RGB")  # Ensure RGB for JPEG
    img.save(buf, format=format)
    return buf.getvalue()

def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string."""
    return base64.b64encode(pil_to_bytes(img)).decode()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏷️ Image Captioning", 
    "❓ Visual Q&A", 
    "🎨 Text-to-Image",
    "🔍 Image Analysis (Vision LLM)"
])

# ─── Tab 1: Image Captioning ──────────────────────────────────
with tab1:
    st.header("Generate Captions for Images")
    st.markdown("""
    The model analyzes an image and produces a natural language description.
    
    **Model used**: `Salesforce/blip-image-captioning-large`  
    BLIP (Bootstrapping Language-Image Pretraining) was trained on 129M image-text pairs.
    """)
    
    img_source = st.radio("Image source:", ["Upload file", "Enter URL"])
    img = None
    
    if img_source == "Upload file":
        uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg", "webp"])
        if uploaded:
            img = Image.open(uploaded)
    else:
        url = st.text_input("Image URL", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg")
        if url and st.button("Load Image"):
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(io.BytesIO(response.content))
            except Exception as e:
                st.error(f"Could not load image: {e}")
    
    if img:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Input Image", use_column_width=True)
        with col2:
            if st.button("Generate Caption", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        caption = client.image_to_text(
                            pil_to_bytes(img),
                            model="Salesforce/blip-image-captioning-large"
                        )
                        st.success("Caption generated!")
                        st.markdown(f"### 📝 Caption:")
                        st.markdown(f"> {caption}")
                        
                        st.markdown("---")
                        st.markdown("**How this works:**")
                        st.markdown("""
                        1. Image is split into 16×16 pixel patches
                        2. Each patch becomes a vector (visual token)  
                        3. Vision encoder processes all patches together
                        4. Language decoder generates caption word by word
                        """)
                    except Exception as e:
                        st.error(f"Error: {e}")

# ─── Tab 2: Visual Q&A ────────────────────────────────────────
with tab2:
    st.header("Ask Questions About Images")
    st.markdown("""
    Visual Question Answering (VQA) — the model answers specific questions about an image.
    
    **Model used**: `dandelin/vilt-b32-finetuned-vqa`  
    ViLT processes image patches and text tokens together through the same transformer.
    """)
    
    vqa_img = None
    vqa_upload = st.file_uploader("Upload image for Q&A", type=["jpg", "png", "jpeg"])
    
    if vqa_upload:
        vqa_img = Image.open(vqa_upload)
        st.image(vqa_img, caption="Your image", width=400)
    
    question = st.text_input("Your question:", "What color is it?")
    
    if vqa_img and question and st.button("Ask Question", type="primary"):
        with st.spinner("Thinking..."):
            try:
                result = client.visual_question_answering(
                    pil_to_bytes(vqa_img),
                    question=question,
                    model="dandelin/vilt-b32-finetuned-vqa"
                )
                st.success("Answer found!")
                
                if isinstance(result, list):
                    st.markdown("**Top answers with confidence:**")
                    for item in result[:3]:
                        confidence = item.get('score', 0) * 100
                        answer = item.get('answer', 'Unknown')
                        st.markdown(f"- **{answer}** ({confidence:.1f}% confidence)")
                else:
                    st.markdown(f"**Answer**: {result}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Good test questions guide
    with st.expander("💡 Good test questions for VQA"):
        st.markdown("""
        VQA models work best with simple, specific questions:
        - Color: "What color is the car?"
        - Count: "How many people are in the image?"
        - Object: "What animal is in the picture?"
        - Action: "What is the person doing?"
        - Yes/No: "Is the sky visible?"
        
        VQA struggles with:
        - Complex reasoning: "Why does the person look sad?"
        - Reading text in images (use OCR for that)
        - Fine details
        """)

# ─── Tab 3: Text-to-Image ─────────────────────────────────────
with tab3:
    st.header("Generate Images from Text")
    st.markdown("""
    Text-to-image generation: describe what you want and the model creates it.
    
    **Model used**: `black-forest-labs/FLUX.1-dev`  
    FLUX uses a diffusion process: starts with random noise and iteratively refines it
    guided by your text description.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        prompt = st.text_area(
            "Describe your image:",
            "A majestic tiger in a misty Indian forest at golden hour, photorealistic, detailed",
            height=100
        )
        negative_prompt = st.text_area(
            "What to avoid (negative prompt):",
            "blurry, low quality, cartoon, watermark",
            height=60
        )
        
        st.markdown("**Prompt tips:**")
        st.markdown("""
        - Be specific about style: "photorealistic", "oil painting", "digital art"
        - Specify lighting: "golden hour", "dramatic shadows", "soft diffused light"
        - Add quality keywords: "highly detailed", "8k", "masterpiece"
        - Specify mood: "serene", "dramatic", "whimsical"
        """)
    
    with col2:
        if st.button("🎨 Generate Image", type="primary"):
            with st.spinner("Drawing... (this may take 15-30 seconds)"):
                try:
                    image = client.text_to_image(
                        prompt,
                        model="black-forest-labs/FLUX.1-dev",
                        negative_prompt=negative_prompt if negative_prompt else None,
                    )
                    st.image(image, caption=prompt[:100], use_column_width=True)
                    
                    # Download button
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button(
                        "⬇️ Download Image",
                        data=buf.getvalue(),
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Try a different model: stabilityai/stable-diffusion-2-1")

# ─── Tab 4: Vision LLM Analysis ───────────────────────────────
with tab4:
    st.header("Deep Image Analysis with Vision LLM")
    st.markdown("""
    Vision-Language Models (VLMs) like LLaVA combine a vision encoder with a full language model.
    They can answer complex questions, read text in images, describe detailed scenes, and reason about images.
    
    **Model**: Provided via HF API (vision-capable models)
    """)
    
    analysis_upload = st.file_uploader("Upload image for analysis", type=["jpg", "png", "jpeg"], key="analysis")
    analysis_prompt = st.text_area(
        "What to analyze:",
        "Describe this image in detail. What objects are present? What is happening? What is the mood or atmosphere?",
        height=80
    )
    
    if analysis_upload and analysis_prompt and st.button("Analyze Image", type="primary"):
        analysis_img = Image.open(analysis_upload)
        st.image(analysis_img, width=400)
        
        with st.spinner("Analyzing..."):
            try:
                img_b64 = pil_to_base64(analysis_img)
                
                # Use vision-capable model via messages API
                response = client.chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                },
                                {
                                    "type": "text",
                                    "text": analysis_prompt
                                }
                            ]
                        }
                    ],
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                    max_tokens=500,
                    stream=False
                )
                
                analysis_result = response.choices[0].message.content
                st.markdown("**Analysis:**")
                st.markdown(analysis_result)
                
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("This tab requires a vision-capable model. Try: meta-llama/Llama-3.2-11B-Vision-Instruct")
```

---

## Step 2: Run It

```bash
streamlit run projects/multimodal_app.py
```

---

## 🧪 Challenges

1. **Auto Alt-Text Generator**: Build a tool that takes a folder of images and generates accessibility alt-text for each one using the captioning model. Useful for making websites accessible.

2. **Document Digitizer**: Use the Vision LLM tab to upload photos of handwritten notes or printed documents. Ask it to transcribe the text. Compare to manual OCR tools.

3. **Image-Based RAG**: Combine multi-modal and RAG: index image captions as text in ChromaDB. Let users search "find me images of cats" and return matching images.

4. **Generate + Evaluate**: Use text-to-image to generate 5 variations of the same prompt. Use the captioning model to generate a caption for each. Compare whether the captions match your original prompt.

---

## ✅ What You Learned

- How vision encoders convert images to vectors
- The difference between captioning, VQA, and text-to-image
- How diffusion models generate images iteratively
- How Vision-Language Models combine visual and text understanding
- How to build a multi-tab multi-modal app

Next: [08_vector_databases.md](08_vector_databases.md) — deep dive into ChromaDB and vector search.
