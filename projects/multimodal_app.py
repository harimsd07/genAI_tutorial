"""
Multi-Modal AI Playground.
Run with: streamlit run projects/multimodal_app.py

Tabs:
  1. Image Captioning   – describe any photo
  2. Visual Q&A         – ask questions about images
  3. Text-to-Image      – generate images from descriptions
  4. Vision LLM         – deep scene understanding with Llama Vision
"""

import streamlit as st, io, base64, requests
from PIL import Image
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()

def img_to_b64(img: Image.Image) -> str:
    return base64.b64encode(img_to_bytes(img)).decode()

st.set_page_config(page_title="Multi-Modal AI", page_icon="🖼️", layout="wide")
st.title("🖼️ Multi-Modal AI Playground")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🏷️ Image Captioning", "❓ Visual Q&A", "🎨 Text-to-Image", "🔍 Vision LLM"]
)

# ── Tab 1: Captioning ─────────────────────────────────────────
with tab1:
    st.header("Generate Captions")
    src = st.radio("Source", ["Upload", "URL"], horizontal=True)
    img = None
    if src == "Upload":
        f = st.file_uploader("Image", type=["jpg","png","jpeg","webp"], key="cap_up")
        if f: img = Image.open(f)
    else:
        url = st.text_input("URL", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg")
        if st.button("Load", key="cap_load"):
            try:
                img = Image.open(io.BytesIO(requests.get(url, timeout=5).content))
            except Exception as e:
                st.error(str(e))
    if img:
        c1, c2 = st.columns(2)
        c1.image(img, use_column_width=True)
        with c2:
            if st.button("Caption it", type="primary"):
                with st.spinner("Analyzing…"):
                    try:
                        cap = client.image_to_text(img_to_bytes(img),
                                                   model="Salesforce/blip-image-captioning-large")
                        st.success("Done!")
                        st.markdown(f"> {cap}")
                    except Exception as e:
                        st.error(str(e))

# ── Tab 2: Visual Q&A ─────────────────────────────────────────
with tab2:
    st.header("Ask Questions About an Image")
    vqa_f = st.file_uploader("Image", type=["jpg","png","jpeg"], key="vqa_up")
    if vqa_f:
        vqa_img = Image.open(vqa_f)
        st.image(vqa_img, width=380)
        q = st.text_input("Question", "What color is it?", key="vqa_q")
        if q and st.button("Ask", type="primary"):
            with st.spinner("Thinking…"):
                try:
                    result = client.visual_question_answering(
                        img_to_bytes(vqa_img), question=q,
                        model="dandelin/vilt-b32-finetuned-vqa"
                    )
                    if isinstance(result, list):
                        for item in result[:3]:
                            st.markdown(f"**{item.get('answer')}** — {item.get('score',0)*100:.1f}%")
                    else:
                        st.markdown(f"**Answer:** {result}")
                except Exception as e:
                    st.error(str(e))

# ── Tab 3: Text-to-Image ──────────────────────────────────────
with tab3:
    st.header("Generate Images from Text")
    c1, c2 = st.columns(2)
    with c1:
        prompt = st.text_area("Describe your image",
            "A majestic tiger in a misty Indian forest at golden hour, photorealistic", height=90)
        neg    = st.text_area("Avoid (negative prompt)", "blurry, low quality, watermark", height=50)
        st.markdown("""**Tips:** add style words like *photorealistic*, *oil painting*, *digital art*,
        and quality words like *highly detailed*, *8k*, *masterpiece*.""")
    with c2:
        if st.button("🎨 Generate", type="primary"):
            with st.spinner("Drawing… (15-30 sec)"):
                try:
                    image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev",
                                                 negative_prompt=neg or None)
                    st.image(image, use_column_width=True)
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.download_button("⬇️ Download", buf.getvalue(), "generated.png", "image/png")
                except Exception as e:
                    st.error(str(e))
                    st.info("Alternative: stabilityai/stable-diffusion-2-1")

# ── Tab 4: Vision LLM ─────────────────────────────────────────
with tab4:
    st.header("Deep Image Analysis")
    st.markdown("Llama 3.2 Vision understands complex scenes, reads text in images, and reasons about content.")
    anlz_f = st.file_uploader("Image", type=["jpg","png","jpeg"], key="anlz_up")
    anlz_p = st.text_area("What to analyze",
        "Describe this image in detail. What objects are present? What is happening?", height=80)
    if anlz_f and anlz_p and st.button("Analyze", type="primary"):
        anlz_img = Image.open(anlz_f)
        st.image(anlz_img, width=380)
        with st.spinner("Analyzing…"):
            try:
                resp = client.chat_completion(
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_to_b64(anlz_img)}"}},
                        {"type": "text", "text": anlz_p}
                    ]}],
                    model="meta-llama/Llama-3.2-11B-Vision-Instruct",
                    max_tokens=500, stream=False
                )
                st.markdown(resp.choices[0].message.content)
            except Exception as e:
                st.error(str(e))
                st.info("Make sure meta-llama/Llama-3.2-11B-Vision-Instruct is available on your free tier.")
