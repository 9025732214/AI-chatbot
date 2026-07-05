import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import base64

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 AI Chatbot")
st.caption("Chat, ask questions about a PDF, upload an image, or ask me to generate an image!")

BASE_INSTRUCTION = (
    "You are a helpful, clear AI assistant. Follow these rules:\n"
    "- Give clear, well-organized answers. Use bullet points or numbered lists when helpful.\n"
    "- Use headings or bold text for key terms when it improves clarity.\n"
    "- Be concise: avoid unnecessary repetition or filler sentences.\n"
    "- When writing math, use plain readable text only. Never use LaTeX commands "
    "like \\tfrac, \\bigl, \\frac, or backslash symbols. Write fractions as a/b, "
    "and coordinates simply as (x, y)."
)

with st.expander("📄 Upload a PDF to ask questions about it", expanded=False):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if "pdf_chunks" not in st.session_state:
        st.session_state.pdf_chunks = []

    def extract_chunks(pdf_file, chunk_size=300):
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        return [c.strip() for c in chunks if c.strip()]

    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Reading PDF..."):
                st.session_state.pdf_chunks = extract_chunks(uploaded_file)
            st.success(f"Processed! {len(st.session_state.pdf_chunks)} sections loaded.")

if st.session_state.get("pdf_chunks"):
    st.info(f"📌 Currently loaded: {len(st.session_state.pdf_chunks)} PDF sections available for questions.")
    if st.button("Clear loaded PDF"):
        st.session_state.pdf_chunks = []
        st.rerun()

with st.expander("🖼️ Upload an image/screenshot to ask about it", expanded=False):
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], key="image_uploader")
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded image", width=300)
        st.session_state.current_image = uploaded_image

if "current_image" not in st.session_state:
    st.session_state.current_image = None

def encode_image(image_file):
    image_file.seek(0)
    return base64.b64encode(image_file.read()).decode("utf-8")

def get_relevant_chunks(question, chunks, top_k=1):
    if not chunks:
        return []
    vectorizer = TfidfVectorizer().fit(chunks + [question])
    chunk_vectors = vectorizer.transform(chunks)
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def wants_image(text):
    keywords = ["generate image", "create image", "draw", "make an image", "image of", "picture of", "generate a picture"]
    return any(k in text.lower() for k in keywords)

def extract_image_prompt(text):
    for k in ["generate image of", "create image of", "draw", "make an image of", "image of", "picture of", "generate a picture of"]:
        if k in text.lower():
            idx = text.lower().find(k) + len(k)
            return text[idx:].strip()
    return text

def get_image_url(prompt):
    encoded = urllib.parse.quote(prompt)
    return f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], use_container_width=True)
        else:
            st.write(msg["content"])

user_input = st.chat_input("Ask a question, upload an image, or say 'draw a sunset'...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
    with st.chat_message("user"):
        st.write(user_input)

    if st.session_state.current_image is not None:
        base64_image = encode_image(st.session_state.current_image)
        response = client.chat.completions.create(
            model="qwen/qwen3.6-27b",
            messages=[
                {"role": "system", "content": BASE_INSTRUCTION},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )
        ai_reply = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": ai_reply, "type": "text"})
        with st.chat_message("assistant"):
            st.write(ai_reply)
        st.session_state.current_image = None

    elif wants_image(user_input):
        prompt = extract_image_prompt(user_input)
        with st.spinner("Generating image..."):
            image_url = get_image_url(prompt)
        st.session_state.messages.append({"role": "assistant", "content": image_url, "type": "image"})
        with st.chat_message("assistant"):
            st.image(image_url, use_container_width=True)

    else:
        context = ""
        if st.session_state.pdf_chunks:
            relevant = get_relevant_chunks(user_input, st.session_state.pdf_chunks, top_k=1)
            context = "\n\n".join(relevant)

        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m.get("type") == "text"]

        if context:
            system_prompt = (
                BASE_INSTRUCTION + "\n\nYou HAVE been given excerpts from a PDF the user uploaded, shown below. "
                "You DO have access to this content. Never say you cannot access files. "
                "Answer the user's question using this content:\n\n" + context
            )
            messages_to_send = [{"role": "system", "content": system_prompt}] + chat_history
        else:
            messages_to_send = [{"role": "system", "content": BASE_INSTRUCTION}] + chat_history

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages_to_send
        )
        ai_reply = response.choices[0].message.content

        st.session_state.messages.append({"role": "assistant", "content": ai_reply, "type": "text"})
        with st.chat_message("assistant"):
            st.write(ai_reply)