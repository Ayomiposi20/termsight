import streamlit as st
from transformers import pipeline
import torch
import re
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Termsight", page_icon="📜", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #0f1117 0%, #1a1d29 100%); }
h1 { font-size: 3rem !important; background: linear-gradient(90deg, #7dd3fc, #a78bfa);
-webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
.subtitle { color: #94a3b8; font-size: 1.1rem; margin-bottom: 1.5rem; }
.stTextArea textarea, .stTextInput input { background-color: #1e2132 !important;
border: 1px solid #353a52 !important; border-radius: 12px !important; color: #e2e8f0 !important; }
.stButton button { background: linear-gradient(90deg, #7dd3fc, #a78bfa) !important;
color: #0f1117 !important; font-weight: 700 !important; border: none !important;
border-radius: 10px !important; padding: 0.6rem 2rem !important; }
.findings-box { background-color: #1e2132; border: 1px solid #353a52; border-left: 4px solid #7dd3fc;
border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem; margin-bottom: 1rem; color: #e2e8f0; }
.findings-box strong { color: #7dd3fc; font-size: 1.05rem; }
.footer-note { color: #64748b; font-size: 0.85rem; margin-top: 2rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📜 Termsight</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Know what you\'re agreeing to.</p>', unsafe_allow_html=True)
st.write(
    "Paste a privacy policy or terms of service — or a link to one — below. "
    "Termsight explains what it says in plain English, every finding linked "
    "back to the original text, and tells you when the document doesn't "
    "give a clear answer."
)

@st.cache_resource
def load_model():
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=dtype,
        device_map="auto"
    )

generator = load_model()

def is_url(text):
    return bool(re.match(r'^https?://', text.strip()))

def fetch_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)

        if len(clean_text) < 200:
            return None, "The page didn't contain enough readable text. Try pasting the policy text directly instead."
        return clean_text[:8000], None
    except Exception as e:
        return None, f"Couldn't fetch that URL ({str(e)}). Try pasting the policy text directly instead."

def split_into_sections(text, target_chunk_chars=600):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        paragraphs = sentences

    sections = {}
    current_chunk = ""
    section_num = 1
    for p in paragraphs:
        if len(current_chunk) + len(p) > target_chunk_chars and current_chunk:
            sections[section_num] = current_chunk.strip()
            section_num += 1
            current_chunk = p
        else:
            current_chunk += " " + p
    if current_chunk.strip():
        sections[section_num] = current_chunk.strip()

    return sections


QUESTIONS = {
    "What information is collected": "What personal information does this document say is collected?",
    "How it may be shared": "How does this document say information may be shared with third parties?",
    "How long it is retained": "How long does this document say information is retained?",
    "What rights you may be giving up": "What rights (access, deletion, opt-out, etc.) does this document say the user has or is giving up?",
}


def build_single_prompt(sections, question):
    numbered_text = "\n\n".join([f"[Section {i}] {text}" for i, text in sections.items()])
    return f"""Below is a document split into numbered sections.

{numbered_text}

Question: {question}

Write a clear, plain-English answer in 1-3 sentences, explaining what the document actually says. Then on a new line, cite the section number(s) your answer is based on, like: Source: [Section 4].

If the document does not clearly address this, respond with exactly: "Not specified in this document."
"""


def analyze_policy(text):
    sections = split_into_sections(text)
    results = {}
    for label, question in QUESTIONS.items():
        prompt = build_single_prompt(sections, question)
        result = generator([{"role": "user", "content": prompt}], max_new_tokens=200)
        answer = result[0]["generated_text"][-1]["content"]
        results[label] = answer
    return results, sections


st.markdown("**Paste a privacy policy, terms of service, or a URL to one:**")
policy_text = st.text_area(
    "", height=220, label_visibility="collapsed",
    placeholder="Paste text here, or paste a URL like https://example.com/privacy"
)

if st.button("🔍 Analyze"):
    if not policy_text.strip():
        st.warning("Please paste some text or a URL first.")
    else:
        input_text = policy_text.strip()

        if is_url(input_text):
            with st.spinner("Fetching page..."):
                fetched_text, error = fetch_text_from_url(input_text)
            if error:
                st.error(error)
                st.stop()
            input_text = fetched_text
            st.success("Fetched the page successfully. Analyzing...")

        with st.spinner("Analyzing (running 4 focused checks)..."):
            results, sections = analyze_policy(input_text)

        st.markdown("### 📋 Findings")
        for label, answer in results.items():
            st.markdown(
                f'<div class="findings-box"><strong>{label}</strong><br><br>{answer}</div>',
                unsafe_allow_html=True
            )

        with st.expander("📄 View original sections"):
            for num, text in sections.items():
                st.markdown(f"**Section {num}:** {text}")

st.markdown(
    '<p class="footer-note">🔒 Termsight\'s AI model runs locally — your text '
    'is never sent to a third-party AI service. When analyzing a URL, this '
    'app fetches that page directly before analysis.</p>',
    unsafe_allow_html=True
)
