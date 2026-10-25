from flask import Flask, request, render_template_string
from transformers import pipeline
import torch
import re
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

print("Loading model...")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=dtype,
    device_map="auto"
)
print("Model loaded.")

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
            return None, "The page didn't contain enough readable text."
        return clean_text[:8000], None
    except Exception as e:
        return None, f"Couldn't fetch that URL ({str(e)})."

def split_into_sections(text, target_chunk_chars=600):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = re.split(r'(?<=[.!?])\s+', text.strip())
    sections, chunk, num = {}, "", 1
    for p in paragraphs:
        if len(chunk) + len(p) > target_chunk_chars and chunk:
            sections[num] = chunk.strip()
            num += 1
            chunk = p
        else:
            chunk += " " + p
    if chunk.strip():
        sections[num] = chunk.strip()
    return sections

QUESTIONS = {
    "What information is collected": "What personal information does this document say is collected?",
    "How it may be shared": "How does this document say information may be shared with third parties?",
    "How long it is retained": "How long does this document say information is retained?",
    "What rights you may be giving up": "What rights (access, deletion, opt-out, etc.) does this document say the user has?",
}

def build_prompt(sections, question):
    numbered = "\n\n".join([f"[Section {i}] {t}" for i, t in sections.items()])
    return f"""Below is a document split into numbered sections.

{numbered}

Question: {question}

Write a clear, plain-English answer in 1-3 sentences. Then cite the section number(s), like Source: [Section 4]. If not addressed, respond exactly: "Not specified in this document."
"""

def analyze_policy(text):
    sections = split_into_sections(text)
    results = {}
    for label, q in QUESTIONS.items():
        prompt = build_prompt(sections, q)
        out = generator([{"role": "user", "content": prompt}], max_new_tokens=200)
        results[label] = out[0]["generated_text"][-1]["content"]
    return results, sections

PAGE = """
<!DOCTYPE html>
<html><head><title>Termsight</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { background: linear-gradient(180deg, #0f1117 0%, #1a1d29 100%); color:#e2e8f0; font-family: system-ui, sans-serif; padding:40px 20px; max-width:750px; margin:0 auto; min-height:100vh; }
  h1 { font-size:3rem; background: linear-gradient(90deg,#7dd3fc,#a78bfa); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800; margin-bottom:0; }
  p.subtitle { color:#94a3b8; font-size:1.1rem; margin-top:4px; }
  label { font-weight:600; display:block; margin:20px 0 8px; }
  textarea { width:100%; background-color:#1e2132; border:1px solid #353a52; border-radius:12px; color:#e2e8f0; padding:14px; font-size:0.95rem; box-sizing:border-box; font-family:inherit; }
  button { background: linear-gradient(90deg,#7dd3fc,#a78bfa); color:#0f1117; font-weight:700; border:none; border-radius:10px; padding:12px 32px; font-size:1rem; cursor:pointer; margin-top:14px; }
  button:hover { opacity:0.9; }
  .findings-box { background-color:#1e2132; border:1px solid #353a52; border-left:4px solid #7dd3fc; border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem; }
  .findings-box strong { color:#7dd3fc; font-size:1.05rem; }
  .error { color:#f87171; background:#2a1718; border:1px solid #7f1d1d; border-radius:10px; padding:12px 16px; margin-top:14px; }
  .footer-note { color:#64748b; font-size:0.85rem; margin-top:2.5rem; text-align:center; }
  h2 { margin-top:2rem; }
</style></head>
<body>
<h1>📜 Termsight</h1>
<p class="subtitle">Know what you're agreeing to.</p>
<p>Paste a privacy policy or terms of service — or a link to one — below. Termsight explains what it says in plain English, every finding linked back to the original text.</p>

<form method="POST">
<label>Paste a privacy policy, terms of service, or a URL to one:</label>
<textarea name="policy_text" rows="8" placeholder="Paste text here, or paste a URL like https://example.com/privacy">{{ input_text or '' }}</textarea><br>
<button type="submit">🔍 Analyze</button>
</form>

{% if error %}<div class="error">{{ error }}</div>{% endif %}

{% if results %}
<h2>📋 Findings</h2>
{% for label, answer in results.items() %}
<div class="findings-box"><strong>{{ label }}</strong><br><br>{{ answer }}</div>
{% endfor %}
{% endif %}

<p class="footer-note">🔒 Termsight's AI model runs locally — your text is never sent to a third-party AI service.</p>
</body></html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    results, error, input_text = None, None, ""
    if request.method == "POST":
        input_text = request.form.get("policy_text", "").strip()
        if not input_text:
            error = "Please paste some text or a URL."
        else:
            text = input_text
            if is_url(text):
                fetched, err = fetch_text_from_url(text)
                if err:
                    error = err
                else:
                    text = fetched
            if not error:
                results, _ = analyze_policy(text)
    return render_template_string(PAGE, results=results, error=error, input_text=input_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
