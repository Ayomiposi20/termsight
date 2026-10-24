# Termsight

Know what you're agreeing to.

Termsight is an open-source AI tool that helps people understand privacy
policies and terms of service without reading pages of legal language.

Paste a policy or a link to one and Termsight explains, in plain
English, what information is collected, how it may be shared, how long
it's retained, and what rights you may be giving up. Every finding cites
the exact section it came from, and if the document doesn't give a clear
answer, Termsight says so instead of guessing.

## Why open-weight AI matters here

Privacy policies and terms of service are sensitive by nature pasting
one into a cloud AI service means trusting yet another third party with
that text. Termsight runs on an open-weight model
([Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct))
rather than a commercial AI API, so the document you're analyzing is
never sent to a third-party AI service.

## How to run it

`bash
pip install -r requirements.txt
streamlit run app.py
