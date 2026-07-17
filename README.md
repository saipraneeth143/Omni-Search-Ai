# 🧠 OmniSearch AI — Unified Knowledge Assistant

**Theme:** AI Automation & Intelligent Agents
**Problem statement:** AI Knowledge Assistant — students, employees, and customers struggle to find accurate information quickly because knowledge is scattered across fragmented sources.

## 🙋 "Isn't this just ChatGPT with a PDF?" — No, and here's the actual difference

That was the judges' feedback on the first version, and it was fair: v1 let you upload *one* PDF, chat with it for that session, and if there was no PDF it just answered from Gemini's general knowledge — which is exactly what any chatbot does. Nothing about it was a real "knowledge assistant."

This version is a different kind of tool:

| | Generic chatbot (ChatGPT/Claude/etc.) | OmniSearch AI |
|---|---|---|
| **Knowledge** | Whatever the model was trained on, or one file per chat | A named, **persistent Knowledge Space** that accumulates PDFs, DOCX, TXT, and web pages over time |
| **When it doesn't know** | Guesses from general training knowledge | Explicitly says "not in the knowledge base," refuses to guess, and logs it |
| **Visibility for admins** | None | Analytics tab: query volume, resolution rate, and a live list of **knowledge gaps** — exactly what content is missing |
| **Escalation** | None | One-click "flag for a human expert" queue |
| **Repetitive work** | You ask, it answers, forever, one question at a time | One-click **Auto-FAQ generator** drafts a whole FAQ document from the knowledge base |
| **Audience adaptation** | Same tone for everyone | Persona modes for Education, Healthcare, HR, Legal, Manufacturing, Sales, Customer Support, Retail |

The core design decision is **grounded-only answering**: the assistant is instructed to answer *only* from retrieved, indexed content and to say so explicitly when it can't — never silently falling back to open-domain knowledge the way a general chatbot would. For regulated or high-stakes use cases (HR, Legal, Healthcare), that's the difference between "a chatbot that might hallucinate a policy" and "a system you can actually trust as a source of truth."

## ✨ What's actually implemented (not aspirational)

- **Multi-source ingestion**: PDF, DOCX, TXT, and web page URLs, all into the same knowledge space.
- **Persistent, named Knowledge Spaces**: create one per team/course/product; documents accumulate instead of being replaced on every upload.
- **Grounded, cited answers**: every answer lists the exact source file(s) and page(s) it came from.
- **Refuses to hallucinate**: if the knowledge base doesn't cover a question, it says so instead of answering from general knowledge.
- **Industry/persona modes**: 9 modes tailored to the theme's target industries, changing tone and emphasis.
- **Analytics & Knowledge Gaps dashboard**: query volume, resolution rate, and a running list of unanswered questions — a real operational tool, not just a chat log.
- **Escalation queue**: flag any unanswered question for human follow-up.
- **Auto-FAQ generator**: one click drafts a downloadable FAQ document from everything in a knowledge space.
- **Rate-limit resilient**: retry/backoff wrapper around all Gemini API calls.

## ⚙️ Tech stack (as actually deployed)

- **UI + orchestration**: Streamlit
- **LLM + embeddings**: Google Gemini (`gemini-2.5-flash`, `gemini-embedding-2-preview`) via `langchain-google-genai`
- **Vector store**: FAISS, persisted to disk per knowledge space
- **Document parsing**: `pypdf`, `python-docx`, `BeautifulSoup`
- **Hosting**: Streamlit Community Cloud

## 🏢 Cross-industry fit

- 🎓 **Education** — students query lecture slides, syllabi, and papers for cited, exam-ready answers.
- 🤝 **HR** — employees ask policy questions and get exact clauses, not guesses.
- ⚖️ **Legal** — search across contracts/case files without reading everything manually.
- 🏭 **Manufacturing** — technicians get step-by-step answers from equipment manuals.
- 🏥 **Healthcare** — staff pull specific guidelines from reference material (not a substitute for clinical judgment).
- 🛍️ **Customer Support / Retail** — agents get instant, cited answers from policy/product docs.
- 📈 **Sales** — reps get exact figures during a live call.

## 🚀 Running locally

```bash
cd backend
pip install -r ../requirements.txt
streamlit run main.py
```

Add your key to `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_key_here"
```

## 🛣️ Honest roadmap (not yet built)

- Connectors for Notion/Confluence/Google Drive (currently: PDF/DOCX/TXT/URL only)
- Role-based access control per document
- Multi-user shared knowledge spaces with real accounts (current persistence is per-deployment disk storage, which is fine for a demo/small team but not a substitute for a hosted database at scale)

## 📄 License

MIT — see `LICENSE`.
