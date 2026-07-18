"""
engine.py — Core logic for OmniSearch AI.

This module is deliberately separate from main.py (the Streamlit UI) so the
"knowledge engine" — ingestion, storage, retrieval, grounding, logging,
automation — can be reasoned about and tested independently of the UI.

Design principles baked in here (these are the answers to "how is this
different from just asking ChatGPT?"):

1. Knowledge Spaces persist and accumulate. Every uploaded file/URL is added
   to a named, disk-persisted FAISS index instead of being thrown away when
   the next file is uploaded. That's what makes it a *knowledge base* rather
   than a one-shot "chat with this one PDF" tool.

2. Grounded-only answers. The assistant is instructed to answer ONLY from
   retrieved context and to say so explicitly when it can't — it never
   silently falls back to its own general knowledge. That's the key
   difference from a generic chatbot: it's an org's single source of truth,
   not a general-purpose LLM with a PDF taped to it.

3. Every query is logged (with an answered/gap outcome), which powers the
   Analytics + Knowledge Gaps tab — an ops view no generic chatbot gives you.

4. Automation actions (Auto-FAQ generation, escalation queue) turn the tool
   from "a place to ask questions" into something that does repetitive work
   for a team, matching the "AI Automation & Intelligent Agents" theme.

5. Answer verification (verify_answer_facts + self_critique below) adds a
   second, independent check on top of every generated answer: a plain
   deterministic string-match pass for numbers/dates/quantities, plus one
   bounded LLM re-check of the answer against its own context. This is a
   modest, honest safety net — NOT causal reasoning, NOT autonomous
   multi-step correction, and NOT a solved general research problem. It
   catches a narrow, real class of grounding errors and says so plainly
   when it can't verify something, rather than claiming more than it does.
"""

import json
import os
import random
import re
import time
from datetime import datetime, timezone

import streamlit as st
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
    HarmBlockThreshold,
    HarmCategory,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic detection results
    _LANGDETECT_AVAILABLE = True
except ModuleNotFoundError:
    # Deployment hasn't picked up the new dependency yet. Language answering
    # still works via the manual dropdown; only auto-detect degrades to "en".
    _LANGDETECT_AVAILABLE = False

    class LangDetectException(Exception):
        pass

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

KB_ROOT = "kb_store"
LOG_DIR = "logs"
QUERY_LOG = os.path.join(LOG_DIR, "query_log.jsonl")
ESCALATION_LOG = os.path.join(LOG_DIR, "escalations.jsonl")

EMBEDDING_MODEL = "gemini-embedding-2-preview"
# NOTE: gemini-2.5-flash began returning early 404s for many accounts in
# July 2026 (ahead of its official Oct 16 2026 deprecation date — a known,
# widely-reported Google-side rollout issue). Pinned to the "-latest" alias
# instead of a specific dated model so this doesn't silently 404 again the
# next time Google retires a specific version.
LLM_MODEL = "gemini-flash-latest"

NOT_FOUND_TOKEN = "NOT_IN_KB"
GAP_MESSAGE = (
    "I couldn't find this in the current knowledge base. It may not have "
    "been added yet, or the documents that cover it haven't been uploaded "
    "to this space."
)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

PERSONAS = {
    "General": "You are a helpful, precise knowledge assistant.",
    "Education": (
        "You are a study assistant helping a student prepare for exams. "
        "Break concepts down clearly, highlight key facts and definitions, "
        "and keep answers exam-relevant."
    ),
    "Healthcare": (
        "You are supporting clinical or administrative staff. Be precise "
        "and literal about what the source material says. Always add a "
        "brief note that this is reference material, not clinical advice."
    ),
    "Human Resources": (
        "You are an internal HR knowledge assistant. Be warm and clear, "
        "and be exact about policy details such as dates, durations, and "
        "eligibility rules."
    ),
    "Legal": (
        "You are supporting a legal professional. Where relevant, reference "
        "the specific clause or section the information comes from, and "
        "add a brief disclaimer that this is not legal advice."
    ),
    "Manufacturing": (
        "You are a shop-floor troubleshooting assistant. Respond with "
        "clear, numbered steps a technician can follow quickly."
    ),
    "Sales": (
        "You are a sales enablement assistant. Be concise and lead with "
        "the exact figure, feature, or fact a rep needs during a live call."
    ),
    "Customer Support": (
        "You are a customer support assistant. Be friendly and concise, "
        "and end with a clear next step for the customer."
    ),
    "Retail": (
        "You are a retail operations assistant supporting store staff. "
        "Be concise and actionable."
    ),
}

# --------------------------------------------------------------------------
# Multi-language answering
# --------------------------------------------------------------------------
# Display name -> ISO 639-1 code (what langdetect returns) shown in the UI
# dropdown. "Auto-detect" (None) means: infer the language from the user's
# question itself, so a Hindi question gets a Hindi answer even if every
# indexed document is in English — Gemini's embeddings are cross-lingual,
# so retrieval still works, only the final generation step needs steering.
SUPPORTED_LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "Japanese": "ja",
    "Russian": "ru",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Urdu": "ur",
}
LANG_CODE_TO_NAME = {code: name for name, code in SUPPORTED_LANGUAGES.items() if code}
DEFAULT_LANG_CODE = "en"

# Localized copies of the "not found in this knowledge base" message, so the
# one sentence a user is most likely to see when the KB is incomplete still
# reads naturally in their language instead of silently falling back to
# English. Anything outside this set falls back to the LLM-authored answer
# language (which still works — this dict only covers the static string).
GAP_MESSAGE = (
    "I couldn't find this in the current knowledge base. It may not have "
    "been added yet, or the documents that cover it haven't been uploaded "
    "to this space."
)
GAP_MESSAGES = {
    "en": GAP_MESSAGE,
    "hi": "मुझे यह वर्तमान नॉलेज बेस में नहीं मिला। हो सकता है यह अभी तक जोड़ा न गया हो, या इससे जुड़े दस्तावेज़ इस स्पेस में अपलोड नहीं किए गए हों।",
    "es": "No pude encontrar esto en la base de conocimientos actual. Puede que aún no se haya añadido, o que los documentos correspondientes no se hayan subido a este espacio.",
    "fr": "Je n'ai pas trouvé cette information dans la base de connaissances actuelle. Elle n'a peut-être pas encore été ajoutée, ou les documents correspondants n'ont pas été chargés dans cet espace.",
    "de": "Ich konnte dies in der aktuellen Wissensdatenbank nicht finden. Möglicherweise wurde es noch nicht hinzugefügt, oder die entsprechenden Dokumente wurden noch nicht in diesen Bereich hochgeladen.",
    "pt": "Não encontrei essa informação na base de conhecimento atual. Ela pode ainda não ter sido adicionada, ou os documentos correspondentes não foram enviados a este espaço.",
    "ar": "لم أتمكن من العثور على هذا في قاعدة المعرفة الحالية. ربما لم تتم إضافته بعد، أو أن المستندات ذات الصلة لم يتم رفعها إلى هذا القسم.",
    "zh-cn": "我在当前知识库中找不到相关信息。可能尚未添加,或者相关文档还没有上传到这个空间。",
    "ja": "現在のナレッジベースにはこの情報が見つかりませんでした。まだ追加されていないか、関連する文書がこのスペースにアップロードされていない可能性があります。",
    "ru": "Я не нашёл эту информацию в текущей базе знаний. Возможно, она ещё не добавлена, либо соответствующие документы не загружены в это пространство.",
    "te": "ఇది ప్రస్తుత నాలెడ్జ్ బేస్‌లో నాకు కనిపించలేదు. ఇది ఇంకా జోడించబడి ఉండకపోవచ్చు, లేదా సంబంధిత పత్రాలు ఈ స్పేస్‌లో అప్‌లోడ్ చేయబడి ఉండకపోవచ్చు.",
    "ta": "இது தற்போதைய அறிவுத் தளத்தில் எனக்குக் கிடைக்கவில்லை. இது இன்னும் சேர்க்கப்படாமல் இருக்கலாம், அல்லது தொடர்புடைய ஆவணங்கள் இந்த இடத்தில் பதிவேற்றப்படாமல் இருக்கலாம்.",
    "bn": "আমি বর্তমান নলেজ বেসে এটি খুঁজে পাইনি। এটি হয়তো এখনও যোগ করা হয়নি, অথবা সংশ্লিষ্ট নথিগুলো এই স্পেসে আপলোড করা হয়নি।",
    "ur": "مجھے یہ موجودہ نالج بیس میں نہیں ملا۔ ہو سکتا ہے یہ ابھی شامل نہ کیا گیا ہو، یا متعلقہ دستاویزات اس اسپیس میں اپ لوڈ نہ کی گئی ہوں۔",
}


def detect_language(text: str) -> str:
    """Best-effort ISO 639-1 code for the language a question is written in.

    Falls back to English when the text is too short for reliable detection
    (langdetect is noisy under ~4 characters, e.g. "PTO?"), when the
    dependency isn't installed, or when detection throws for any reason —
    auto-detect should never be the thing that breaks a chat response.
    """
    if not _LANGDETECT_AVAILABLE or len(text.strip()) < 4:
        return DEFAULT_LANG_CODE
    try:
        code = detect(text)
    except LangDetectException:
        return DEFAULT_LANG_CODE
    # langdetect sometimes returns close variants (e.g. "hi" vs "mr" for
    # short Devanagari strings); only trust codes we actually support a
    # localized experience for, otherwise fall back to English rather than
    # silently answering in a language nobody selected.
    return code if code in LANG_CODE_TO_NAME else DEFAULT_LANG_CODE


def resolve_answer_language(query: str, selected_language: str) -> tuple[str, str]:
    """Returns (lang_code, lang_name) given the sidebar selection.

    selected_language is a key of SUPPORTED_LANGUAGES. "Auto-detect" defers
    to the question's own language; anything else pins the answer language
    regardless of what language the source documents or the question are in.
    """
    if selected_language == "Auto-detect" or not selected_language:
        code = detect_language(query)
    else:
        code = SUPPORTED_LANGUAGES.get(selected_language, DEFAULT_LANG_CODE)
    return code, LANG_CODE_TO_NAME.get(code, "English")


SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    return slug or "default"


def _clean(text: str) -> str:
    # Same critical fix as before: null bytes crash the Gemini API.
    return text.replace("\x00", "")


def call_with_retry(fn, *args, max_retries=4, base_delay=2, **kwargs):
    """Exponential backoff for Gemini rate-limit (429) errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - we re-raise non-rate-limit errors below
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "resourceexhausted" in msg or "rate" in msg or "quota" in msg:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
                continue
            raise
    raise last_err


@st.cache_resource(show_spinner=False)
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.0, safety_settings=SAFETY_SETTINGS)


# --------------------------------------------------------------------------
# Knowledge Space management (persisted FAISS indexes on disk)
# --------------------------------------------------------------------------

def _space_dir(space: str) -> str:
    return os.path.join(KB_ROOT, _slug(space))


def _meta_path(space: str) -> str:
    return os.path.join(_space_dir(space), "meta.json")


def list_spaces():
    os.makedirs(KB_ROOT, exist_ok=True)
    return sorted(d for d in os.listdir(KB_ROOT) if os.path.isdir(os.path.join(KB_ROOT, d)))


def load_meta(space: str) -> dict:
    p = _meta_path(space)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"display_name": space, "documents": [], "created": datetime.now(timezone.utc).isoformat()}


def save_meta(space: str, meta: dict):
    os.makedirs(_space_dir(space), exist_ok=True)
    with open(_meta_path(space), "w") as f:
        json.dump(meta, f, indent=2)


def load_vector_store(space: str, embeddings):
    d = _space_dir(space)
    if os.path.exists(os.path.join(d, "index.faiss")):
        return FAISS.load_local(d, embeddings, allow_dangerous_deserialization=True)
    return None


def save_vector_store(space: str, vs):
    d = _space_dir(space)
    os.makedirs(d, exist_ok=True)
    vs.save_local(d)


# --------------------------------------------------------------------------
# Loaders for each source type (kept lightweight & dependency-minimal —
# past deployment breakage came from heavy/unstable packages, so this
# intentionally avoids things like `unstructured` in favor of plain
# python-docx / BeautifulSoup)
# --------------------------------------------------------------------------

def load_pdf_docs(path: str, filename: str):
    docs = PyPDFLoader(path).load()
    for d in docs:
        d.page_content = _clean(d.page_content)
        d.metadata["source"] = filename
    return docs


def load_docx_docs(path: str, filename: str):
    try:
        from docx import Document as DocxReader
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The 'python-docx' package isn't installed on this deployment. "
            "Add `python-docx==1.1.2` to requirements.txt and reboot the app."
        ) from e
    reader = DocxReader(path)
    text = "\n".join(p.text for p in reader.paragraphs if p.text.strip())
    return [Document(page_content=_clean(text), metadata={"source": filename, "page": 0})]


def load_txt_docs(path: str, filename: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=_clean(text), metadata={"source": filename, "page": 0})]


def load_url_docs(url: str):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "The 'requests'/'beautifulsoup4' packages aren't installed on this "
            "deployment. Add `beautifulsoup4==4.12.3` and `requests==2.32.3` to "
            "requirements.txt and reboot the app."
        ) from e
    resp = requests.get(url, timeout=15, headers={"User-Agent": "OmniSearchAI/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n{2,}", "\n\n", soup.get_text(separator="\n")).strip()
    return [Document(page_content=_clean(text), metadata={"source": url, "page": 0})]


def ingest_documents(space: str, raw_docs: list, embeddings) -> int:
    """Splits raw_docs into chunks and merges them into the space's index.
    Returns the number of chunks added (0 if there was no extractable text —
    e.g. a scanned/image-only PDF)."""
    chunks = SPLITTER.split_documents(raw_docs)
    if not chunks:
        return 0
    new_vs = call_with_retry(FAISS.from_documents, chunks, embeddings)
    existing = load_vector_store(space, embeddings)
    if existing is not None:
        existing.merge_from(new_vs)
        save_vector_store(space, existing)
    else:
        save_vector_store(space, new_vs)
    return len(chunks)


# --------------------------------------------------------------------------
# Grounded retrieval + answering
# --------------------------------------------------------------------------

def _is_gap_response(raw: str) -> bool:
    cleaned = raw.strip().strip("*_` \n").upper()
    return cleaned.startswith(NOT_FOUND_TOKEN) or (NOT_FOUND_TOKEN in cleaned and len(cleaned) < 40)


# --------------------------------------------------------------------------
# Answer verification: a symbolic (deterministic) pass + a bounded neural
# (single LLM call) pass, run on top of every generated answer.
#
# What this IS: two independent, narrow checks that catch a real class of
# grounding errors — a number the model invented, or a claim the answer
# makes that its own retrieved context doesn't actually support.
#
# What this is NOT: causal/physical world modeling, autonomous multi-step
# self-correction, or continuous learning. Both checks are heuristic and
# bounded — one regex pass, one extra LLM call — and both fail open (they
# flag a caveat to the user rather than silently rewriting the answer or
# blocking it), because a heuristic verifier being wrong should never be
# more disruptive than the thing it's checking.
# --------------------------------------------------------------------------

# Matches quantities likely to be load-bearing facts: currency, percentages,
# numbers with a time/quantity unit, and bare numbers of 2+ digits (catches
# years, counts, dollar figures without a symbol, etc). Deliberately skips
# bare single digits, since those are usually list markers or prose ("one
# of the two options") rather than facts worth cross-checking.
_FACT_RE = re.compile(
    r"""
    \$\s?\d[\d,]*(?:\.\d+)?                                   # $75, $1,200.50
    | \d[\d,]*(?:\.\d+)?\s?%                                  # 18%, 4.5%
    | \d[\d,]*(?:\.\d+)?\s?(?:days?|months?|years?|hours?|minutes?|weeks?)  # 18 days, 6 months
    | \b\d{2,}\b                                              # 2026, 500, 30
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _normalize_fact(raw: str) -> str:
    s = raw.strip().lower().replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def verify_answer_facts(answer: str, source_docs: list) -> dict:
    """Deterministic, non-LLM check: does every number/percentage/quantity
    claim in the generated answer literally appear somewhere in the
    retrieved source text? Pure string matching — no model call, so this
    step cannot itself hallucinate.

    Caveat (stated plainly, not hidden): this only catches exact-token
    mismatches. If the model paraphrases "18 days" as "eighteen days" or
    computes a derived figure, this heuristic won't recognize the match and
    may over-flag. It's a supplementary signal on top of the grounded
    prompt, not a substitute for it.
    """
    context_text = " ".join(d.page_content for d in source_docs)
    context_facts = {_normalize_fact(m) for m in _FACT_RE.findall(context_text)}
    answer_facts = {_normalize_fact(m) for m in _FACT_RE.findall(answer)}
    verified = sorted(f for f in answer_facts if f in context_facts)
    unverified = sorted(f for f in answer_facts if f not in context_facts)
    return {"checked": bool(answer_facts), "verified": verified, "unverified": unverified}


def self_critique(llm, answer: str, source_docs: list, max_context_chars: int = 6000) -> dict:
    """One bounded verification pass: ask the model to check its own answer
    against the retrieved context. This is a single extra call, not a loop
    — it never re-plans, re-runs, or edits anything on its own. If the
    check itself fails for any reason (rate limit, transient API error), we
    fail open: the original answer still reaches the user, just without a
    verification badge, rather than blocking on a check that couldn't run.
    """
    context_text = "\n\n".join(d.page_content for d in source_docs)[:max_context_chars]
    prompt = (
        "You are a strict fact-checker reviewing an AI-generated answer against "
        "its source context. Check ONLY whether every factual claim in the "
        "answer is directly supported by the context below — do not comment on "
        "style, tone, completeness, or language.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Answer to check:\n{answer}\n\n"
        "If every claim in the answer is directly supported by the context, "
        "reply with exactly one word: SUPPORTED\n"
        "If any claim is not supported by the context, reply starting with "
        "UNSUPPORTED: followed by a short comma-separated list of the specific "
        "unsupported claims. Do not explain, do not add anything else."
    )
    try:
        resp = call_with_retry(llm.invoke, prompt)
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    except Exception:
        return {"checked": False, "passed": True, "issues": []}

    if text.upper().startswith("SUPPORTED"):
        return {"checked": True, "passed": True, "issues": []}
    issues_str = text.split(":", 1)[1].strip() if ":" in text else text
    issues = [i.strip() for i in issues_str.split(",") if i.strip()]
    return {"checked": True, "passed": False, "issues": issues}


def _build_prompt(persona: str, answer_lang_name: str) -> ChatPromptTemplate:
    persona_instruction = PERSONAS.get(persona, PERSONAS["General"])
    language_instruction = (
        f"Write your entire answer in {answer_lang_name}, even if the context "
        f"below is written in a different language and even if the question "
        f"was asked in a different language. Translate faithfully: do not add, "
        f"omit, or guess at information while translating. Keep proper nouns, "
        f"product names, and numbers/dates in their original form unless a "
        f"localized form is clearly more natural in {answer_lang_name}.\n\n"
    )
    system_prompt = (
        f"{persona_instruction}\n\n"
        f"{language_instruction}"
        "Answer the user's question using ONLY the context below, which comes "
        "from the organization's own knowledge base. Do not use any outside "
        "knowledge, even if you happen to know the answer generally — this "
        "assistant must only ever speak from what has actually been uploaded.\n\n"
        f"If, and only if, the context genuinely does not contain enough "
        f"information to answer, reply with exactly this token and nothing "
        f"else (do not translate the token itself): {NOT_FOUND_TOKEN}\n\n"
        "Context:\n{context}"
    )
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])


def answer_question(
    vs, query: str, persona: str, llm, answer_language: str = "Auto-detect",
    k: int = 4, verify: bool = True,
) -> dict:
    lang_code, lang_name = resolve_answer_language(query, answer_language)
    localized_gap = GAP_MESSAGES.get(lang_code, GAP_MESSAGE)

    docs = vs.similarity_search(query, k=k)
    if not docs:
        return {
            "answer": localized_gap, "sources": [], "pages": [], "status": "gap",
            "language": lang_name, "language_code": lang_code,
        }

    chain = create_stuff_documents_chain(llm, _build_prompt(persona, lang_name))
    raw = call_with_retry(chain.invoke, {"input": query, "context": docs})

    if _is_gap_response(raw):
        return {
            "answer": localized_gap, "sources": [], "pages": [], "status": "gap",
            "language": lang_name, "language_code": lang_code,
        }

    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    pages = sorted({(d.metadata.get("page", 0) or 0) + 1 for d in docs})
    result = {
        "answer": raw, "sources": sources, "pages": pages, "status": "answered",
        "language": lang_name, "language_code": lang_code,
    }

    if verify:
        result["fact_check"] = verify_answer_facts(raw, docs)
        result["critique"] = self_critique(llm, raw, docs)
    else:
        result["fact_check"] = {"checked": False, "verified": [], "unverified": []}
        result["critique"] = {"checked": False, "passed": True, "issues": []}

    return result


# --------------------------------------------------------------------------
# Logging, analytics, escalation queue
# --------------------------------------------------------------------------

def _append_jsonl(path: str, entry: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def log_interaction(
    space: str, persona: str, query: str, status: str, sources: list,
    language: str = "English", verification_flagged: bool = False,
):
    _append_jsonl(QUERY_LOG, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "space": space,
        "persona": persona,
        "query": query,
        "status": status,
        "sources": sources,
        "language": language,
        "verification_flagged": verification_flagged,
    })


def log_escalation(space: str, persona: str, query: str):
    _append_jsonl(ESCALATION_LOG, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "space": space,
        "persona": persona,
        "query": query,
        "resolved": False,
    })


def get_analytics(space: str) -> dict:
    rows = [r for r in read_jsonl(QUERY_LOG) if r.get("space") == space]
    total = len(rows)
    answered = sum(1 for r in rows if r.get("status") == "answered")
    gaps = [r for r in rows if r.get("status") == "gap"]
    flagged = sum(1 for r in rows if r.get("verification_flagged"))
    by_persona: dict = {}
    by_language: dict = {}
    for r in rows:
        p = r.get("persona", "General")
        by_persona[p] = by_persona.get(p, 0) + 1
        lang = r.get("language", "English")
        by_language[lang] = by_language.get(lang, 0) + 1
    return {
        "total_queries": total,
        "answered": answered,
        "gap_count": len(gaps),
        "verification_flagged": flagged,
        "resolution_rate": round(100 * answered / total, 1) if total else 0.0,
        "gaps": list(reversed(gaps))[:25],
        "by_persona": by_persona,
        "by_language": by_language,
    }


def get_escalations(space: str) -> list:
    rows = [r for r in read_jsonl(ESCALATION_LOG) if r.get("space") == space]
    return list(reversed(rows))


# --------------------------------------------------------------------------
# Automation: Auto-FAQ generator
# --------------------------------------------------------------------------

def generate_faq(vs, llm, space: str, language: str = "English", max_chunks: int = 40):
    all_docs = list(vs.docstore._dict.values())  # noqa: SLF001 - no public "list all" API on FAISS wrapper
    if not all_docs:
        return None
    sample = all_docs[:max_chunks]
    context_text = "\n\n---\n\n".join(d.page_content[:800] for d in sample)
    lang_name = language if language and language != "Auto-detect" else "English"
    prompt = (
        f"You are creating an FAQ document from the knowledge base content below, "
        f"for a knowledge space called \"{space}\". Identify the 8-12 most useful "
        "questions someone would realistically ask about this content, and answer "
        "each one concisely and accurately using ONLY the content given. Write both "
        f"the questions and answers in {lang_name}, even though the source content "
        "below may be in a different language — translate faithfully without adding "
        "or omitting information. Format the output as markdown, using '### Q: ...' "
        "followed by 'A: ...' for each pair.\n\n"
        f"Content:\n{context_text}"
    )
    resp = call_with_retry(llm.invoke, prompt)
    return resp.content
