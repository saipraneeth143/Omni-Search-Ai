import os
from datetime import datetime, timezone

import streamlit as st

from engine import (
    PERSONAS,
    SUPPORTED_LANGUAGES,
    _slug,
    generate_faq,
    get_analytics,
    get_embeddings,
    get_escalations,
    get_llm,
    list_spaces,
    load_meta,
    load_url_docs,
    load_vector_store,
    log_escalation,
    log_interaction,
    load_pdf_docs,
    load_docx_docs,
    load_txt_docs,
    ingest_documents,
    answer_question,
    save_meta,
)

DEMO_SPACE_NAME = "🎬 Demo — Acme HR Handbook"
DEMO_DOC_NAME = "acme_hr_handbook_sample.txt"
DEMO_DOC_TEXT = """Acme Corp — Employee Handbook (Sample)

1. Paid Time Off (PTO)
Full-time employees accrue 18 days of PTO per year, credited monthly at 1.5 days.
PTO requests must be submitted at least 5 business days in advance through the HR
portal. Unused PTO up to 5 days may be carried over into the next calendar year;
anything beyond that is forfeited unless local law requires otherwise.

2. Remote Work Policy
Employees may work remotely up to 3 days per week with manager approval. Fully
remote arrangements require VP-level sign-off and are reviewed every 6 months.
Remote employees must be reachable during core hours, 10am-4pm in their local
time zone, and must use company-issued VPN software when accessing internal tools.

3. Expense Reimbursement
Business expenses under $75 can be submitted with a receipt via the Expenses app
and are typically reimbursed within 5 business days. Expenses over $75 require
pre-approval from a manager. Travel expenses (flights, hotels) must be booked
through the corporate travel portal to qualify for reimbursement.

4. Information Security
All laptops must have disk encryption and endpoint protection enabled before
connecting to company systems. Employees must complete security awareness
training annually. Any suspected phishing email should be reported using the
"Report Phishing" button in the email client, not simply deleted.

5. Onboarding
New hires complete a 2-week onboarding program covering company tools, security
training, and team introductions. A dedicated onboarding buddy is assigned for
the first 30 days. IT equipment is shipped to arrive at least 2 business days
before the employee's start date.

6. Benefits
Acme offers medical, dental, and vision coverage starting day one of employment.
The company matches 401(k) contributions up to 4% of salary. Employees also
receive an annual $500 wellness stipend that can be used for gym memberships,
fitness equipment, or mental health apps.
"""

# ============================================================================
# MULTI-LANGUAGE SUPPORT
#
# engine.py already implements this properly: SUPPORTED_LANGUAGES (imported
# above), auto-detection of the question's language, and localized answers
# generated in a single grounded LLM call (retrieval stays anchored to the
# original document language/embeddings; only the generation step is
# steered to the target language) — so no separate translation pass is
# needed here.
# ============================================================================


def get_retrieved_evidence(vs, query: str, k: int = 4):
    """Best-effort fetch of the raw passages behind an answer, for the
    transparency panel. Never raises — returns [] if unsupported, so a
    missing vector-store method can never break the chat response.
    Returns a list of (text, metadata, score_or_None)."""
    if vs is None:
        return []
    try:
        if hasattr(vs, "similarity_search_with_score"):
            results = vs.similarity_search_with_score(query, k=k)
            return [(doc.page_content, doc.metadata, score) for doc, score in results]
        if hasattr(vs, "similarity_search"):
            docs = vs.similarity_search(query, k=k)
            return [(doc.page_content, doc.metadata, None) for doc in docs]
    except Exception:
        pass
    return []

st.set_page_config(
    page_title="OmniSearch AI",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# ============================================================================
# THEME / CSS  — pure CSS, no extra pip packages, safe for existing deploys
# ============================================================================
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Inter, sans-serif;
        }

        :root {
            --os-primary: #818CF8;
            --os-primary-dark: #6366F1;
            --os-accent: #22D3EE;
            --os-success: #34D399;
            --os-warning: #FBBF24;
            --os-danger: #F87171;
            --os-bg-soft: rgba(255,255,255,0.04);
            --os-border: rgba(255,255,255,0.10);
            --os-text: #E7EAF6;
            --os-text-dim: #9AA3C0;
        }

        /* ---------- Dynamic cosmic background ---------- */
        .stApp {
            background: radial-gradient(ellipse 80% 60% at 15% -10%, rgba(99,102,241,0.30), transparent 60%),
                        radial-gradient(ellipse 70% 60% at 100% 0%, rgba(34,211,238,0.18), transparent 55%),
                        radial-gradient(ellipse 60% 50% at 50% 100%, rgba(129,140,248,0.14), transparent 60%),
                        linear-gradient(180deg, #06070F 0%, #0A0C1B 45%, #0B0E1E 100%);
            background-attachment: fixed;
        }
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stHeader"] { background: rgba(6,7,15,0.4); }

        /* Hide default Streamlit chrome clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---------- Global text contrast fix ----------
           Forces readable light text everywhere in the main app body,
           regardless of the viewer's light/dark Streamlit theme setting. */
        .main .block-container, .main .block-container p, .main .block-container span,
        .main .block-container li, .main .block-container label,
        .stMarkdown, .stMarkdown p, .stCaption, [data-testid="stCaptionContainer"],
        div[data-testid="stExpander"] summary, div[data-testid="stExpander"] p,
        div[data-testid="stExpander"] span,
        div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] span,
        div[data-testid="stChatMessage"] li,
        div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
            color: var(--os-text) !important;
        }
        .main .block-container .stCaption, .main .block-container small,
        div[data-testid="stExpander"] svg { color: var(--os-text-dim) !important; }
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--os-border);
            border-radius: 14px;
            backdrop-filter: blur(10px);
        }
        div[data-testid="stChatMessage"] { color: var(--os-text) !important; }

        /* ---------- Hero header ---------- */
        .os-hero {
            background:
                radial-gradient(circle at 15% 20%, rgba(129,140,248,0.35), transparent 45%),
                radial-gradient(circle at 90% 15%, rgba(34,211,238,0.30), transparent 50%),
                linear-gradient(120deg, rgba(30,27,75,0.85) 0%, rgba(49,46,129,0.75) 55%, rgba(8,145,178,0.55) 100%);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 22px;
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 20px 60px -20px rgba(79, 70, 229, 0.55), inset 0 1px 0 rgba(255,255,255,0.08);
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(18px);
            background-size: 200% 200%;
            animation: os-hero-shift 12s ease-in-out infinite;
        }
        @keyframes os-hero-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        .os-hero::after {
            content: "";
            position: absolute;
            top: -80px; right: -80px;
            width: 280px; height: 280px;
            background: radial-gradient(circle, rgba(34,211,238,0.28) 0%, rgba(34,211,238,0) 70%);
            border-radius: 50%;
            animation: os-pulse 6s ease-in-out infinite;
        }
        @keyframes os-pulse {
            0%, 100% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.15); opacity: 1; }
        }
        .os-hero h1 {
            color: #FFFFFF !important;
            font-size: 2.15rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 24px rgba(99,102,241,0.5);
        }
        .os-hero p {
            color: rgba(231,234,246,0.92) !important;
            font-size: 1.02rem;
            margin: 0;
            max-width: 700px;
        }
        .os-badges {
            margin-top: 1.1rem;
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
        }
        .os-badge {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.22);
            color: #F1F3FC !important;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            backdrop-filter: blur(6px);
            box-shadow: 0 0 0 rgba(34,211,238,0);
            transition: box-shadow 0.25s ease, transform 0.2s ease;
        }
        .os-badge:hover {
            box-shadow: 0 0 18px rgba(34,211,238,0.45);
            transform: translateY(-1px);
        }

        /* ---------- KPI cards ---------- */
        .os-kpi {
            background: rgba(255,255,255,0.045);
            border: 1px solid var(--os-border);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 8px 24px -12px rgba(0,0,0,0.5);
            border-left: 4px solid var(--os-primary);
            backdrop-filter: blur(14px);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            height: 100%;
        }
        .os-kpi:hover {
            transform: translateY(-3px);
            box-shadow: 0 0 22px -4px var(--os-glow, rgba(129,140,248,0.5));
        }
        .os-kpi .os-kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--os-text-dim);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .os-kpi .os-kpi-value {
            font-size: 1.95rem;
            font-weight: 800;
            color: #FFFFFF;
            margin-top: 0.15rem;
        }
        .os-kpi .os-kpi-icon { font-size: 1.3rem; }

        /* How-it-works steps */
        .os-step {
            background: rgba(255,255,255,0.045);
            border: 1px solid var(--os-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            height: 100%;
            position: relative;
            backdrop-filter: blur(14px);
            transition: transform 0.18s ease, border-color 0.18s ease;
        }
        .os-step:hover {
            transform: translateY(-2px);
            border-color: rgba(129,140,248,0.5);
        }
        .os-step .os-step-num {
            position: absolute;
            top: 0.7rem; right: 0.9rem;
            font-size: 0.72rem;
            font-weight: 700;
            color: #6C74A8;
        }
        .os-step .os-step-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
        .os-step .os-step-title {
            font-weight: 700;
            font-size: 0.92rem;
            color: #FFFFFF;
            margin-bottom: 0.2rem;
        }
        .os-step .os-step-desc {
            font-size: 0.82rem;
            color: var(--os-text-dim);
            line-height: 1.35;
        }

        /* ---------- Section cards ---------- */
        .os-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--os-border);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 8px 24px -14px rgba(0,0,0,0.6);
            margin-bottom: 1rem;
            backdrop-filter: blur(14px);
        }
        .os-section-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #FFFFFF;
            margin-bottom: 0.3rem;
        }

        /* Gap / escalation pills */
        .os-pill-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0.85rem;
            border-radius: 10px;
            background: rgba(255,255,255,0.035);
            margin-bottom: 0.45rem;
            border: 1px solid var(--os-border);
            color: var(--os-text);
        }
        .os-pill-row b { color: #FFFFFF; }
        .os-pill-status-open {
            background: rgba(248,113,113,0.16); color: #FCA5A5;
            padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
            border: 1px solid rgba(248,113,113,0.35);
        }
        .os-pill-status-resolved {
            background: rgba(52,211,153,0.16); color: #6EE7B7;
            padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
            border: 1px solid rgba(52,211,153,0.35);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(190deg, rgba(10,12,27,0.97) 0%, rgba(30,27,75,0.94) 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(18px);
        }
        section[data-testid="stSidebar"] * {
            color: #E7EAF6 !important;
        }
        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
            color: #9AA3C0 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.10);
        }
        section[data-testid="stSidebar"] .stButton>button {
            background: linear-gradient(120deg, #6366F1, #22D3EE);
            color: #06070F !important;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            box-shadow: 0 6px 18px -6px rgba(99,102,241,0.6);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        section[data-testid="stSidebar"] .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 26px -6px rgba(34,211,238,0.55);
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea {
            background: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.16) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderDropzone"] {
            background: rgba(255,255,255,0.04);
            border: 1px dashed rgba(255,255,255,0.25);
            border-radius: 12px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            border-bottom: 1px solid var(--os-border);
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            border-radius: 10px 10px 0 0;
            padding: 0 1.1rem;
            font-weight: 600;
            color: var(--os-text-dim) !important;
        }
        .stTabs [data-baseweb="tab"] p { color: inherit !important; }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(129,140,248,0.16), rgba(129,140,248,0.0));
            color: #FFFFFF !important;
            border-bottom: 3px solid var(--os-primary);
        }
        .stTabs [aria-selected="true"] p { color: #FFFFFF !important; }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.5rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--os-border);
            backdrop-filter: blur(14px);
            box-shadow: 0 6px 18px -12px rgba(0,0,0,0.6);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            background: linear-gradient(120deg, rgba(99,102,241,0.16), rgba(99,102,241,0.04));
            border-color: rgba(129,140,248,0.35);
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
            background: linear-gradient(120deg, rgba(34,211,238,0.12), rgba(34,211,238,0.03));
            border-color: rgba(34,211,238,0.30);
        }
        [data-testid="stChatInput"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            backdrop-filter: blur(14px);
        }
        [data-testid="stChatInput"] textarea { color: #FFFFFF !important; }

        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
            background: rgba(255,255,255,0.06);
            color: #FFFFFF;
            border: 1px solid rgba(255,255,255,0.16);
            transition: all 0.15s ease;
        }
        .stButton>button:hover {
            border-color: rgba(129,140,248,0.6);
            box-shadow: 0 0 16px rgba(129,140,248,0.35);
        }
        .stButton>button[kind="primary"] {
            background: linear-gradient(120deg, #6366F1, #22D3EE);
            color: #06070F;
            border: none;
        }
        .stDownloadButton>button {
            border-radius: 10px;
            font-weight: 700;
            background: linear-gradient(120deg, #34D399, #22D3EE);
            color: #06070F;
            border: none;
            box-shadow: 0 6px 18px -6px rgba(52,211,153,0.5);
        }

        /* Metrics fallback (native st.metric, unused but styled just in case) */
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--os-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            backdrop-filter: blur(14px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Read the Groq API key from Streamlit secrets ---
# Embeddings need no key at all (they run locally); only the chat model
# (Groq) needs one. Get a free key at https://console.groq.com/keys
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error(
        "Please add your GROQ_API_KEY to Streamlit secrets "
        "(Settings → Secrets, in your Streamlit Cloud app dashboard)."
    )
    st.stop()

embeddings = get_embeddings()
llm = get_llm()

# ============================== HERO HEADER ================================
st.markdown(
    """
    <div class="os-hero">
        <h1>🔍 OmniSearch AI</h1>
        <p>A grounded, multi-source knowledge assistant — not a general chatbot.
        It only answers from the documents your organization has added, and says
        so explicitly when it can't.</p>
        <div class="os-badges">
            <span class="os-badge">🧠 Retrieval-Grounded</span>
            <span class="os-badge">🚫 No Hallucinated Answers</span>
            <span class="os-badge">📊 Live Analytics</span>
            <span class="os-badge">⚡ Auto-Generated FAQs</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================== HOW IT WORKS ================================
with st.expander("ℹ️  How OmniSearch AI works", expanded=False):
    steps = [
        ("📥", "1. Upload", "Add PDFs, DOCX, TXT files, or a web page URL to a named Knowledge Space."),
        ("🧩", "2. Chunk & Embed", "Documents are split into passages and embedded into a searchable vector index."),
        ("🔎", "3. Retrieve", "Your question is matched against only the most relevant passages — nothing else."),
        ("✅", "4. Grounded Answer", "The model answers strictly from retrieved context, or flags a knowledge gap instead of guessing."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(
            f"""
            <div class="os-step">
                <div class="os-step-icon">{icon}</div>
                <div class="os-step-title">{title}</div>
                <div class="os-step-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================== SIDEBAR ==================================
with st.sidebar:
    # Apply a pending space switch (e.g. from the "instant demo" button) before
    # the selectbox widget below is instantiated.
    if "_pending_space_switch" in st.session_state:
        st.session_state["space_selector"] = st.session_state.pop("_pending_space_switch")

    st.markdown("### 🎬 New here?")
    st.caption("No files handy? Load a ready-made sample HR handbook and start asking questions instantly.")
    if st.button("✨ Load instant demo space", use_container_width=True):
        demo_meta = load_meta(DEMO_SPACE_NAME)
        existing_names = {d["name"] for d in demo_meta["documents"]}
        if DEMO_DOC_NAME not in existing_names:
            temp_path = f"temp_{_slug(DEMO_DOC_NAME)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(DEMO_DOC_TEXT)
            try:
                docs = load_txt_docs(temp_path, DEMO_DOC_NAME)
                n = ingest_documents(DEMO_SPACE_NAME, docs, embeddings)
                if n:
                    demo_meta["documents"].append({
                        "name": DEMO_DOC_NAME,
                        "chunks": n,
                        "added": datetime.now(timezone.utc).isoformat(),
                    })
                    save_meta(DEMO_SPACE_NAME, demo_meta)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        st.session_state["_pending_space_switch"] = DEMO_SPACE_NAME
        st.rerun()

    st.divider()
    st.markdown("### 🗂️ Knowledge Space")
    st.caption("A named, persistent knowledge base — e.g. 'HR Policies' or 'Course 101'.")

    spaces = list_spaces()
    create_new_label = "➕ Create new space..."
    options = spaces + [create_new_label]
    default_idx = 0 if spaces else 0
    choice = st.selectbox("Active space", options, index=default_idx, key="space_selector")

    active_space = None
    if choice == create_new_label:
        new_name = st.text_input("Name your new space", placeholder="e.g. HR Policies 2026")
        if new_name.strip():
            active_space = new_name.strip()
            save_meta(active_space, load_meta(active_space))
    else:
        active_space = choice

    st.divider()
    st.markdown("### 🎭 Assistant Mode")
    persona = st.selectbox("Industry / persona", list(PERSONAS.keys()))
    st.caption(PERSONAS[persona])

    st.divider()
    st.markdown("### 🌐 Answer Language")
    answer_language = st.selectbox(
        "Respond in", list(SUPPORTED_LANGUAGES.keys()), index=0, key="answer_language"
    )
    st.caption(
        "Auto-detect answers in whatever language the question was asked in. "
        "Retrieval always stays grounded in the original source documents — "
        "only the generated answer changes language, so accuracy doesn't get "
        "diluted by the language choice."
    )

    st.divider()
    st.markdown("### 📥 Add Knowledge")
    if not active_space:
        st.info("Create or select a space above first.")
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF / DOCX / TXT (multiple allowed)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        url_col1, url_col2 = st.columns([3, 1])
        url_input = url_col1.text_input("Or add a web page URL", label_visibility="collapsed",
                                         placeholder="https://example.com/handbook")
        add_url = url_col2.button("Add", use_container_width=True)

        meta = load_meta(active_space)
        existing_names = {d["name"] for d in meta["documents"]}

        if uploaded_files:
            new_files = [f for f in uploaded_files if f.name not in existing_names]
            skipped = [f.name for f in uploaded_files if f.name in existing_names]
            if skipped:
                st.caption(f"Already indexed, skipped: {', '.join(skipped)}")
            if new_files:
                with st.spinner(f"Indexing {len(new_files)} file(s)... (may take a moment on the free tier)"):
                    total_chunks = 0
                    for f in new_files:
                        temp_path = f"temp_{_slug(f.name)}"
                        with open(temp_path, "wb") as out:
                            out.write(f.getvalue())
                        try:
                            if f.name.lower().endswith(".pdf"):
                                docs = load_pdf_docs(temp_path, f.name)
                            elif f.name.lower().endswith(".docx"):
                                docs = load_docx_docs(temp_path, f.name)
                            else:
                                docs = load_txt_docs(temp_path, f.name)

                            n = ingest_documents(active_space, docs, embeddings)
                            if n == 0:
                                st.warning(
                                    f"⚠️ No readable text found in **{f.name}**. It may be a "
                                    "scanned/image-only document — try running it through OCR first."
                                )
                            else:
                                total_chunks += n
                                meta["documents"].append({
                                    "name": f.name,
                                    "chunks": n,
                                    "added": datetime.now(timezone.utc).isoformat(),
                                })
                        except Exception as e:
                            st.error(f"Failed to index {f.name}: {e}")
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    save_meta(active_space, meta)
                    if total_chunks:
                        st.success(f"Indexed {total_chunks} new chunks into '{active_space}'.")

        if add_url and url_input.strip():
            url = url_input.strip()
            if url in existing_names:
                st.caption("This URL is already indexed.")
            else:
                with st.spinner("Fetching and indexing the page..."):
                    try:
                        docs = load_url_docs(url)
                        n = ingest_documents(active_space, docs, embeddings)
                        if n:
                            meta["documents"].append({
                                "name": url, "chunks": n,
                                "added": datetime.now(timezone.utc).isoformat(),
                            })
                            save_meta(active_space, meta)
                            st.success(f"Indexed {n} chunks from that page.")
                        else:
                            st.warning("No readable text found at that URL.")
                    except Exception as e:
                        st.error(f"Couldn't fetch that URL: {e}")

        meta = load_meta(active_space)
        if meta["documents"]:
            st.caption(f"📄 {len(meta['documents'])} source(s) in this space:")
            for d in meta["documents"]:
                st.caption(f"• {d['name']} ({d['chunks']} chunks)")
        else:
            st.caption("No sources indexed yet.")

# ============================== MAIN AREA =================================
if not active_space:
    st.info("👈 Create or select a Knowledge Space in the sidebar to get started.")
    st.stop()

tab_chat, tab_analytics, tab_faq = st.tabs(["💬 Chat", "📊 Analytics & Gaps", "📝 Auto-FAQ"])

# --------------------------------- Chat -----------------------------------
with tab_chat:
    vs = load_vector_store(active_space, embeddings)
    history_key = f"messages::{active_space}"
    if history_key not in st.session_state:
        welcome = (
            f"Ask me anything about the documents in **{active_space}**, in "
            f"any language — I'll answer in whichever language you ask in "
            f"(or the language you've pinned in the sidebar), in {persona} mode."
        )
        st.session_state[history_key] = [{"role": "assistant", "content": welcome}]

    def _render_evidence(evidence):
        with st.expander(f"🔍 Show retrieved evidence ({len(evidence)} passages)"):
            for i, (text, meta, score) in enumerate(evidence, start=1):
                meta = meta if isinstance(meta, dict) else {}
                src = meta.get("source") or meta.get("name") or "Unknown source"
                page = meta.get("page", meta.get("page_number", "-"))
                score_str = f" · relevance {score:.2f}" if isinstance(score, (int, float)) else ""
                st.markdown(f"**[{i}] {src}** (page {page}){score_str}")
                snippet = text[:400] + ("..." if len(text) > 400 else "")
                st.caption(snippet)

    def _render_verification(fact_check, critique):
        unverified = (fact_check or {}).get("unverified", [])
        issues = (critique or {}).get("issues", [])
        flagged = bool(unverified) or ((critique or {}).get("checked") and not critique.get("passed", True))
        if flagged:
            st.caption("⚠️ Verification flagged — some claims couldn't be fully confirmed against the source text.")
            with st.expander("🧪 Verification details"):
                if unverified:
                    st.markdown("**Unconfirmed figures/quantities:** " + ", ".join(unverified))
                if issues:
                    st.markdown("**Fact-check notes:** " + "; ".join(issues))
        else:
            st.caption("✅ Verified against source documents (fact-check + self-critique pass).")
        return flagged

    for m in st.session_state[history_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("language") and m["language"] != "English":
                st.caption(f"🌐 Answered in {m['language']}")
            if "fact_check" in m:
                _render_verification(m.get("fact_check"), m.get("critique"))
            if m.get("evidence"):
                _render_evidence(m["evidence"])

    if prompt := st.chat_input("Ask a question about this knowledge space..."):
        st.session_state[history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if vs is None:
                msg = "This knowledge space doesn't have any documents indexed yet — add some in the sidebar first."
                st.markdown(msg)
                st.session_state[history_key].append({"role": "assistant", "content": msg})
            else:
                with st.spinner("Searching the knowledge base..."):
                    try:
                        result = answer_question(vs, prompt, persona, llm, answer_language=answer_language, verify=False)
                    except Exception as e:
                        result = None
                        err_msg = (
                            "⚠️ The AI service hit an error while generating this answer. "
                            "This is almost always a temporary **rate limit or quota** issue "
                            "on the free Groq API tier (too many requests in a short window) "
                            "— wait about a minute and try again."
                        )
                        st.error(err_msg)
                        with st.expander("Technical details (for debugging)"):
                            st.code(f"{type(e).__name__}: {e}")
                        st.session_state[history_key].append({"role": "assistant", "content": err_msg})

                if result is not None:
                    fact_check = result.get("fact_check", {})
                    critique = result.get("critique", {})
                    verification_flagged = bool(fact_check.get("unverified")) or (
                        critique.get("checked") and not critique.get("passed", True)
                    )
                    log_interaction(
                        active_space, persona, prompt, result["status"], result["sources"],
                        language=result.get("language", "English"),
                        verification_flagged=verification_flagged if result["status"] == "answered" else False,
                    )

                    if result["status"] == "gap":
                        st.warning(result["answer"])
                        if result.get("language") and result["language"] != "English":
                            st.caption(f"🌐 Answered in {result['language']}")
                        flag_key = f"flag_{active_space}_{len(st.session_state[history_key])}"
                        if st.button("🚩 Flag this for a human expert", key=flag_key):
                            log_escalation(active_space, persona, prompt)
                            st.success("Flagged — an admin will see this in the Analytics tab.")
                        st.session_state[history_key].append({
                            "role": "assistant", "content": result["answer"],
                            "language": result.get("language"),
                        })
                    else:
                        pages_str = ", ".join(str(p) for p in result["pages"])
                        sources_str = ", ".join(result["sources"])
                        full = f"{result['answer']}\n\n**Sources:** {sources_str} (pages {pages_str})"
                        st.markdown(full)
                        if result.get("language") and result["language"] != "English":
                            st.caption(f"🌐 Answered in {result['language']} · grounded in the original source documents")
                        _render_verification(fact_check, critique)

                        evidence = get_retrieved_evidence(vs, prompt, k=4)
                        if evidence:
                            _render_evidence(evidence)

                        st.session_state[history_key].append({
                            "role": "assistant", "content": full, "language": result.get("language"),
                            "fact_check": fact_check, "critique": critique, "evidence": evidence,
                        })

# ------------------------------ Analytics ----------------------------------
with tab_analytics:
    stats = get_analytics(active_space)

    kpi_defs = [
        ("🔎", "Total queries", stats["total_queries"], "#6366F1"),
        ("✅", "Answered", stats["answered"], "#10B981"),
        ("🕳️", "Knowledge gaps", stats["gap_count"], "#F59E0B"),
        ("📈", "Resolution rate", f"{stats['resolution_rate']}%", "#22D3EE"),
        ("🧪", "Verification flagged", stats.get("verification_flagged", 0), "#F472B6"),
    ]
    cols = st.columns(5)
    for col, (icon, label, value, color) in zip(cols, kpi_defs):
        col.markdown(
            f"""
            <div class="os-kpi" style="border-left-color:{color};">
                <div class="os-kpi-icon">{icon}</div>
                <div class="os-kpi-label">{label}</div>
                <div class="os-kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    def _render_bar_chart(title, data):
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="os-section-title">{title}</div>', unsafe_allow_html=True)
        items = list(data.items()) if hasattr(data, "items") else list(enumerate(data))
        max_val = max((v for _, v in items), default=0) or 1
        palette = ["#6366F1", "#22D3EE", "#34D399", "#FBBF24", "#F472B6", "#A78BFA"]
        bars_html = ""
        for i, (label, val) in enumerate(items):
            pct = (val / max_val) * 100
            color = palette[i % len(palette)]
            bars_html += f"""
            <div style="margin-bottom:0.7rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; color:var(--os-text-dim); margin-bottom:0.28rem;">
                    <span>{label}</span><span style="color:#FFFFFF; font-weight:700;">{val}</span>
                </div>
                <div style="background:rgba(255,255,255,0.07); border-radius:8px; height:11px; overflow:hidden;">
                    <div style="width:{pct}%; height:100%; background:linear-gradient(90deg,{color},#22D3EE);
                                border-radius:8px; box-shadow:0 0 12px {color}66;"></div>
                </div>
            </div>
            """
        st.markdown(f'<div>{bars_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if stats["by_persona"]:
            _render_bar_chart("📊 Queries by mode", stats["by_persona"])
    with chart_col2:
        if stats.get("by_language"):
            _render_bar_chart("🌐 Queries by language", stats["by_language"])


    st.markdown('<div class="os-card">', unsafe_allow_html=True)
    st.markdown('<div class="os-section-title">🕳️ Recent knowledge gaps</div>', unsafe_allow_html=True)
    st.caption(
        "Questions the assistant couldn't answer from this knowledge base — "
        "effectively a to-do list of what content to add next."
    )
    if stats["gaps"]:
        for g in stats["gaps"]:
            ts = g['ts'][:16].replace('T', ' ')
            st.markdown(
                f"""
                <div class="os-pill-row">
                    <div><b>{g['query']}</b><br><span style="color:#64748B; font-size:0.8rem;">
                    mode: {g['persona']} · {ts} UTC</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No gaps recorded yet.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="os-card">', unsafe_allow_html=True)
    st.markdown('<div class="os-section-title">🚩 Escalation queue</div>', unsafe_allow_html=True)
    escalations = get_escalations(active_space)
    if escalations:
        for e in escalations:
            resolved = e.get("resolved")
            status_html = (
                '<span class="os-pill-status-resolved">✅ RESOLVED</span>'
                if resolved else
                '<span class="os-pill-status-open">🔴 OPEN</span>'
            )
            ts = e['ts'][:16].replace('T', ' ')
            st.markdown(
                f"""
                <div class="os-pill-row">
                    <div><b>{e['query']}</b><br><span style="color:#64748B; font-size:0.8rem;">
                    mode: {e['persona']} · {ts} UTC</span></div>
                    {status_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No escalations yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------- Auto-FAQ ----------------------------------
with tab_faq:
    st.markdown('<div class="os-card">', unsafe_allow_html=True)
    st.markdown('<div class="os-section-title">📝 Auto-generate a FAQ</div>', unsafe_allow_html=True)
    st.write(
        "Generate a first-draft FAQ document from everything indexed in this "
        "space — useful for onboarding new team members or publishing a "
        "self-serve help page. This is an autonomous action: one click scans "
        "the whole knowledge base and drafts the document for you."
    )
    generate_clicked = st.button("⚡ Generate FAQ now", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if generate_clicked:
        vs = load_vector_store(active_space, embeddings)
        if vs is None:
            st.warning("No documents indexed in this space yet.")
        else:
            with st.spinner("Reading through the knowledge base and drafting an FAQ..."):
                try:
                    faq_md = generate_faq(vs, llm, active_space, language=answer_language)
                except Exception as e:
                    faq_md = None
                    st.error(
                        "⚠️ The AI service hit an error while generating the FAQ. "
                        "This is almost always a temporary rate limit or quota issue "
                        "on the free Groq API tier — wait about a minute and try again."
                    )
                    with st.expander("Technical details (for debugging)"):
                        st.code(f"{type(e).__name__}: {e}")
            if faq_md is not None and not isinstance(faq_md, str):
                # response.content can sometimes come back as a list of
                # content-part dicts instead of a plain string depending on
                # the response shape. st.markdown tolerates that loosely;
                # st.download_button does not, so normalize here rather
                # than trusting the return type.
                if isinstance(faq_md, list):
                    faq_md = "\n".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in faq_md
                    ).strip()
                else:
                    faq_md = str(faq_md)

            if faq_md:
                st.markdown('<div class="os-card">', unsafe_allow_html=True)
                st.markdown(faq_md)
                st.markdown('</div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download FAQ.md", faq_md, file_name=f"{_slug(active_space)}_faq.md")
            else:
                st.warning("Nothing to generate from yet.")
