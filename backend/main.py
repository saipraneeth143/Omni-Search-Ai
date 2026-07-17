import os
from datetime import datetime, timezone

import streamlit as st

from engine import (
    PERSONAS,
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        :root {
            --os-primary: #6366F1;
            --os-primary-dark: #4338CA;
            --os-accent: #22D3EE;
            --os-success: #10B981;
            --os-warning: #F59E0B;
            --os-danger: #EF4444;
            --os-bg-soft: #F8FAFC;
            --os-border: #E5E7EB;
        }

        /* App background */
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 350px);
        }

        /* Hide default Streamlit chrome clutter */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* ---------- Hero header ---------- */
        .os-hero {
            background: linear-gradient(120deg, #4338CA 0%, #6366F1 55%, #22D3EE 100%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 12px 30px -12px rgba(67, 56, 202, 0.55);
            position: relative;
            overflow: hidden;
        }
        .os-hero::after {
            content: "";
            position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0) 70%);
            border-radius: 50%;
        }
        .os-hero h1 {
            color: white;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.5px;
        }
        .os-hero p {
            color: rgba(255,255,255,0.92);
            font-size: 1.02rem;
            margin: 0;
            max-width: 700px;
        }
        .os-badges {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .os-badge {
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.35);
            color: white;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            backdrop-filter: blur(4px);
        }

        /* ---------- KPI cards ---------- */
        .os-kpi {
            background: white;
            border: 1px solid var(--os-border);
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            border-left: 4px solid var(--os-primary);
            transition: transform 0.15s ease;
            height: 100%;
        }
        .os-kpi:hover { transform: translateY(-2px); }
        .os-kpi .os-kpi-label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .os-kpi .os-kpi-value {
            font-size: 1.9rem;
            font-weight: 800;
            color: #0F172A;
            margin-top: 0.15rem;
        }
        .os-kpi .os-kpi-icon { font-size: 1.3rem; }

        /* How-it-works steps */
        .os-step {
            background: white;
            border: 1px solid var(--os-border);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            height: 100%;
            position: relative;
        }
        .os-step .os-step-num {
            position: absolute;
            top: 0.7rem; right: 0.9rem;
            font-size: 0.72rem;
            font-weight: 700;
            color: #C7CBF7;
        }
        .os-step .os-step-icon { font-size: 1.4rem; margin-bottom: 0.3rem; }
        .os-step .os-step-title {
            font-weight: 700;
            font-size: 0.92rem;
            color: #0F172A;
            margin-bottom: 0.2rem;
        }
        .os-step .os-step-desc {
            font-size: 0.82rem;
            color: #64748B;
            line-height: 1.35;
        }

        /* ---------- Section cards ---------- */
        .os-card {
            background: white;
            border: 1px solid var(--os-border);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            margin-bottom: 1rem;
        }
        .os-section-title {
            font-weight: 700;
            font-size: 1.02rem;
            color: #0F172A;
            margin-bottom: 0.3rem;
        }

        /* Gap / escalation pills */
        .os-pill-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            background: var(--os-bg-soft);
            margin-bottom: 0.4rem;
            border: 1px solid var(--os-border);
        }
        .os-pill-status-open {
            background: #FEE2E2; color: #991B1B;
            padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
        }
        .os-pill-status-resolved {
            background: #D1FAE5; color: #065F46;
            padding: 0.15rem 0.6rem; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0 !important;
        }
        section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] small {
            color: #94A3B8 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.12);
        }
        section[data-testid="stSidebar"] .stButton>button {
            background: var(--os-primary);
            color: white !important;
            border: none;
            border-radius: 8px;
            font-weight: 600;
        }
        section[data-testid="stSidebar"] .stButton>button:hover {
            background: var(--os-primary-dark);
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: rgba(255,255,255,0.06);
            border-color: rgba(255,255,255,0.18);
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
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(99,102,241,0.08), rgba(99,102,241,0.0));
            color: var(--os-primary-dark) !important;
            border-bottom: 3px solid var(--os-primary);
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.3rem 0.2rem;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 10px;
            font-weight: 600;
        }
        .stDownloadButton>button {
            border-radius: 10px;
            font-weight: 600;
            background: var(--os-success);
            color: white;
            border: none;
        }

        /* Metrics fallback (native st.metric, unused but styled just in case) */
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--os-border);
            border-radius: 14px;
            padding: 0.8rem 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- CRITICAL FIX (kept): read the API key from Streamlit secrets ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit secrets.")
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
        st.session_state[history_key] = [{
            "role": "assistant",
            "content": (
                f"Ask me anything about the documents in **{active_space}**. "
                f"I'll only answer from what's been indexed here, in {persona} mode."
            ),
        }]

    for m in st.session_state[history_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

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
                    result = answer_question(vs, prompt, persona, llm)
                log_interaction(active_space, persona, prompt, result["status"], result["sources"])

                if result["status"] == "gap":
                    st.warning(result["answer"])
                    flag_key = f"flag_{active_space}_{len(st.session_state[history_key])}"
                    if st.button("🚩 Flag this for a human expert", key=flag_key):
                        log_escalation(active_space, persona, prompt)
                        st.success("Flagged — an admin will see this in the Analytics tab.")
                    st.session_state[history_key].append({"role": "assistant", "content": result["answer"]})
                else:
                    pages_str = ", ".join(str(p) for p in result["pages"])
                    sources_str = ", ".join(result["sources"])
                    full = f"{result['answer']}\n\n**Sources:** {sources_str} (pages {pages_str})"
                    st.markdown(full)
                    st.caption("✅ Answered from the knowledge base above.")
                    st.session_state[history_key].append({"role": "assistant", "content": full})

# ------------------------------ Analytics ----------------------------------
with tab_analytics:
    stats = get_analytics(active_space)

    kpi_defs = [
        ("🔎", "Total queries", stats["total_queries"], "#6366F1"),
        ("✅", "Answered", stats["answered"], "#10B981"),
        ("🕳️", "Knowledge gaps", stats["gap_count"], "#F59E0B"),
        ("📈", "Resolution rate", f"{stats['resolution_rate']}%", "#22D3EE"),
    ]
    cols = st.columns(4)
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

    if stats["by_persona"]:
        st.markdown('<div class="os-card">', unsafe_allow_html=True)
        st.markdown('<div class="os-section-title">📊 Queries by mode</div>', unsafe_allow_html=True)
        st.bar_chart(stats["by_persona"], color="#6366F1")
        st.markdown('</div>', unsafe_allow_html=True)

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
                faq_md = generate_faq(vs, llm, active_space)
            if faq_md:
                st.markdown('<div class="os-card">', unsafe_allow_html=True)
                st.markdown(faq_md)
                st.markdown('</div>', unsafe_allow_html=True)
                st.download_button("⬇️ Download FAQ.md", faq_md, file_name=f"{_slug(active_space)}_faq.md")
            else:
                st.warning("Nothing to generate from yet.")
