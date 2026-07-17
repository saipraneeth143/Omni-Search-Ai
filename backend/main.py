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

st.set_page_config(page_title="OmniSearch AI 🔍", layout="wide", page_icon="🔍")

# --- CRITICAL FIX (kept): read the API key from Streamlit secrets ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()

embeddings = get_embeddings()
llm = get_llm()

st.title("OmniSearch AI 🔍")
st.caption(
    "A grounded, multi-source knowledge assistant — not a general chatbot. "
    "It only answers from the documents your organization has added, and says "
    "so explicitly when it can't."
)

# ============================== SIDEBAR ==================================
with st.sidebar:
    st.header("🗂️ Knowledge Space")
    st.caption("A named, persistent knowledge base — e.g. 'HR Policies' or 'Course 101'.")

    spaces = list_spaces()
    create_new_label = "➕ Create new space..."
    options = spaces + [create_new_label]
    default_idx = 0 if spaces else 0
    choice = st.selectbox("Active space", options, index=default_idx)

    active_space = None
    if choice == create_new_label:
        new_name = st.text_input("Name your new space", placeholder="e.g. HR Policies 2026")
        if new_name.strip():
            active_space = new_name.strip()
            save_meta(active_space, load_meta(active_space))
    else:
        active_space = choice

    st.divider()
    st.header("🎭 Assistant Mode")
    persona = st.selectbox("Industry / persona", list(PERSONAS.keys()))
    st.caption(PERSONAS[persona])

    st.divider()
    st.header("📥 Add Knowledge")
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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total queries", stats["total_queries"])
    c2.metric("Answered", stats["answered"])
    c3.metric("Knowledge gaps", stats["gap_count"])
    c4.metric("Resolution rate", f"{stats['resolution_rate']}%")

    if stats["by_persona"]:
        st.subheader("Queries by mode")
        st.bar_chart(stats["by_persona"])

    st.subheader("🕳️ Recent knowledge gaps")
    st.caption(
        "Questions the assistant couldn't answer from this knowledge base — "
        "effectively a to-do list of what content to add next."
    )
    if stats["gaps"]:
        for g in stats["gaps"]:
            st.write(f"- **{g['query']}** _(mode: {g['persona']}, {g['ts'][:16].replace('T', ' ')} UTC)_")
    else:
        st.caption("No gaps recorded yet.")

    st.subheader("🚩 Escalation queue")
    escalations = get_escalations(active_space)
    if escalations:
        for e in escalations:
            status = "✅ resolved" if e.get("resolved") else "🔴 open"
            st.write(f"- **{e['query']}** _(mode: {e['persona']}, {e['ts'][:16].replace('T', ' ')} UTC)_ — {status}")
    else:
        st.caption("No escalations yet.")

# -------------------------------- Auto-FAQ ----------------------------------
with tab_faq:
    st.write(
        "Generate a first-draft FAQ document from everything indexed in this "
        "space — useful for onboarding new team members or publishing a "
        "self-serve help page. This is an autonomous action: one click scans "
        "the whole knowledge base and drafts the document for you."
    )
    if st.button("⚡ Generate FAQ now"):
        vs = load_vector_store(active_space, embeddings)
        if vs is None:
            st.warning("No documents indexed in this space yet.")
        else:
            with st.spinner("Reading through the knowledge base and drafting an FAQ..."):
                faq_md = generate_faq(vs, llm, active_space)
            if faq_md:
                st.markdown(faq_md)
                st.download_button("Download FAQ.md", faq_md, file_name=f"{_slug(active_space)}_faq.md")
            else:
                st.warning("Nothing to generate from yet.")

