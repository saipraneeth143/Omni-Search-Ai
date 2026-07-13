import streamlit as st
import tempfile
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
    HarmCategory,
    HarmBlockThreshold,
)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE CONFIG — must be the first Streamlit command
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="OmniSearch AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREMIUM CUSTOM CSS — Dark Glassmorphism Theme
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ───────────────────────────────── */
    *, *::before, *::after { font-family: 'Inter', sans-serif !important; }

    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #0d1527 40%, #0f1a30 100%);
    }

    /* ── Hide Streamlit defaults ────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Custom Scrollbar ───────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00D4FF44, #7C3AED44);
        border-radius: 10px;
    }

    /* ── Sidebar ── Glassmorphism ────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.95) 0%, rgba(10, 15, 30, 0.98) 100%) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(0, 212, 255, 0.08);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] p {
        color: #9CA3AF;
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(0, 212, 255, 0.1);
        margin: 1.2rem 0;
    }

    /* ── File Uploader ── Animated Border ────────────── */
    [data-testid="stFileUploader"] {
        background: rgba(0, 212, 255, 0.03);
        border: 1px dashed rgba(0, 212, 255, 0.25);
        border-radius: 12px;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 212, 255, 0.5);
        background: rgba(0, 212, 255, 0.06);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.08);
    }

    [data-testid="stFileUploader"] label p {
        color: #E8EAED !important;
        font-weight: 500 !important;
    }

    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #00D4FF, #7C3AED) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }

    /* ── Chat Messages ── Custom Bubbles ──────────────── */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
        animation: fadeSlideIn 0.4s ease-out;
    }

    /* User message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(0, 212, 255, 0.04);
        border-left: 3px solid #00D4FF;
    }

    /* Assistant message accent */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: rgba(124, 58, 237, 0.04);
        border-left: 3px solid #7C3AED;
    }

    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Chat Input ── Styled ────────────────────────── */
    [data-testid="stChatInput"] {
        border-color: rgba(0, 212, 255, 0.15) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 12px !important;
        color: #E8EAED !important;
        font-size: 0.95rem !important;
        caret-color: #00D4FF;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(0, 212, 255, 0.4) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.1) !important;
    }

    [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #00D4FF, #7C3AED) !important;
        border: none !important;
        border-radius: 10px !important;
    }

    /* ── Buttons ── Gradient Style ────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #1e293b, #1a2332) !important;
        color: #E8EAED !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        border-color: rgba(0, 212, 255, 0.5) !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.15) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Metrics Cards ───────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(0, 212, 255, 0.04);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }

    [data-testid="stMetric"] label { color: #9CA3AF !important; font-size: 0.8rem !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00D4FF !important; font-weight: 700 !important; }

    /* ── Success/Error/Warning alerts ─────────────────── */
    .stSuccess {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.2) !important;
        border-radius: 10px !important;
        color: #6EE7B7 !important;
    }

    .stAlert, [data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* ── Spinner ─────────────────────────────────────── */
    .stSpinner > div { color: #00D4FF !important; }

    /* ── Expander ─────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
    }

    /* ── Tabs ──────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #9CA3AF;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(0, 212, 255, 0.1) !important;
        color: #00D4FF !important;
    }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API KEY — Secure retrieval from Streamlit secrets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 **API Key Missing** — Please add your `GOOGLE_API_KEY` to Streamlit secrets.")
    st.info("Go to **Settings → Secrets** in Streamlit Cloud and add:\n```\nGOOGLE_API_KEY = \"your-key-here\"\n```")
    st.stop()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SESSION STATE — Initialize all state variables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = 0
if "doc_pages" not in st.session_state:
    st.session_state.doc_pages = 0
if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPER — Build safety settings for Gemini
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def get_safety_settings():
    """Disable Gemini's strict safety filters to prevent blocking PDF content."""
    return {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


def get_llm():
    """Create a ChatGoogleGenerativeAI instance with safe settings."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        safety_settings=get_safety_settings(),
    )


def format_citation(pages):
    """Format page numbers into a clean citation string."""
    sorted_pages = sorted(pages)
    if len(sorted_pages) == 1:
        return f"📄 **Source:** Page {sorted_pages[0]}"
    else:
        page_list = ", ".join(str(p) for p in sorted_pages)
        return f"📄 **Sources:** Pages {page_list}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — Document Upload & Controls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    # ── Logo & Branding ──
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 0.5rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🔮</div>
        <h1 style="
            background: linear-gradient(135deg, #00D4FF, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.6rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        ">OmniSearch AI</h1>
        <p style="color: #6B7280; font-size: 0.78rem; margin-top: 4px; font-weight: 400;">
            AI-Powered Knowledge Assistant
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── File Uploader ──
    st.markdown("##### 📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        help="Upload a PDF document to start asking questions about its content.",
    )

    # ── Process PDF (only if new file) ──
    if uploaded_file is not None:
        # Create a unique ID for this file to prevent re-processing on every rerun
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.processed_file_id != file_id:
            with st.spinner("🔄 Ingesting document..."):
                try:
                    # Write to a safe temporary file (works on Streamlit Cloud)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        temp_path = tmp.name

                    # 1. Load PDF
                    loader = PyPDFLoader(temp_path)
                    docs = loader.load()

                    # 2. Strip null bytes that crash the Gemini API
                    for doc in docs:
                        doc.page_content = doc.page_content.replace('\x00', '')

                    # 3. Split into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    chunks = text_splitter.split_documents(docs)

                    # 4. Guard against empty/scanned PDFs
                    if not chunks:
                        os.unlink(temp_path)
                        st.error(
                            "⚠️ **No readable text found.** This PDF may be scanned "
                            "or image-only. Please upload a PDF with selectable text."
                        )
                        st.stop()

                    # 5. Create embeddings and vector store
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/text-embedding-004"
                    )
                    st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

                    # 6. Store document metadata in session state
                    st.session_state.doc_name = uploaded_file.name
                    st.session_state.doc_chunks = len(chunks)
                    st.session_state.doc_pages = len(docs)
                    st.session_state.processed_file_id = file_id

                    # 7. Clean up temp file
                    os.unlink(temp_path)

                except Exception as e:
                    # Clean up temp file on error
                    if 'temp_path' in locals() and os.path.exists(temp_path):
                        os.unlink(temp_path)
                    st.error(f"❌ **Processing failed:** {e}")
                    st.stop()

            st.success("✅ Document indexed successfully!")

    # ── Document Stats ──
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.markdown("##### 📊 Document Stats")

        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.doc_pages)
        col2.metric("Chunks", st.session_state.doc_chunks)

        st.markdown(f"""
        <div style="
            background: rgba(0, 212, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            margin-top: 0.5rem;
        ">
            <p style="color: #9CA3AF; font-size: 0.75rem; margin: 0 0 4px 0;">ACTIVE DOCUMENT</p>
            <p style="color: #E8EAED; font-size: 0.85rem; margin: 0; font-weight: 500;">
                📎 {st.session_state.doc_name}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Controls ──
    st.markdown("---")

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("📄  Remove Document", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.doc_name = None
        st.session_state.doc_chunks = 0
        st.session_state.doc_pages = 0
        st.session_state.processed_file_id = None
        st.session_state.messages = []
        st.rerun()

    # ── Tech Stack Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0;">
        <p style="color: #4B5563; font-size: 0.7rem; margin: 0 0 8px 0;">POWERED BY</p>
        <div style="display: flex; gap: 6px; justify-content: center; flex-wrap: wrap;">
            <span style="
                background: rgba(0, 212, 255, 0.08);
                color: #00D4FF;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.68rem;
                font-weight: 500;
                border: 1px solid rgba(0, 212, 255, 0.15);
            ">Gemini 2.5</span>
            <span style="
                background: rgba(124, 58, 237, 0.08);
                color: #A78BFA;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.68rem;
                font-weight: 500;
                border: 1px solid rgba(124, 58, 237, 0.15);
            ">LangChain</span>
            <span style="
                background: rgba(16, 185, 129, 0.08);
                color: #6EE7B7;
                padding: 3px 10px;
                border-radius: 20px;
                font-size: 0.68rem;
                font-weight: 500;
                border: 1px solid rgba(16, 185, 129, 0.15);
            ">FAISS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN AREA — Hero or Chat Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Hero Section (shown when no document is loaded and no messages) ──
if st.session_state.vector_store is None and not st.session_state.messages:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem 2rem 2rem;
        max-width: 700px;
        margin: 0 auto;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔮</div>
        <h1 style="
            background: linear-gradient(135deg, #00D4FF 0%, #7C3AED 50%, #EC4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0 0 0.8rem 0;
            letter-spacing: -1px;
            line-height: 1.1;
        ">OmniSearch AI</h1>
        <p style="
            color: #9CA3AF;
            font-size: 1.15rem;
            line-height: 1.6;
            margin-bottom: 2.5rem;
            font-weight: 300;
        ">
            Your AI-powered knowledge assistant. Upload any PDF and ask questions — 
            get instant, context-aware answers with exact page citations.
        </p>
        <div style="
            display: flex;
            gap: 1.5rem;
            justify-content: center;
            flex-wrap: wrap;
        ">
            <div style="
                background: rgba(0, 212, 255, 0.05);
                border: 1px solid rgba(0, 212, 255, 0.12);
                border-radius: 16px;
                padding: 1.5rem;
                width: 180px;
                text-align: center;
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">📄</div>
                <h4 style="color: #E8EAED; font-size: 0.9rem; margin: 0 0 4px 0;">Upload</h4>
                <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">Drop any PDF document</p>
            </div>
            <div style="
                background: rgba(124, 58, 237, 0.05);
                border: 1px solid rgba(124, 58, 237, 0.12);
                border-radius: 16px;
                padding: 1.5rem;
                width: 180px;
                text-align: center;
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🧠</div>
                <h4 style="color: #E8EAED; font-size: 0.9rem; margin: 0 0 4px 0;">Understand</h4>
                <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">AI processes & indexes it</p>
            </div>
            <div style="
                background: rgba(236, 72, 153, 0.05);
                border: 1px solid rgba(236, 72, 153, 0.12);
                border-radius: 16px;
                padding: 1.5rem;
                width: 180px;
                text-align: center;
            ">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">💬</div>
                <h4 style="color: #E8EAED; font-size: 0.9rem; margin: 0 0 4px 0;">Ask</h4>
                <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">Chat & get cited answers</p>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: #374151; font-size: 0.75rem;">
            ← Upload a PDF in the sidebar to get started
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Display Chat History ──
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHAT INPUT — Process user questions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if prompt := st.chat_input("Ask anything about your document..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                if st.session_state.vector_store is not None:
                    # ── RAG Path: Answer from document context ──
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    )

                    llm = get_llm()

                    system_prompt = (
                        "You are OmniSearch AI, a helpful and precise knowledge assistant. "
                        "Use the following retrieved context from the user's uploaded document "
                        "to answer their question accurately.\n\n"
                        "Rules:\n"
                        "- Base your answer ONLY on the provided context.\n"
                        "- If the context doesn't contain enough information, say so honestly.\n"
                        "- Use clear formatting with bullet points and headers when appropriate.\n"
                        "- Be concise but thorough.\n\n"
                        "Context:\n{context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    qa_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, qa_chain)

                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]

                    # Build clean citations
                    context_docs = response.get("context", [])
                    pages = sorted(set(
                        doc.metadata.get("page", 0) + 1 for doc in context_docs
                    ))

                    citation = format_citation(pages) if pages else ""
                    full_response = f"{answer}\n\n---\n{citation}"

                else:
                    # ── Fallback: No document uploaded ──
                    llm = get_llm()
                    result = llm.invoke(prompt)
                    full_response = (
                        result.content
                        + "\n\n---\n"
                        + "💡 *No document uploaded — answering from general knowledge. "
                        "Upload a PDF in the sidebar for document-specific answers.*"
                    )

            except Exception as e:
                full_response = (
                    f"⚠️ **Something went wrong while generating a response.**\n\n"
                    f"```\n{type(e).__name__}: {e}\n```\n\n"
                    f"Please try again or re-upload your document."
                )

        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
