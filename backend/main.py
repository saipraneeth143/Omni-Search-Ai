import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

st.set_page_config(page_title="OmniSearch AI 🔍", layout="wide")
st.title("OmniSearch AI 🔍")

# Retrieve API Key securely from Streamlit Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please add your GOOGLE_API_KEY to Streamlit secrets.")
    st.stop()

# Initialize session state for vector store and chat history
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload a PDF in the sidebar, and I'll help you search through it!"}]

# --- SIDEBAR: PDF UPLOADER ---
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Processing PDF... (this may take a moment on the free tier)"):
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # 1. Load the PDF text
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            # CRITICAL FIX 1: Strip invisible null bytes (\x00) from the PDF text. 
            # If we don't do this, Gemini will instantly crash with a 400 Bad Request.
            for doc in docs:
                doc.page_content = doc.page_content.replace('\x00', '')
            
            # 2. Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            # CRITICAL FIX 3: Guard against PDFs with no extractable text
            # (e.g. scanned/image-only PDFs). Without this check, FAISS.from_documents
            # receives an empty list, embed_documents([]) returns [], and
            # faiss.IndexFlatL2(len(embeddings[0])) crashes with an IndexError.
            if not final_documents:
                os.remove(temp_file_path)
                st.error(
                    "⚠️ No readable text was found in this PDF. It may be a scanned "
                    "or image-only document. Please try a PDF with selectable text "
                    "(or run it through OCR first)."
                )
                st.stop()

            # 3. Create Embeddings
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
                st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
            except Exception as e:
                os.remove(temp_file_path)
                st.error(f"Failed to build the vector index: {e}")
                st.stop()

            os.remove(temp_file_path)
            st.success(f"Successfully indexed {len(final_documents)} chunks from {uploaded_file.name}!")

# --- MAIN CHAT INTERFACE ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about your uploaded documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base & thinking..."):
            
            if st.session_state.vector_store is not None:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                
                # CRITICAL FIX 2: Turn off Gemini's strict safety filters so it doesn't block PDF chunks.
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
                
                # We also ensure temperature is a float (0.0) to satisfy strict API requirements
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash", 
                    temperature=0.0,
                    safety_settings=safety_settings
                )
                
                system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know.\n\n"
                    "Context:\n{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                sources = set([doc.metadata.get("source", "Unknown") for doc in response.get("context", [])])
                pages = set([doc.metadata.get("page", 0) + 1 for doc in response.get("context", [])])
                
                citation_text = f"\n\n**Sources Consulted:** Pages {list(pages)}"
                full_response = answer + citation_text
                
            else:
                # CRITICAL FIX 4: .predict() was deprecated in LangChain 0.1.7 and is
                # removed in current versions — use .invoke().content instead.
                # Also switched to gemini-2.5-flash for consistency with the RAG path
                # above (gemini-1.5-flash is an older model).
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
                full_response = llm.invoke(prompt).content + "\n\n*(Note: No document uploaded. Answering using base knowledge)*"

            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
