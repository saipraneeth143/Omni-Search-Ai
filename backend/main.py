import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
            
            # 2. Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            
            # 3. Create Embeddings using Google's free embedding model
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
            st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
            
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
                
                # Setup Google Gemini LLM
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                
                prompt_template = ChatPromptTemplate.from_template(
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know.\n\n"
                    "Context:\n{context}\n\n"
                    "Question: {input}"
                                          )
                
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                
                sources = set([doc.metadata.get("source", "Unknown") for doc in response.get("context", [])])
                pages = set([doc.metadata.get("page", 0) + 1 for doc in response.get("context", [])])
                
                citation_text = f"\n\n**Sources Consulted:** Pages {list(pages)}"
                full_response = answer + citation_text
                
            else:
                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
                full_response = llm.predict(prompt) + "\n\n*(Note: No document uploaded. Answering using base knowledge)*"

            st.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
