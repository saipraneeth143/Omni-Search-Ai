import streamlit as st
import time

# 1. Setup the UI (This replaces your FastAPI app instantiation and root endpoint)
st.set_page_config(page_title="OmniSearch AI", page_icon="🔍")
st.title("OmniSearch AI 🔍")
st.caption("Welcome to the OmniSearch AI. System is running.")

# 2. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle New User Input (This replaces your @app.post("/api/chat") endpoint)
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Save user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- MOCK RAG/LLM LOGIC ---
    # In the future, you will replace this block with your Pinecone/OpenAI code
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            time.sleep(1) # Simulating a slight delay for realism
            
            mock_answer = "This is an AI-generated answer retrieved from your fragmented documents."
            mock_citations = ["HR_Handbook.pdf - Page 4", "Company_Policy_Notion_Page"]
            
            # Format the output beautifully for Streamlit
            response_text = f"{mock_answer}\n\n**Citations:**\n"
            for citation in mock_citations:
                response_text += f"* `{citation}`\n"
            
            st.markdown(response_text)
            
    # Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
