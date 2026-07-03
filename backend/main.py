import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client using Streamlit Secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ... [Keep your page config and chat history code exactly as it is] ...

# 4. Handle New User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- REAL AI LOGIC ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Send the entire chat history to OpenAI so it remembers the conversation
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, affordable, and smart
                messages=st.session_state.messages
            )
            
            # Extract the AI's actual answer
            ai_answer = response.choices[0].message.content
            
            st.markdown(ai_answer)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
