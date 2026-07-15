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
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">ðŸ’¬</div>
                <h4 style="color: #E8EAED; font-size: 0.9rem; margin: 0 0 4px 0;">Ask</h4>
                <p style="color: #6B7280; font-size: 0.75rem; margin: 0;">Chat & get cited answers</p>
            </div>
        </div>
    </div>
    
    <div style="text-align: center; margin-top: 2rem;">
        <p style="color: #374151; font-size: 0.75rem;">
            â† Upload a PDF in the sidebar to get started
        </p>
    </div>
    """, unsafe_allow_html=True)


# â”€â”€ Display Chat History â”€â”€
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
# CHAT INPUT â€” Process user questions
# â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”
if prompt := st.chat_input("Ask anything about your document..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("ðŸ” Searching knowledge base..."):
            try:
                if st.session_state.vector_store is not None:
                    # â”€â”€ RAG Path: Answer from document context â”€â”€
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

                    response = call_with_backoff(rag_chain.invoke, {"input": prompt})
                    answer = response["answer"]

                    # Build clean citations
                    context_docs = response.get("context", [])
                    pages = sorted(set(
                        doc.metadata.get("page", 0) + 1 for doc in context_docs
                    ))

                    citation = format_citation(pages) if pages else ""
                    full_response = f"{answer}\n\n---\n{citation}"

                else:
                    # â”€â”€ Fallback: No document uploaded â”€â”€
                    llm = get_llm()
                    result = call_with_backoff(llm.invoke, prompt)
                    full_response = (
                        result.content
                        + "\n\n---\n"
                        + "ðŸ’¡ *No document uploaded â€” answering from general knowledge. "
                        "Upload a PDF in the sidebar for document-specific answers.*"
                    )

            except Exception as e:
                if is_quota_error(e):
                    full_response = (
                        "â³ **Gemini's free-tier rate limit was hit** and automatic retries "
                        "didn't clear it in time. This quota is shared across everyone using "
                        "the demo right now â€” please wait about a minute and ask again."
                    )
                else:
                    full_response = (
                        f"âš ï¸ **Something went wrong while generating a response.**\n\n"
                        f"```\n{type(e).__name__}: {e}\n```\n\n"
                        f"Please try again or re-upload your document."
                    )

        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})