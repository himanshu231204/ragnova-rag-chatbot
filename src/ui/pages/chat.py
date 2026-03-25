"""Chat Tab UI and Logic"""

import streamlit as st
from ..config import APP_NAME, MAX_CHUNK_PREVIEW
from src.backend.rag_client import get_rag_client


def render_chat_page(config):
    """Render the main chat interface."""
    st.markdown('<div class="app-shell">', unsafe_allow_html=True)
    st.markdown(
        '<div class="repo-link-wrap"><a class="repo-link" href="https://github.com/himanshu231204/simple-rag-pipeline" target="_blank"><img src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub logo"><span>GitHub Repo</span></a></div>',
        unsafe_allow_html=True,
    )

    # Initialize session state
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "raw_hits" not in st.session_state:
        st.session_state.raw_hits = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Display welcome message if no chat history
    if not st.session_state.chat_history:
        st.markdown(f'<div class="hero"><span class="brand-title">{APP_NAME}</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="subhero"><span class="brand-tagline">{config["app_tagline"]}</span>. '
            f'Ask from your indexed docs and get real-time answers.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="start-space"></div>', unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.chat_history:
        avatar = "🧑" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            chunks = msg.get("chunks", [])
            for i, item in enumerate(chunks, start=1):
                md = item.get("metadata") or {}
                chunk_text = md.get("text", "")
                distance = item.get("distance")
                expander_label = (
                    f"Chunk {i} | distance={distance:.4f}"
                    if distance is not None
                    else f"Chunk {i}"
                )
                with st.expander(expander_label):
                    st.write(chunk_text[:MAX_CHUNK_PREVIEW] if chunk_text else "No text content.")

    # Process pending query
    if st.session_state.pending_query:
        query_to_run = st.session_state.pending_query

        try:
            rag = get_rag_client(
                config["persist_dir"],
                config["embedding_model"],
                config["llm_model"],
                config["active_api_key"],
            )
        except ValueError as e:
            st.error(
                f"**Embedding Model Mismatch Error:**\n\n{str(e)}\n\n"
                f"**Solution:** Click 'Build/Rebuild Index' in the sidebar to rebuild with the selected embedding model."
            )
            st.session_state.pending_query = None
            st.stop()

        with st.chat_message("assistant", avatar="🤖"):
            try:
                st.session_state.summary = st.write_stream(
                    rag.stream_search_and_summarize(
                        query_to_run,
                        top_k=config["top_k"],
                        response_mode=config["response_mode"],
                    )
                )
            except (TypeError, AttributeError):
                get_rag_client.clear()
                rag = get_rag_client(
                    config["persist_dir"],
                    config["embedding_model"],
                    config["llm_model"],
                    config["active_api_key"],
                )
                st.session_state.summary = rag.search_and_summarize(query_to_run, top_k=config["top_k"])

        st.session_state.raw_hits = rag.vectorstore.query(query_to_run, top_k=config["top_k"])
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": st.session_state.summary,
                "chunks": st.session_state.raw_hits,
            }
        )
        st.session_state.pending_query = None
        st.rerun()

    # Chat input area
    st.caption("Type your query and press Enter.")
    query = st.chat_input(f"Ask {APP_NAME} anything about your indexed PDFs...")

    col1, col2 = st.columns([3, 1])
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.summary = ""
        st.session_state.raw_hits = []
        st.session_state.chat_history = []
        st.session_state.pending_query = None
        st.rerun()

    # Handle query submission
    if query is not None:
        if not query.strip():
            st.warning("Please enter a query first.")
            st.stop()
        if not config["active_api_key"]:
            st.error("GROQ API key is required. Provide one from sidebar.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.pending_query = query
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
