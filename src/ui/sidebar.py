"""Sidebar UI and Settings Management"""

import os
import streamlit as st
from .config import (
    APP_NAME, APP_TAGLINE, DEFAULT_PERSIST_DIR, DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL, DEFAULT_THEME, DEFAULT_DATA_DIR
)


def render_sidebar_settings():
    """Render sidebar with settings and return configuration."""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="settings-card">
                <div class="t">{APP_NAME} Settings</div>
                <div class="s">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Theme selection
        st.selectbox(
            "Theme Mode",
            options=["Dark", "Light"],
            key="theme_mode",
            help="Switch the application appearance.",
        )

        # Vector store path
        persist_dir = st.text_input("Vector store path", value=DEFAULT_PERSIST_DIR)

        # Fixed models (no selector)
        embedding_model = DEFAULT_EMBED_MODEL
        llm_model = DEFAULT_LLM_MODEL

        # Retrieval settings
        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=3)

        # Response format
        st.subheader("Response Format")
        response_format = st.radio(
            "Choose output type",
            options=["Short Summary (400-500 words)", "Detailed"],
            index=0,
        )
        response_mode = "short" if response_format.startswith("Short") else "detailed"

        # API Key configuration
        st.subheader("API Key")
        env_key = os.getenv("GROQ_API_KEY", "")
        use_custom_key = st.toggle("Use custom GROQ API key", value=False)
        custom_key = st.text_input("Custom GROQ API key", type="password", disabled=not use_custom_key)

        if use_custom_key:
            active_api_key = custom_key.strip()
            if not active_api_key:
                st.warning("Enter your custom GROQ API key to run queries.")
        else:
            active_api_key = env_key.strip()
            if not active_api_key:
                st.warning("No GROQ_API_KEY found in environment. Enable custom key to provide one.")

        return {
            "persist_dir": persist_dir,
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "top_k": top_k,
            "response_mode": response_mode,
            "active_api_key": active_api_key,
        }


def render_build_index_button(config, build_index_callback):
    """Render index build button in sidebar."""
    from src.backend.index_manager import get_source_documents_count
    
    with st.sidebar:
        if st.button("Build/Rebuild Index"):
            source_count = get_source_documents_count(DEFAULT_DATA_DIR)
            if source_count == 0:
                st.error("No source documents found in `data/`. Add documents first, then rebuild index.")
                st.stop()
            with st.spinner("Building FAISS index from data folder..."):
                build_index_callback(config["persist_dir"], config["embedding_model"])
            st.success("Index built successfully.")
