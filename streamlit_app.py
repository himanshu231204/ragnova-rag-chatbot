"""
RAGNOVA - Retrieval-Augmented Generation Application
Main Entry Point for Streamlit Application
"""

import streamlit as st
from dotenv import load_dotenv
from src.ui.config import APP_NAME, APP_TAGLINE, DEFAULT_DATA_DIR
from src.ui.styles import inject_styles
from src.ui.sidebar import render_sidebar_settings, render_build_index_button
from src.ui.pages.chat import render_chat_page
from src.ui.pages.about import render_about_page
from src.ui.pages.developer import render_developer_page
from src.ui.pages.analytics import render_analytics_page
from src.backend.index_manager import index_exists, get_source_documents_count, build_index
from src.backend.rag_client import get_rag_client

load_dotenv()



def initialize_app():
    """Initialize Streamlit app configuration and session state."""
    st.set_page_config(page_title=APP_NAME, page_icon="RN", layout="wide")

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"


def main():
    """Main application entry point."""
    # Initialize app
    initialize_app()

    # Apply theme styling
    is_light_mode = st.session_state.theme_mode == "Light"
    inject_styles(is_light_mode)

    # Render sidebar and get configuration
    config = render_sidebar_settings()
    
    # Add app_tagline to config for pages
    config["app_tagline"] = APP_TAGLINE

    # Render build index button
    render_build_index_button(config, build_index)

    # Check if index exists
    if not index_exists(config["persist_dir"]):
        source_count = get_source_documents_count(DEFAULT_DATA_DIR)
        if source_count == 0:
            st.warning(
                "FAISS index not found, and no source docs were detected in `data/`. "
                "For Streamlit Cloud, push your documents to the repo (or include a prebuilt `faiss_store`), "
                "then click Build/Rebuild Index."
            )
        else:
            st.info(
                f"FAISS index not found. Detected {source_count} source file(s). "
                "Click 'Build/Rebuild Index' in the sidebar."
            )
        st.stop()

    # Render main tabs
    tab_chat, tab_analytics, tab_about, tab_dev = st.tabs(["Chat", "📊 Analytics", "About RAGNOVA", "Developer Info"])

    with tab_chat:
        render_chat_page(config)

    with tab_analytics:
        render_analytics_page()

    with tab_about:
        render_about_page()

    with tab_dev:
        render_developer_page()

    # Render footer
    st.divider()
    st.markdown(
        """
        <div style="text-align:center; font-size:14px;">
            👨‍💻 Developed by <b>Himanshu Kumar</b><br><br>
            🔗 
            <a href="https://www.linkedin.com/in/himanshu231204" target="_blank">LinkedIn</a> |
            <a href="https://github.com/himanshu231204" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
