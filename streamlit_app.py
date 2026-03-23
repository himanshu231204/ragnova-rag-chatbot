import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.data_loader import load_all_documents
from src.search import RAGSearch


load_dotenv()


DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
DEFAULT_PERSIST_DIR = "faiss_store"
DEFAULT_LLM_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
DEFAULT_EMBED_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]


def _index_exists(persist_dir: str) -> bool:
    base = Path(persist_dir)
    return (base / "faiss.index").exists() and (base / "metadata.pkl").exists()


@st.cache_resource(show_spinner=False)
def get_rag_client(
    persist_dir: str,
    embedding_model: str,
    llm_model: str,
    groq_api_key: str,
) -> RAGSearch:
    return RAGSearch(
        persist_dir=persist_dir,
        embedding_model=embedding_model,
        llm_model=llm_model,
        groq_api_key=groq_api_key,
    )


def build_index(persist_dir: str, embedding_model: str) -> None:
    # Local import avoids import cycles and keeps startup fast.
    from src.vectorstore import FaissVectorStore

    docs = load_all_documents("data")
    store = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
    store.build_from_documents(docs)


def main() -> None:
    st.set_page_config(page_title="Simple RAG Chat", page_icon="R", layout="wide")

    st.markdown(
        """
        <style>
        :root {
            --bg: #111111;
            --panel: #1a1a1a;
            --border: #2a2a2a;
            --text: #e9e9e9;
            --muted: #a0a0a0;
            --accent: #00a67e;
        }
        .stApp {
            background: radial-gradient(circle at 20% 20%, #1b1b1b 0%, var(--bg) 48%);
            color: var(--text);
        }
        .app-shell {
            max-width: 980px;
            margin: 0 auto;
            padding-top: 0.5rem;
        }
        .hero {
            text-align: center;
            margin-top: 8vh;
            margin-bottom: 1.2rem;
            font-size: clamp(1.8rem, 3vw, 2.5rem);
            font-weight: 700;
            color: #f3f3f3;
            letter-spacing: 0.01em;
        }
        .repo-link-wrap {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 0.3rem;
        }
        .repo-link {
            background: #202020;
            color: #f1f1f1 !important;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.45rem 0.9rem;
            text-decoration: none;
            font-size: 0.9rem;
        }
        .repo-link:hover {
            border-color: #3d3d3d;
            background: #252525;
        }
        .result-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Settings")
        persist_dir = st.text_input("Vector store path", value=DEFAULT_PERSIST_DIR)

        st.subheader("Embedding Model")
        embed_choice = st.selectbox(
            "Choose embedding model",
            options=DEFAULT_EMBED_MODELS + ["Custom..."],
            index=0,
        )
        if embed_choice == "Custom...":
            embedding_model = st.text_input("Custom embedding model", value=DEFAULT_EMBED_MODEL)
        else:
            embedding_model = embed_choice

        st.subheader("LLM Model")
        llm_choice = st.selectbox(
            "Choose LLM model",
            options=DEFAULT_LLM_MODELS + ["Custom..."],
            index=0,
        )
        if llm_choice == "Custom...":
            llm_model = st.text_input("Custom LLM model", value=DEFAULT_LLM_MODEL)
        else:
            llm_model = llm_choice

        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=3)

        st.subheader("Response Format")
        response_format = st.radio(
            "Choose output type",
            options=["Short Summary (400-500 words)", "Detailed"],
            index=0,
        )
        response_mode = "short" if response_format.startswith("Short") else "detailed"

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

        if st.button("Build/Rebuild Index"):
            with st.spinner("Building FAISS index from data folder..."):
                build_index(persist_dir, embedding_model)
                get_rag_client.clear()
            st.success("Index built successfully.")

    if not _index_exists(persist_dir):
        st.info("FAISS index not found. Click 'Build/Rebuild Index' in the sidebar.")
        st.stop()

    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "raw_hits" not in st.session_state:
        st.session_state.raw_hits = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    tab_chat, tab_dev = st.tabs(["Chat", "Developer Info"])

    with tab_chat:
        st.markdown('<div class="app-shell">', unsafe_allow_html=True)
        st.markdown(
            '<div class="repo-link-wrap"><a class="repo-link" href="https://github.com/himanshu231204/simple-rag-pipeline" target="_blank">GitHub Repo</a></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            st.markdown('<div class="hero">Where should we begin?</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                chunks = msg.get("chunks", [])
                for i, item in enumerate(chunks, start=1):
                    md = item.get("metadata") or {}
                    chunk_text = md.get("text", "")
                    distance = item.get("distance")
                    with st.expander(
                        f"Chunk {i} | distance={distance:.4f}" if distance is not None else f"Chunk {i}"
                    ):
                        st.write(chunk_text[:3000] if chunk_text else "No text content.")

        st.caption("Type your query and press Enter to run.")
        query = st.chat_input("Ask anything about your indexed PDFs...")

        clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.summary = ""
            st.session_state.raw_hits = []
            st.session_state.chat_history = []

        if query is not None:
            if not query.strip():
                st.warning("Please enter a query first.")
                st.stop()
            if not active_api_key:
                st.error("GROQ API key is required. Provide one from sidebar.")
                st.stop()

            st.session_state.chat_history.append({"role": "user", "content": query})

            with st.spinner("Running retrieval and summarization..."):
                rag = get_rag_client(
                    persist_dir,
                    embedding_model,
                    llm_model,
                    active_api_key,
                )
                try:
                    st.session_state.summary = rag.search_and_summarize(
                        query,
                        top_k=top_k,
                        response_mode=response_mode,
                    )
                except TypeError:
                    # Compatibility fallback for stale cached clients created before response_mode existed.
                    get_rag_client.clear()
                    rag = get_rag_client(
                        persist_dir,
                        embedding_model,
                        llm_model,
                        active_api_key,
                    )
                    st.session_state.summary = rag.search_and_summarize(query, top_k=top_k)
                st.session_state.raw_hits = rag.vectorstore.query(query, top_k=top_k)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": st.session_state.summary,
                    "chunks": st.session_state.raw_hits,
                }
            )
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_dev:
        st.subheader("Developer")
        st.markdown(
            "[![GitHub](https://img.shields.io/badge/GitHub-himanshu231204-181717?style=flat-square&logo=github)](https://github.com/himanshu231204)"
        )
        st.markdown(
            "[![LinkedIn](https://img.shields.io/badge/LinkedIn-himanshu231204-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/himanshu231204)"
        )
        st.markdown(
            "[![Twitter](https://img.shields.io/badge/Twitter-himanshu231204-1DA1F2?style=flat-square&logo=twitter)](https://twitter.com/himanshu231204)"
        )
        st.markdown(
            "[![Email](https://img.shields.io/badge/Email-himanshu231204%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:himanshu231204@gmail.com)"
        )


if __name__ == "__main__":
    main()
