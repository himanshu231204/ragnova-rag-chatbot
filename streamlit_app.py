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
APP_NAME = "RAGNOVA"
APP_TAGLINE = "Your live AI knowledge copilot"
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

def _source_documents_count(data_dir: str = "data") -> int:
    base = Path(data_dir)
    if not base.exists():
        return 0
    patterns = ["**/*.pdf", "**/*.txt", "**/*.csv", "**/*.xlsx", "**/*.docx", "**/*.json"]
    return sum(len(list(base.glob(pattern))) for pattern in patterns)

@st.cache_resource(show_spinner=False)
def get_rag_client(
    persist_dir: str,
    embedding_model: str,
    llm_model: str,
    groq_api_key: str,
) -> RAGSearch:
    try:
        return RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
            groq_api_key=groq_api_key,
        )
    except ValueError as e:
        # Re-raise to be caught by the caller
        raise e
    

def build_index(persist_dir: str, embedding_model: str) -> None:
    from src.vectorstore import FaissVectorStore

    docs = load_all_documents("data")
    store = FaissVectorStore(persist_dir=persist_dir, embedding_model=embedding_model)
    store.build_from_documents(docs)

def _light_mode_overrides() -> str:
    return """
        .stApp {
            background: radial-gradient(circle at 20% 20%, #f3f8ff 0%, #eaf1fb 45%, #f7f9fc 100%);
            color: #0f172a;
        }
        section[data-testid="stSidebar"] > div:first-child {
            background: linear-gradient(180deg, #f3f8ff 0%, #e6eef9 55%, #edf1ff 100%);
            border-right: 1px solid #c9d7eb;
        }
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stCaption {
            color: #1f2a44 !important;
        }
        section[data-testid="stSidebar"] .stTextInput input,
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] .stTextArea textarea {
            background: #ffffff !important;
            border: 1px solid #b4c7e3 !important;
            color: #12213a !important;
        }
        .settings-card {
            background: linear-gradient(145deg, #ffffff 0%, #edf3ff 60%, #eaf7ff 100%);
            border: 1px solid #b9ccec;
        }
        .settings-card .t { color: #16233d; }
        .settings-card .s { color: #435575; }
        .repo-link {
            background: #ffffff;
            color: #0f172a !important;
            border: 1px solid #c2d2ea;
        }
        .repo-link:hover {
            border-color: #9db8de;
            background: #f6f9ff;
        }
        .subhero {
            color: #31435f;
        }
        .brand-tagline {
            background: linear-gradient(90deg, #1f3a70 0%, #1f7c69 50%, #5b3b9b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
        }
        .dev-card {
            background: linear-gradient(145deg, #ffffff 0%, #f4f8ff 100%);
            border: 1px solid #c6d8ef;
            box-shadow: 0 8px 22px rgba(71, 102, 151, 0.12);
        }
        .dev-title, .dev-role, .dev-desc { color: #1b2a45; }
        .dev-pill {
            border: 1px solid #bfd3ee;
            background: #f7faff;
        }
        .dev-pill .k { color: #5a6c8d; }
        .dev-pill .v { color: #1a2a45; }
        .social-link {
            color: #ffffff !important;
        }
    """

def _inject_styles(is_light_mode: bool) -> None:
    css_path = Path("assets/ragnova.css")
    css_text = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    mode_overrides = _light_mode_overrides() if is_light_mode else ""
    css_text = css_text.replace("/* MODE_OVERRIDES */", mode_overrides)
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="RN", layout="wide")

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Dark"

    is_light_mode = st.session_state.theme_mode == "Light"
    _inject_styles(is_light_mode)

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
        st.selectbox(
            "Theme Mode",
            options=["Dark", "Light"],
            key="theme_mode",
            help="Switch the application appearance.",
        )

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
            source_count = _source_documents_count("data")
            if source_count == 0:
                st.error("No source documents found in `data/`. Add documents first, then rebuild index.")
                st.stop()
            with st.spinner("Building FAISS index from data folder..."):
                build_index(persist_dir, embedding_model)
                get_rag_client.clear()
            st.success("Index built successfully.")

    if not _index_exists(persist_dir):
        source_count = _source_documents_count("data")
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

    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "raw_hits" not in st.session_state:
        st.session_state.raw_hits = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    tab_chat, tab_about, tab_dev = st.tabs(["Chat", "About RAGNOVA", "Developer Info"])

    with tab_chat:
        st.markdown('<div class="app-shell">', unsafe_allow_html=True)
        st.markdown(
            '<div class="repo-link-wrap"><a class="repo-link" href="https://github.com/himanshu231204/simple-rag-pipeline" target="_blank"><img src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub logo"><span>GitHub Repo</span></a></div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.chat_history:
            st.markdown(f'<div class="hero"><span class="brand-title">{APP_NAME}</span></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="subhero"><span class="brand-tagline">{APP_TAGLINE}</span>. Ask from your indexed docs and get real-time answers.</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="start-space"></div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                chunks = msg.get("chunks", [])
                for i, item in enumerate(chunks, start=1):
                    md = item.get("metadata") or {}
                    chunk_text = md.get("text", "")
                    distance = item.get("distance")
                    with st.expander(f"Chunk {i} | distance={distance:.4f}" if distance is not None else f"Chunk {i}"):
                        st.write(chunk_text[:3000] if chunk_text else "No text content.")

        if st.session_state.pending_query:
            query_to_run = st.session_state.pending_query
            
            try:
                rag = get_rag_client(persist_dir, embedding_model, llm_model, active_api_key)
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
                            top_k=top_k,
                            response_mode=response_mode,
                        )
                    )
                except (TypeError, AttributeError):
                    get_rag_client.clear()
                    rag = get_rag_client(persist_dir, embedding_model, llm_model, active_api_key)
                    st.session_state.summary = rag.search_and_summarize(query_to_run, top_k=top_k)

            st.session_state.raw_hits = rag.vectorstore.query(query_to_run, top_k=top_k)
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": st.session_state.summary,
                    "chunks": st.session_state.raw_hits,
                }
            )
            st.session_state.pending_query = None
            st.rerun()

        st.caption("Type your query and press Enter.")
        query = st.chat_input(f"Ask {APP_NAME} anything about your indexed PDFs...")

        clear_clicked = st.button("Clear", use_container_width=True)

        if clear_clicked:
            st.session_state.summary = ""
            st.session_state.raw_hits = []
            st.session_state.chat_history = []
            st.session_state.pending_query = None

        if query is not None:
            if not query.strip():
                st.warning("Please enter a query first.")
                st.stop()
            if not active_api_key:
                st.error("GROQ API key is required. Provide one from sidebar.")
                st.stop()

            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.pending_query = query
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with tab_about:
        st.subheader("About RAGNOVA")
        st.markdown(
            """
            **RAGNOVA** is a production-oriented Retrieval-Augmented Generation (RAG) assistant for
            document-grounded question answering. It combines semantic retrieval, vector search,
            and large language model generation to deliver responses that are aligned with your indexed content.

            ### System Positioning
            RAGNOVA is designed for practical AI knowledge workflows where answer quality, traceability,
            and configurable model behavior are important. The app supports interactive chat with retrieval context,
            streaming output, and configurable response style.

            ### How Knowledge Is Used
            RAGNOVA does **not** train or fine-tune foundation models on your files during runtime.

            It operates with:
            - Pre-trained embedding models for semantic retrieval
            - Pre-trained LLMs for natural language generation
            - Your indexed document chunks as context at inference time

            This architecture keeps responses grounded in retrieved evidence rather than hidden model memory updates.

            ### Data Sources and Index Layer
            Primary project paths:
            - `data/`: source documents
            - `faiss_store/`: vector index and retrieval metadata

            Supported ingestion formats:
            - PDF, TXT, CSV, XLSX, DOCX, JSON

            ### Response Generation Pipeline
            For each query, RAGNOVA executes:
            1. Query embedding generation
            2. Top-K semantic retrieval from FAISS
            3. Context-grounded prompt construction
            4. LLM response generation (streamed in UI)
            5. Evidence visibility via retrieved chunks

            ### Output Control
            - **Short Summary**: concise response targeting approximately 400-500 words
            - **Detailed**: structured, broader explanation with deeper coverage

            ### Reliability Guidance
            RAGNOVA quality depends on document quality, chunking strategy, and retrieval relevance.
            For best results, maintain clean source files, rebuild index after updates, and validate responses
            against retrieved chunks for critical use cases.

            ### Security and Privacy Notes
            - API key can be provided securely through environment or UI input
            - Retrieval executes against your indexed project data
            - Final generation uses the configured remote model provider
            """
        )

    with tab_dev:
        st.markdown('<div class="dev-shell">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="dev-card">
                <div class="dev-title">Himanshu Kumar</div>
                <div class="dev-role">AI/ML Developer | RAG Systems Builder</div>
                <p class="dev-desc">
                    Builder of <strong>RAGNOVA</strong>, focused on practical retrieval-augmented systems, reliable
                    document pipelines, and production-ready AI app UX.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dev-card">
                <div class="dev-grid">
                    <div class="dev-pill">
                        <div class="k">Project</div>
                        <div class="v">RAGNOVA</div>
                    </div>
                    <div class="dev-pill">
                        <div class="k">Domain</div>
                        <div class="v">RAG and LLM Apps</div>
                    </div>
                    <div class="dev-pill">
                        <div class="k">Primary Stack</div>
                        <div class="v">Python, LangChain, FAISS</div>
                    </div>
                    <div class="dev-pill">
                        <div class="k">Interface</div>
                        <div class="v">Streamlit</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="dev-card">
                <div class="social-grid">
                    <a class="social-link github" href="https://github.com/himanshu231204" target="_blank">
                        <div class="social-icon"><img src="https://cdn.simpleicons.org/github/ffffff" alt="GitHub logo"></div>
                        <div class="social-text">
                            <div class="social-label">GitHub</div>
                            <div class="social-handle">@himanshu231204</div>
                        </div>
                    </a>
                    <a class="social-link linkedin" href="https://www.linkedin.com/in/himanshu231204" target="_blank">
                        <div class="social-icon"><img src="https://cdn.simpleicons.org/linkedin/ffffff" alt="LinkedIn logo"></div>
                        <div class="social-text">
                            <div class="social-label">LinkedIn</div>
                            <div class="social-handle">/in/himanshu231204</div>
                        </div>
                    </a>
                    <a class="social-link twitter" href="https://twitter.com/himanshu231204" target="_blank">
                        <div class="social-icon"><img src="https://cdn.simpleicons.org/x/ffffff" alt="Twitter/X logo"></div>
                        <div class="social-text">
                            <div class="social-label">Twitter / X</div>
                            <div class="social-handle">@himanshu231204</div>
                        </div>
                    </a>
                    <a class="social-link email" href="mailto:himanshu231204@gmail.com">
                        <div class="social-icon"><img src="https://cdn.simpleicons.org/gmail/ffffff" alt="Gmail logo"></div>
                        <div class="social-text">
                            <div class="social-label">Email</div>
                            <div class="social-handle">himanshu231204@gmail.com</div>
                        </div>
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # FOOTER
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
