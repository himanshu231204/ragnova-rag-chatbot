"""Styling and Theme Management"""

from pathlib import Path
import streamlit as st


def get_light_mode_overrides() -> str:
    """Generate CSS overrides for light mode."""
    return """
        :root {
            --bg: #f7f9fc;
            --panel: #ffffff;
            --border: #e0e8f0;
            --text: #0f172a;
            --muted: #6b7c94;
            --accent: #00a67e;
        }
        .stApp {
            background: radial-gradient(circle at 20% 20%, #f3f8ff 0%, #eaf1fb 45%, #f7f9fc 100%);
            color: #0f172a !important;
        }
        * {
            color: #0f172a !important;
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
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] svg {
            filter: none !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: linear-gradient(90deg, #00a67e 0%, #0f8fb0 100%) !important;
            color: #ffffff !important;
        }
        .settings-card {
            background: linear-gradient(145deg, #ffffff 0%, #edf3ff 60%, #eaf7ff 100%);
            border: 1px solid #b9ccec;
            color: #0f172a !important;
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
            color: #0f172a;
        }
        .dev-title, .dev-role, .dev-desc { color: #1b2a45; }
        .dev-pill {
            border: 1px solid #bfd3ee;
            background: #f7faff;
            color: #1a2a45;
        }
        .dev-pill .k { color: #5a6c8d; }
        .dev-pill .v { color: #1a2a45; }
        .social-link {
            color: #0f172a !important;
        }
        .dev-shell .social-link,
        .dev-shell .social-link .social-label {
            color: #f3f7ff !important;
        }
        .dev-shell .social-link .social-handle {
            color: #cbd8ee !important;
        }
        .dev-shell .social-link .social-icon {
            background: rgba(255, 255, 255, 0.14) !important;
        }
        [data-testid="stChatMessage"] {
            color: #0f172a !important;
        }
        [data-testid="stChatMessageContent"] {
            color: #0f172a !important;
        }
        .stMarkdown {
            color: #0f172a !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
        }
        p, span, div, label {
            color: #0f172a !important;
        }
    """


def inject_styles(is_light_mode: bool) -> None:
    """Apply CSS styling based on theme mode."""
    css_path = Path("assets/ragnova.css")
    css_text = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    mode_overrides = get_light_mode_overrides() if is_light_mode else ""
    css_text = css_text.replace("/* MODE_OVERRIDES */", mode_overrides)
    st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
