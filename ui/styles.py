"""Visual theme for the Streamlit interface."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        letter-spacing: -0.01em;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e6e9f2;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        color: #5b6478;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #1f2430;
        font-weight: 700;
    }
    .dpp-hero {
        background: linear-gradient(135deg, #2f6fed 0%, #1b49b8 100%);
        color: #ffffff;
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.25rem;
    }
    .dpp-hero h1 {
        color: #ffffff;
        font-size: 1.8rem;
        margin: 0 0 0.35rem 0;
    }
    .dpp-hero p {
        color: #dbe5ff;
        margin: 0;
        font-size: 1rem;
    }
    .dpp-section {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5b6478;
        margin: 1.2rem 0 0.5rem 0;
    }
    .dpp-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: #eaf0ff;
        color: #2f6fed;
        margin-right: 0.35rem;
    }
    .dpp-muted {
        color: #5b6478;
        font-size: 0.9rem;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stSidebarNav"] a {
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.4rem 0.9rem;
    }
</style>
"""


def apply_theme() -> None:
    """Inject the application CSS once per run."""
    st.markdown(_CSS, unsafe_allow_html=True)
