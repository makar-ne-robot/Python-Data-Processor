"""Reusable Streamlit widgets shared by the pages."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def page_header(title: str, subtitle: str = "") -> None:
    """Render a page title with an optional subtitle."""
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<span class='dpp-muted'>{subtitle}</span>", unsafe_allow_html=True)
    st.write("")


def empty_state(message: str) -> None:
    """Render a friendly empty-state placeholder."""
    st.info(message, icon="ℹ️")


def section(title: str) -> None:
    """Render a small uppercase section label."""
    st.markdown(f"<div class='dpp-section'>{title}</div>", unsafe_allow_html=True)


def metrics(items: Sequence[tuple[str, object, str | None]]) -> None:
    """Render a responsive row of metric cards.

    Args:
        items: Tuples of ``(label, value, help_text)``.  ``help_text`` may be
            ``None``.
    """
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, _format_metric(value), help=help_text)


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_table(dataframe: pd.DataFrame, *, height: int | None = None) -> None:
    """Display a DataFrame, optionally with a fixed height."""
    if height is None:
        st.dataframe(dataframe, width="stretch")
    else:
        st.dataframe(dataframe, width="stretch", height=height)


def render_plotly(figure: go.Figure) -> None:
    """Display a Plotly figure using the modern Streamlit width API."""
    st.plotly_chart(figure, theme="streamlit")
