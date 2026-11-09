"""Dashboard page: KPIs and interactive charts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import streamlit as st

from data_processor import analyze_dataframe, missing_summary
from data_processor.analyzer import categorical_columns, numeric_columns
from data_processor.models import ColumnProfile
from ui import state
from ui.components import empty_state, metrics, page_header, render_plotly, section

_PRIMARY = "#2f6fed"
_CHART_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "colorway": [_PRIMARY, "#6b8afd", "#9db4ff", "#c3d1ff"],
}


def render() -> None:
    """Render key indicators and charts for the current data view."""
    page_header(
        "Dashboard",
        "Key indicators, missing values and distributions for the current data view.",
    )

    if not state.has_dataset():
        empty_state("Upload a CSV file on the **Upload** page to see the dashboard.")
        return

    working = state.get_working()
    current = state.get_result_dataframe()
    source = state.get_source()
    if working is None or current is None or source is None:
        empty_state("No dataset is loaded.")
        return

    analysis = analyze_dataframe(current)
    active_filters = len(state.get_filters())
    active_sorts = len(state.get_sort_specs())

    metrics(
        [
            ("Rows", analysis.rows, "Rows in the current view"),
            ("Columns", analysis.columns, None),
            ("Missing cells", analysis.total_missing, "Across the whole view"),
            ("Duplicate rows", analysis.duplicate_rows, "Full-row duplicates"),
        ]
    )
    metrics(
        [
            ("Original rows", source.row_count, None),
            ("Rows removed", source.row_count - len(working), "By cleaning operations"),
            ("Missing %", analysis.missing_pct, "Share of all cells"),
            ("Active filters", active_filters, f"{active_sorts} sort key(s)"),
        ]
    )

    if active_filters:
        st.caption(
            f"Showing {analysis.rows:,} of {len(working):,} rows after "
            f"{active_filters} filter condition(s)."
        )

    st.divider()
    _render_missing_chart(current)
    st.divider()
    _render_numeric_section(current, analysis.numeric_stats)
    st.divider()
    _render_categorical_section(current)
    st.divider()
    _render_column_profiles(analysis.column_profiles)


def _render_missing_chart(current: pd.DataFrame) -> None:
    section("Missing values")
    summary = missing_summary(current)
    affected = summary[summary["missing"] > 0] if not summary.empty else summary
    if affected.empty:
        st.success("No missing values in the current data view.", icon="✅")
        return

    chart_data = affected.sort_values("missing_pct", ascending=True)
    figure = px.bar(
        chart_data,
        x="missing_pct",
        y="column",
        orientation="h",
        text="missing",
        labels={"missing_pct": "Missing (%)", "column": "Column", "missing": "Missing"},
        color_discrete_sequence=[_PRIMARY],
    )
    figure.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
    figure.update_layout(
        **_CHART_LAYOUT,
        height=max(260, 38 * len(chart_data) + 110),
        xaxis_title="Missing values (%)",
        yaxis_title="",
    )
    render_plotly(figure)


def _render_numeric_section(
    current: pd.DataFrame, numeric_stats: dict[str, dict[str, float]]
) -> None:
    section("Numeric distribution")
    columns = numeric_columns(current)
    if not columns:
        st.info("The current data view has no numeric columns.")
        return

    left, right = st.columns([1, 3])
    selected = left.selectbox("Numeric column", columns, key="dash_numeric_column")
    series = pd.to_numeric(current[selected], errors="coerce").dropna()
    if series.empty:
        st.info("The selected column has no numeric values to plot.")
        return

    bins = int(min(60, max(10, series.nunique())))
    figure = px.histogram(
        x=series,
        nbins=bins,
        labels={"x": selected, "count": "Frequency"},
        color_discrete_sequence=[_PRIMARY],
    )
    figure.update_layout(
        **_CHART_LAYOUT,
        height=340,
        xaxis_title=selected,
        yaxis_title="Frequency",
        showlegend=False,
    )
    right.plotly_chart(figure, theme="streamlit", width="stretch")

    statistics = numeric_stats.get(selected, {})
    if statistics:
        stat_columns = st.columns(5)
        for column, (label, key) in zip(
            stat_columns,
            [
                ("Count", "count"),
                ("Mean", "mean"),
                ("Median", "50%"),
                ("Min", "min"),
                ("Max", "max"),
            ],
            strict=True,
        ):
            value = statistics.get(key)
            rendered = f"{value:,.2f}" if isinstance(value, float) else "—"
            column.metric(label, rendered)


def _render_categorical_section(current: pd.DataFrame) -> None:
    section("Categorical distribution")
    columns = categorical_columns(current)
    if not columns:
        st.info("The current data view has no categorical columns.")
        return

    left, right = st.columns([1, 3])
    selected = left.selectbox("Categorical column", columns, key="dash_category_column")
    values = current[selected].astype("string").fillna("(missing)")
    counts = values.value_counts().head(15)
    if counts.empty:
        st.info("The selected column has no values to plot.")
        return

    chart_data = (
        counts.rename_axis("value")
        .reset_index(name="count")
        .assign(value=lambda frame: frame["value"].astype(str))
    )
    figure = px.bar(
        chart_data,
        x="value",
        y="count",
        labels={"value": selected, "count": "Rows"},
        color_discrete_sequence=[_PRIMARY],
    )
    figure.update_layout(
        **_CHART_LAYOUT,
        height=340,
        xaxis_title=selected,
        yaxis_title="Rows",
        showlegend=False,
    )
    figure.update_xaxes(tickangle=-30)
    right.plotly_chart(figure, theme="streamlit", width="stretch")

    unique = int(current[selected].nunique(dropna=True))
    top_share = counts.iloc[0] / max(len(current), 1) * 100.0
    st.caption(
        f"{unique:,} unique value(s); the most frequent value covers "
        f"{top_share:.1f}% of rows (top 15 shown)."
    )


def _render_column_profiles(profiles: Sequence[ColumnProfile]) -> None:
    section("Column profile")
    table = pd.DataFrame(
        {
            "column": [profile.name for profile in profiles],
            "dtype": [profile.dtype for profile in profiles],
            "non-null": [profile.non_null for profile in profiles],
            "missing": [profile.missing for profile in profiles],
            "missing %": [profile.missing_pct for profile in profiles],
            "unique": [profile.unique for profile in profiles],
        }
    )
    st.dataframe(table, width="stretch", hide_index=True)
