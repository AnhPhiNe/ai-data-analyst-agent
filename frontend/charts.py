import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_distribution_chart(
    spec: dict[str, object], chart_key: str | None = None
) -> None:
    data = pd.DataFrame(spec["data"])
    chart_type = spec["chart_type"]

    if data.empty:
        st.caption("No chart data available.")
        return

    if chart_type == "histogram":
        fig = px.bar(
            data,
            x="bin",
            y="count",
            labels={"bin": spec["x_label"], "count": spec["y_label"]},
            color_discrete_sequence=["#4f46e5"],
        )
    else:
        fig = px.bar(
            data,
            x="category",
            y="count",
            labels={"category": spec["x_label"], "count": spec["y_label"]},
            color_discrete_sequence=["#06b6d4"],
        )

    fig.update_layout(
        height=320,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        template="plotly_white",
        font_family="Outfit, sans-serif",
        font_color="#2b2d42",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_chart_spec(
    dataframe: pd.DataFrame, spec: dict[str, object], chart_key: str | None = None
) -> None:
    chart_type = spec.get("chart_type")
    title = spec.get("title") or default_chart_title(spec)
    colors = ["#4f46e5", "#06b6d4", "#ec4899", "#f59e0b", "#10b981"]

    if dataframe.empty:
        st.caption("No chart data available.")
        return

    if chart_type == "bar":
        x_col = str(spec["x"])
        y_col = str(spec["y"])
        if (
            not pd.api.types.is_numeric_dtype(dataframe[x_col])
            or dataframe[x_col].nunique(dropna=True) <= 20
        ):
            agg_df = dataframe.groupby(x_col, as_index=False)[y_col].mean()
            agg_df = agg_df.sort_values(by=y_col, ascending=False)
            if pd.api.types.is_numeric_dtype(agg_df[x_col]):
                agg_df[x_col] = agg_df[x_col].astype(str)
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                color=spec.get("color"),
                title=title or f"Average {y_col} by {x_col}",
                color_discrete_sequence=colors,
            )
            fig.update_layout(yaxis_title=f"Average {y_col}")
        else:
            fig = px.bar(
                dataframe,
                x=x_col,
                y=y_col,
                color=spec.get("color"),
                title=title,
                color_discrete_sequence=colors,
            )
    elif chart_type == "line":
        fig = px.line(
            dataframe,
            x=spec["x"],
            y=spec["y"],
            color=spec.get("color"),
            title=title,
            color_discrete_sequence=colors,
        )
    elif chart_type == "histogram":
        bins = int(spec.get("bins") or histogram_bin_count(dataframe, str(spec["x"])))
        fig = px.histogram(
            dataframe,
            x=spec["x"],
            nbins=bins,
            title=title,
            labels={spec["x"]: spec["x"], "count": "Count"},
            opacity=0.85,
            color_discrete_sequence=["#4f46e5"],
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(bargap=0.05)
    elif chart_type == "scatter":
        fig = px.scatter(
            dataframe,
            x=spec["x"],
            y=spec["y"],
            color=spec.get("color"),
            title=title,
            color_discrete_sequence=colors,
        )
    elif chart_type == "box":
        fig = px.box(
            dataframe,
            x=spec.get("x"),
            y=spec["y"],
            color=spec.get("color"),
            title=title,
            color_discrete_sequence=colors,
        )
    elif chart_type == "pie":
        fig = px.pie(
            dataframe,
            names=spec["names"],
            values=spec.get("values"),
            title=title,
            color_discrete_sequence=colors,
        )
    elif chart_type == "correlation_heatmap":
        columns = spec["columns"]
        correlation = dataframe[columns].corr(numeric_only=True)
        labels = correlation.round(2).astype(str).values if len(columns) <= 10 else None
        heatmap_height = 420
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                text=labels,
                texttemplate="%{text}" if labels is not None else None,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
                colorscale=[
                    [0.0, "#e11d48"],
                    [0.25, "#fda4af"],
                    [0.5, "#f8fafc"],
                    [0.75, "#bae6fd"],
                    [1.0, "#0284c7"],
                ],
                zmin=-1,
                zmid=0,
                zmax=1,
                colorbar={"title": "r", "thickness": 14, "len": 0.82},
                xgap=1,
                ygap=1,
            )
        )
        fig.update_xaxes(side="bottom", tickangle=-35, automargin=True)
        fig.update_yaxes(autorange="reversed", automargin=True)
        fig.update_layout(
            title=title,
            height=heatmap_height,
            xaxis_title=None,
            yaxis_title=None,
            plot_bgcolor="#f8fafc",
        )
    else:
        st.error("Unsupported chart type.")
        return

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 56 if title else 24, "b": 48},
        template="plotly_white",
        font_family="Outfit, sans-serif",
        font_color="#2b2d42",
        plot_bgcolor="#f8fafc",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    if chart_type != "correlation_heatmap":
        fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def default_chart_title(spec: dict[str, object]) -> str | None:
    chart_type = spec.get("chart_type")
    if chart_type == "correlation_heatmap":
        return "Correlation heatmap"
    if chart_type == "histogram" and spec.get("x"):
        return f"Distribution of {spec['x']}"
    if chart_type in {"bar", "line", "scatter"} and spec.get("x") and spec.get("y"):
        return f"{spec['y']} by {spec['x']}"
    return None


def histogram_bin_count(dataframe: pd.DataFrame, column: str) -> int:
    series = dataframe[column].dropna()
    row_count = int(series.count())
    unique_count = int(series.nunique())
    if row_count <= 0 or unique_count <= 0:
        return 10
    if unique_count <= 20:
        return max(1, unique_count)
    rice_bins = math.ceil(2 * (row_count ** (1 / 3)))
    return max(8, min(50, unique_count, rice_bins))
