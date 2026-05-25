import io
import math
import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("session_token")
    return {"X-Session-Token": str(token)} if token else {}


def fetch_profile(session_id: str) -> dict[str, object]:
    response = httpx.get(f"{BACKEND_URL}/datasets/{session_id}/profile", headers=auth_headers(), timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_suggestions(session_id: str) -> dict[str, object]:
    response = httpx.get(f"{BACKEND_URL}/datasets/{session_id}/suggestions", headers=auth_headers(), timeout=45.0)
    response.raise_for_status()
    return response.json()


def fetch_auto_analysis(session_id: str) -> dict[str, object]:
    response = httpx.get(f"{BACKEND_URL}/datasets/{session_id}/auto-analysis", headers=auth_headers(), timeout=45.0)
    response.raise_for_status()
    return response.json()


def send_chat_question(session_id: str, question: str) -> dict[str, object]:
    response = httpx.post(
        f"{BACKEND_URL}/chat/query",
        json={"session_id": session_id, "question": question},
        headers=auth_headers(),
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()


def parse_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(content)
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def render_distribution_chart(spec: dict[str, object], chart_key: str | None = None) -> None:
    data = pd.DataFrame(spec["data"])
    chart_type = spec["chart_type"]

    if data.empty:
        st.caption("No chart data available.")
        return

    if chart_type == "histogram":
        fig = px.bar(data, x="bin", y="count", labels={"bin": spec["x_label"], "count": spec["y_label"]})
    else:
        fig = px.bar(data, x="category", y="count", labels={"category": spec["x_label"], "count": spec["y_label"]})

    fig.update_layout(height=320, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def render_chart_spec(dataframe: pd.DataFrame, spec: dict[str, object], chart_key: str | None = None) -> None:
    chart_type = spec.get("chart_type")
    title = spec.get("title") or default_chart_title(spec)

    if dataframe.empty:
        st.caption("No chart data available.")
        return

    if chart_type == "bar":
        fig = px.bar(dataframe, x=spec["x"], y=spec["y"], color=spec.get("color"), title=title)
    elif chart_type == "line":
        fig = px.line(dataframe, x=spec["x"], y=spec["y"], color=spec.get("color"), title=title)
    elif chart_type == "histogram":
        bins = int(spec.get("bins") or histogram_bin_count(dataframe, str(spec["x"])))
        fig = px.histogram(
            dataframe,
            x=spec["x"],
            nbins=bins,
            title=title,
            labels={spec["x"]: spec["x"], "count": "Count"},
            opacity=0.85,
        )
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(bargap=0.05)
    elif chart_type == "scatter":
        fig = px.scatter(dataframe, x=spec["x"], y=spec["y"], color=spec.get("color"), title=title)
    elif chart_type == "box":
        fig = px.box(dataframe, x=spec.get("x"), y=spec["y"], color=spec.get("color"), title=title)
    elif chart_type == "pie":
        fig = px.pie(dataframe, names=spec["names"], values=spec.get("values"), title=title)
    elif chart_type == "correlation_heatmap":
        columns = spec["columns"]
        correlation = dataframe[columns].corr(numeric_only=True)
        labels = correlation.round(2).astype(str).values if len(columns) <= 10 else None
        heatmap_height = max(420, min(720, 72 * len(columns)))
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                text=labels,
                texttemplate="%{text}" if labels is not None else None,
                hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r=%{z:.3f}<extra></extra>",
                colorscale=[
                    [0.0, "#b2182b"],
                    [0.25, "#ef8a62"],
                    [0.5, "#f7f7f7"],
                    [0.75, "#67a9cf"],
                    [1.0, "#2166ac"],
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
            plot_bgcolor="white",
        )
    else:
        st.error("Unsupported chart type.")
        return

    fig.update_layout(margin={"l": 20, "r": 20, "t": 56 if title else 24, "b": 48})
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


def render_insight_dashboard(profile: dict[str, object], auto_analysis: dict[str, object], suggestions: dict[str, object]) -> None:
    st.subheader("Insight Dashboard")

    overview = auto_analysis.get("overview", {})
    quality = auto_analysis.get("data_quality", {})
    numeric = auto_analysis.get("numeric_highlights", [])
    categorical = auto_analysis.get("categorical_highlights", [])
    correlations = auto_analysis.get("correlation_highlights", [])

    rows = overview.get("rows", profile.get("rows", 0))
    columns = overview.get("columns", profile.get("columns", 0))
    missing_cells = quality.get("total_missing_cells", 0)
    chart_count = len(auto_analysis.get("recommended_charts", []))

    a, b, c, d = st.columns(4)
    a.metric("Rows", rows)
    b.metric("Columns", columns)
    c.metric("Missing cells", missing_cells)
    d.metric("Recommended charts", chart_count)

    charts = auto_analysis.get("recommended_charts", [])
    dataset_frame = st.session_state.get("dataset_frame")
    if charts and isinstance(dataset_frame, pd.DataFrame):
        st.write("Visual analysis")
        chart_columns = st.columns(min(2, len(charts)))
        for index, item in enumerate(charts[:4]):
            with chart_columns[index % len(chart_columns)]:
                st.write(str(item.get("title", "Recommended chart")))
                spec = item.get("chart_spec")
                if isinstance(spec, dict):
                    render_chart_spec(dataset_frame, spec, chart_key=f"dashboard-main-chart-{index}")
                st.caption(str(item.get("reason", "")))

    story_col, quality_col = st.columns([1.25, 1])
    with story_col:
        st.write("Key insights")
        insight_items = _dashboard_insight_items(auto_analysis, suggestions)
        if not insight_items:
            st.info("No insight highlights available yet.")
        for insight in insight_items:
            st.markdown(f"- {insight}")

    with quality_col:
        st.write("Data quality")
        missing = pd.DataFrame(quality.get("top_missing_columns", []))
        if missing.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(
                missing[["name", "missing_count", "missing_percent"]],
                use_container_width=True,
                hide_index=True,
            )

    highlight_tabs = st.tabs(["Numeric", "Categories", "Relationships", "Ask next"])
    with highlight_tabs[0]:
        numeric_frame = pd.DataFrame(numeric)
        if numeric_frame.empty:
            st.info("No numeric columns detected.")
        else:
            st.dataframe(numeric_frame, use_container_width=True, hide_index=True)

    with highlight_tabs[1]:
        categorical_frame = pd.DataFrame(categorical)
        if categorical_frame.empty:
            st.info("No categorical columns detected.")
        else:
            st.dataframe(categorical_frame, use_container_width=True, hide_index=True)

    with highlight_tabs[2]:
        correlation_frame = pd.DataFrame(correlations)
        if correlation_frame.empty:
            st.info("Not enough numeric columns for relationship highlights.")
        else:
            st.dataframe(correlation_frame, use_container_width=True, hide_index=True)

    with highlight_tabs[3]:
        questions = _dedupe_texts(
            [str(item) for item in auto_analysis.get("next_questions", [])]
            + [str(item) for item in suggestions.get("questions", [])]
        )
        for index, question_item in enumerate(questions[:8]):
            if st.button(question_item, key=f"dashboard-question-{index}", use_container_width=True):
                st.session_state.pending_chat_question = question_item
                st.rerun()


def _dashboard_insight_items(auto_analysis: dict[str, object], suggestions: dict[str, object]) -> list[str]:
    items: list[str] = []
    items.extend(str(item) for item in suggestions.get("insights", [])[:3])

    numeric = auto_analysis.get("numeric_highlights", [])
    if numeric:
        item = numeric[0]
        items.append(
            f"{item.get('column')} has the widest numeric range: "
            f"{item.get('min')} to {item.get('max')}."
        )

    categorical = auto_analysis.get("categorical_highlights", [])
    if categorical:
        item = categorical[0]
        items.append(
            f"{item.get('column')} is led by \"{item.get('top_value')}\" "
            f"at about {item.get('percent')}% of rows."
        )

    correlations = auto_analysis.get("correlation_highlights", [])
    if correlations:
        item = correlations[0]
        items.append(
            f"Strongest numeric relationship found: {item.get('column_a')} vs "
            f"{item.get('column_b')} (r={item.get('correlation')})."
        )
    return _dedupe_texts(items)[:6]


def _dedupe_texts(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = item.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def render_chat_artifacts(message: dict[str, object], message_index: int | None = None) -> None:
    message_id = message.get("id") or message_index or "current"
    if message.get("table"):
        st.dataframe(pd.DataFrame(message["table"]), use_container_width=True)
    if message.get("chart_spec"):
        dataset_frame = st.session_state.get("dataset_frame")
        if isinstance(dataset_frame, pd.DataFrame):
            render_chart_spec(dataset_frame, message["chart_spec"], chart_key=f"chat-chart-{message_id}")
        else:
            st.json(message["chart_spec"])
    if message.get("tool_trace"):
        with st.expander("Tool trace"):
            st.json(message["tool_trace"])
    if message.get("clarification_options"):
        columns = st.columns(min(3, len(message["clarification_options"])))
        for index, option in enumerate(message["clarification_options"]):
            with columns[index % len(columns)]:
                if st.button(str(option), key=f"clarify-option-{message_id}-{index}", use_container_width=True):
                    st.session_state.pending_chat_question = str(option)
                    st.rerun()


def clear_dataset_state() -> None:
    for key in (
        "session_id",
        "session_token",
        "upload_result",
        "profile",
        "suggestions",
        "auto_analysis",
        "dataset_frame",
        "chat_messages",
        "pending_chat_question",
    ):
        st.session_state.pop(key, None)


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="DA",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption("Safe tabular data analysis with FastAPI, Streamlit, pandas tools, Plotly, and optional Gemini.")

with st.sidebar:
    st.header("Project Status")
    st.write("Portfolio build: end-to-end agent workflow")
    st.write(f"Backend: `{BACKEND_URL}`")
    if "session_id" in st.session_state:
        st.write(f"Session: `{st.session_state.session_id}`")

st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

upload_clicked = st.button("Upload", type="primary", disabled=uploaded_file is None)

if upload_clicked and uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            uploaded_file.type or "application/octet-stream",
        )
    }

    with st.spinner("Uploading and profiling dataset..."):
        try:
            response = httpx.post(f"{BACKEND_URL}/datasets/upload", files=files, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", "Upload failed.")
            st.error(detail)
        except httpx.RequestError:
            st.error("Could not reach the backend. Start FastAPI first, then try again.")
        else:
            payload = response.json()
            clear_dataset_state()
            st.session_state.session_id = payload["session_id"]
            st.session_state.session_token = payload.get("session_token")
            st.session_state.upload_result = payload
            try:
                st.session_state.dataset_frame = parse_uploaded_dataframe(uploaded_file.name, file_bytes)
            except Exception:
                st.session_state.dataset_frame = pd.DataFrame()
            try:
                st.session_state.profile = fetch_profile(payload["session_id"])
                st.session_state.suggestions = fetch_suggestions(payload["session_id"])
                st.session_state.auto_analysis = fetch_auto_analysis(payload["session_id"])
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get("detail", "Could not load dataset profile or suggestions.")
                clear_dataset_state()
                st.error(
                    f"{detail} Upload request succeeded, but the analysis session could not be loaded. "
                    "If the backend just restarted, please upload once more."
                )
            except httpx.RequestError:
                clear_dataset_state()
                st.error("Dataset upload reached the backend, but profile/suggestions could not be loaded.")
            else:
                st.success("Dataset uploaded successfully.")

if "profile" in st.session_state:
    profile = st.session_state.profile
    if "auto_analysis" in st.session_state and "suggestions" in st.session_state:
        render_insight_dashboard(profile, st.session_state.auto_analysis, st.session_state.suggestions)

    with st.expander("Raw dataset profile", expanded=False):
        left, middle, right = st.columns(3)
        left.metric("Rows", profile["rows"])
        middle.metric("Columns", profile["columns"])
        right.metric("Missing Cells", sum(column["missing_count"] for column in profile["dtypes"]))

        preview_tab, schema_tab, missing_tab, numeric_tab, category_tab, chart_tab = st.tabs(
            ["Preview", "Schema", "Missing", "Numeric", "Categories", "Distributions"]
        )

        with preview_tab:
            st.dataframe(pd.DataFrame(profile["preview"]), use_container_width=True)

        with schema_tab:
            st.dataframe(pd.DataFrame(profile["dtypes"]), use_container_width=True)

        with missing_tab:
            missing = pd.DataFrame(profile["missing_values"])
            if missing.empty:
                st.success("No missing values detected.")
            else:
                st.dataframe(missing, use_container_width=True)

        with numeric_tab:
            numeric_summary = pd.DataFrame(profile["numeric_summary"])
            if numeric_summary.empty:
                st.info("No numeric columns detected.")
            else:
                st.dataframe(numeric_summary, use_container_width=True)

        with category_tab:
            if not profile["top_categories"]:
                st.info("No categorical columns detected.")
            for item in profile["top_categories"]:
                st.write(f"Top values for `{item['column']}`")
                st.dataframe(pd.DataFrame(item["values"]), use_container_width=True)

        with chart_tab:
            if not profile["distributions"]:
                st.info("No distribution charts available.")
            for index, spec in enumerate(profile["distributions"]):
                st.write(f"`{spec['column']}`")
                render_distribution_chart(spec, chart_key=f"profile-distribution-{index}-{spec['column']}")

    st.subheader("Chat")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for index, message in enumerate(st.session_state.chat_messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_chat_artifacts(message, message_index=index)

    pending_question = st.session_state.pop("pending_chat_question", None)
    typed_question = st.chat_input("Ask a question about the uploaded dataset")
    question = pending_question or typed_question
    if question:
        user_message = {"id": len(st.session_state.chat_messages), "role": "user", "content": question}
        st.session_state.chat_messages.append(user_message)
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Analyzing question..."):
            try:
                chat_response = send_chat_question(st.session_state.session_id, question)
            except httpx.HTTPStatusError as exc:
                content = exc.response.json().get("detail", "Chat request failed.")
                assistant_message = {"id": len(st.session_state.chat_messages), "role": "assistant", "content": content}
            except httpx.RequestError:
                assistant_message = {
                    "id": len(st.session_state.chat_messages),
                    "role": "assistant",
                    "content": "Could not reach the backend.",
                }
            else:
                assistant_message = {
                    "id": len(st.session_state.chat_messages),
                    "role": "assistant",
                    "content": chat_response["answer"],
                    "table": chat_response.get("table"),
                    "chart_spec": chat_response.get("chart_spec"),
                    "tool_trace": chat_response.get("tool_trace"),
                    "clarification_options": chat_response.get("clarification_options"),
                }

        st.session_state.chat_messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.write(assistant_message["content"])
            render_chat_artifacts(assistant_message, message_index=int(assistant_message["id"]))
