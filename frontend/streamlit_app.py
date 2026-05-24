import io
import math
import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def fetch_profile(session_id: str) -> dict[str, object]:
    response = httpx.get(f"{BACKEND_URL}/datasets/{session_id}/profile", timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_suggestions(session_id: str) -> dict[str, object]:
    response = httpx.get(f"{BACKEND_URL}/datasets/{session_id}/suggestions", timeout=45.0)
    response.raise_for_status()
    return response.json()


def send_chat_question(session_id: str, question: str) -> dict[str, object]:
    response = httpx.post(
        f"{BACKEND_URL}/chat/query",
        json={"session_id": session_id, "question": question},
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()


def parse_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(content)
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def render_distribution_chart(spec: dict[str, object]) -> None:
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
    st.plotly_chart(fig, use_container_width=True)


def render_chart_spec(dataframe: pd.DataFrame, spec: dict[str, object]) -> None:
    chart_type = spec.get("chart_type")
    title = spec.get("title")

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
        fig = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
            )
        )
        fig.update_layout(title=title)
    else:
        st.error("Unsupported chart type.")
        return

    fig.update_layout(height=420, margin={"l": 20, "r": 20, "t": 48 if title else 20, "b": 20})
    st.plotly_chart(fig, use_container_width=True)


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


def render_chat_artifacts(message: dict[str, object]) -> None:
    if message.get("table"):
        st.dataframe(pd.DataFrame(message["table"]), use_container_width=True)
    if message.get("chart_spec"):
        dataset_frame = st.session_state.get("dataset_frame")
        if isinstance(dataset_frame, pd.DataFrame):
            render_chart_spec(dataset_frame, message["chart_spec"])
        else:
            st.json(message["chart_spec"])
    if message.get("tool_trace"):
        with st.expander("Tool trace"):
            st.json(message["tool_trace"])


def clear_dataset_state() -> None:
    for key in ("session_id", "upload_result", "profile", "suggestions", "dataset_frame", "chat_messages"):
        st.session_state.pop(key, None)


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="DA",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption("MVP for learning AI agents and data analysis with FastAPI, Streamlit, and safe pandas tools.")

with st.sidebar:
    st.header("Project Status")
    st.write("Phase 10: Suggested Questions + Insights")
    st.write(f"Backend: `{BACKEND_URL}`")
    if "session_id" in st.session_state:
        st.write(f"Session: `{st.session_state.session_id}`")

st.subheader("Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])

if uploaded_file is not None:
    if st.button("Upload", type="primary"):
        file_bytes = uploaded_file.getvalue()
        files = {
            "file": (
                uploaded_file.name,
                file_bytes,
                uploaded_file.type or "application/octet-stream",
            )
        }

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
            st.session_state.session_id = payload["session_id"]
            st.session_state.upload_result = payload
            try:
                st.session_state.dataset_frame = parse_uploaded_dataframe(uploaded_file.name, file_bytes)
            except Exception:
                st.session_state.dataset_frame = pd.DataFrame()
            try:
                st.session_state.profile = fetch_profile(payload["session_id"])
                st.session_state.suggestions = fetch_suggestions(payload["session_id"])
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
    st.subheader("Dataset Profile")

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
        for spec in profile["distributions"]:
            st.write(f"`{spec['column']}`")
            render_distribution_chart(spec)

    if "suggestions" in st.session_state:
        suggestions = st.session_state.suggestions
        st.subheader("Suggested Analysis")
        insight_col, question_col = st.columns(2)

        with insight_col:
            st.write("Light insights")
            if not suggestions.get("insights"):
                st.info("No suggested insights available.")
            for insight in suggestions.get("insights", []):
                st.markdown(f"- {insight}")

        with question_col:
            st.write("Suggested questions")
            if not suggestions.get("questions"):
                st.info("No suggested questions available.")
            for suggested_question in suggestions.get("questions", []):
                st.markdown(f"- {suggested_question}")

    st.subheader("Chat")
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_chat_artifacts(message)

    question = st.chat_input("Ask a question about the uploaded dataset")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        try:
            chat_response = send_chat_question(st.session_state.session_id, question)
        except httpx.HTTPStatusError as exc:
            content = exc.response.json().get("detail", "Chat request failed.")
            assistant_message = {"role": "assistant", "content": content}
        except httpx.RequestError:
            assistant_message = {"role": "assistant", "content": "Could not reach the backend."}
        else:
            assistant_message = {
                "role": "assistant",
                "content": chat_response["answer"],
                "table": chat_response.get("table"),
                "chart_spec": chat_response.get("chart_spec"),
                "tool_trace": chat_response.get("tool_trace"),
            }

        st.session_state.chat_messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.write(assistant_message["content"])
            render_chat_artifacts(assistant_message)
