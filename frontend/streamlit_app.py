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
        fig = px.histogram(dataframe, x=spec["x"], nbins=spec.get("bins"), title=title)
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
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
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
                st.session_state.profile = fetch_profile(payload["session_id"])
                st.session_state.suggestions = fetch_suggestions(payload["session_id"])
            except httpx.HTTPStatusError as exc:
                st.error(exc.response.json().get("detail", "Could not load dataset profile or suggestions."))
            except httpx.RequestError:
                st.error("Dataset uploaded, but the profile or suggestions request failed.")
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
            if message.get("table"):
                st.dataframe(pd.DataFrame(message["table"]), use_container_width=True)
            if message.get("chart_spec"):
                st.json(message["chart_spec"])
            if message.get("tool_trace"):
                with st.expander("Tool trace"):
                    st.json(message["tool_trace"])

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
            if assistant_message.get("table"):
                st.dataframe(pd.DataFrame(assistant_message["table"]), use_container_width=True)
            if assistant_message.get("chart_spec"):
                st.json(assistant_message["chart_spec"])
            if assistant_message.get("tool_trace"):
                with st.expander("Tool trace"):
                    st.json(assistant_message["tool_trace"])
