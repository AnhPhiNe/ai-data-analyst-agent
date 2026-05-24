import os

import httpx
import pandas as pd
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="DA",
    layout="wide",
)

st.title("AI Data Analyst Agent")
st.caption("MVP for learning AI agents and data analysis with FastAPI, Streamlit, and safe pandas tools.")

with st.sidebar:
    st.header("Project Status")
    st.write("Phase 2: Upload + Session")
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
            st.success("Dataset uploaded successfully.")

if "upload_result" in st.session_state:
    result = st.session_state.upload_result
    left, right = st.columns(2)
    left.metric("Rows", result["rows"])
    right.metric("Columns", result["columns"])

    st.write("Columns")
    st.code(", ".join(result["column_names"]))

    st.write("Preview")
    st.dataframe(pd.DataFrame(result["preview"]), use_container_width=True)
