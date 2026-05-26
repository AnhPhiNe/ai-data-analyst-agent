import io
import sys
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.api import (  # noqa: E402
    BACKEND_URL,
    fetch_profile,
    fetch_suggestions,
    fetch_auto_analysis,
)
from frontend.components import (  # noqa: E402
    render_dashboard_tab,
    render_data_explorer_tab,
    render_chat_tab,
)


def parse_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(content)
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


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
        "uploaded_filename",
    ):
        st.session_state.pop(key, None)


st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide",
)

# Inject ultra-premium modern CSS stylesheet
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #fafbfe !important;
    }

    /* Modern scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }

    /* Sidebar Glassmorphic styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%) !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    /* Headers styling */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.5px !important;
    }

    /* Custom main gradient title */
    .main-title {
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-bottom: 6px !important;
        background: linear-gradient(135deg, #1e1b4b 0%, #4f46e5 50%, #06b6d4 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        letter-spacing: -1px !important;
    }

    .main-caption {
        font-size: 15px !important;
        color: #64748b !important;
        margin-bottom: 25px !important;
        font-weight: 400 !important;
    }

    /* Metrics card system */
    .metric-container {
        display: flex;
        gap: 16px;
        margin: 24px 0;
        flex-wrap: wrap;
        width: 100%;
    }
    .custom-card {
        flex: 1;
        min-width: 200px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
        transition: all 0.25s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }
    .card-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .card-value {
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Key Insights & Alerts */
    .insight-item {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        background: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.02);
        transition: border-color 0.2s;
    }
    .insight-item:hover {
        border-color: #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .insight-icon {
        font-size: 20px;
        line-height: 1;
    }
    .insight-text {
        font-size: 14px;
        font-weight: 400;
        color: #334155;
        line-height: 1.6;
    }

    .custom-alert-success {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 12px;
        padding: 16px;
        color: #166534;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }

    /* Sidebar Badge Component */
    .sidebar-badge {
        background: rgba(79, 70, 229, 0.08);
        color: #4f46e5;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(79, 70, 229, 0.15);
        display: inline-block;
        margin-top: 10px;
    }

    /* Sidebar file info card */
    .sidebar-file-info {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
    }
    .sidebar-file-info .file-name {
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        word-break: break-all;
    }
    .sidebar-file-info .file-meta {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }

    /* Suggested question chips */
    .question-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }

    /* Statistics section divider */
    .stat-section-title {
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Radio-as-Tab-Bar Navigation */
    div[data-testid="stRadio"] > div {
        display: flex;
        gap: 0;
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stRadio"] > div > label {
        flex: 1;
        text-align: center;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 15px;
        color: #64748b;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
        background: transparent;
        margin: 0 !important;
    }
    div[data-testid="stRadio"] > div > label:hover {
        color: #4f46e5;
        background: rgba(79, 70, 229, 0.05);
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {
        background: #ffffff;
        color: #4f46e5;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.15);
        font-weight: 700;
    }
    /* Hide the radio circle indicator */
    div[data-testid="stRadio"] > div > label > div:first-child {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 25px; padding-top: 10px;">
            <div style="font-size: 48px; margin-bottom: 8px;">🤖</div>
            <div style="font-size: 20px; font-weight: 700; color: #0f172a; letter-spacing: -0.5px;">Data Agent Hub</div>
            <div class="sidebar-badge">Production Hardened</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    upload_clicked = st.button(
        "🚀 Analyze",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

    # Show file info if dataset is loaded
    if "profile" in st.session_state:
        profile = st.session_state.profile
        filename = st.session_state.get("uploaded_filename", "dataset")
        st.markdown(
            f"""
            <div class="sidebar-file-info">
                <div class="file-name">📄 {filename}</div>
                <div class="file-meta">{profile.get('rows', 0):,} rows × {profile.get('columns', 0)} columns</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Upload New Dataset", use_container_width=True):
            clear_dataset_state()
            st.rerun()

# ─── HANDLE UPLOAD ────────────────────────────────────────────────────────────
if upload_clicked and uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            uploaded_file.type or "application/octet-stream",
        )
    }

    with st.status("Agent is analyzing the dataset...", expanded=True) as status_box:
        try:
            st.write("Uploading file and creating an analysis session...")
            response = httpx.post(
                f"{BACKEND_URL}/datasets/upload", files=files, timeout=30.0
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.json().get("detail", "Upload failed.")
            status_box.update(
                label="Dataset analysis failed.", state="error", expanded=True
            )
            st.error(detail)
        except httpx.RequestError:
            status_box.update(
                label="Dataset analysis failed.", state="error", expanded=True
            )
            st.error(
                "Could not reach the backend. Start FastAPI first, then try again."
            )
        else:
            payload = response.json()
            clear_dataset_state()
            st.session_state.session_id = payload["session_id"]
            st.session_state.session_token = payload.get("session_token")
            st.session_state.upload_result = payload
            st.session_state.uploaded_filename = uploaded_file.name
            st.write("Parsing a local preview for charts...")
            try:
                st.session_state.dataset_frame = parse_uploaded_dataframe(
                    uploaded_file.name, file_bytes
                )
            except Exception:
                st.session_state.dataset_frame = pd.DataFrame()
            st.write(
                "Profiling schema, missing values, numeric summaries, and categories..."
            )
            try:
                st.session_state.profile = fetch_profile(payload["session_id"])
                rows = st.session_state.profile.get("rows", 0)
                columns = st.session_state.profile.get("columns", 0)
                st.write(f"Loaded {rows} rows and {columns} columns.")
                st.write("Generating suggested questions and grounded insights...")
                st.session_state.suggestions = fetch_suggestions(payload["session_id"])
                st.write(
                    "Building the automated insight dashboard and chart recommendations..."
                )
                st.session_state.auto_analysis = fetch_auto_analysis(
                    payload["session_id"]
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.json().get(
                    "detail", "Could not load dataset profile or suggestions."
                )
                clear_dataset_state()
                status_box.update(
                    label="Dataset analysis failed.",
                    state="error",
                    expanded=True,
                )
                st.error(
                    f"{detail} Upload request succeeded, but the analysis session could not be loaded. "
                    "If the backend just restarted, please upload once more."
                )
            except httpx.RequestError:
                clear_dataset_state()
                status_box.update(
                    label="Dataset analysis failed.",
                    state="error",
                    expanded=True,
                )
                st.error(
                    "Dataset upload reached the backend, but profile/suggestions could not be loaded."
                )
            else:
                status_box.update(
                    label="Dataset analysis completed.",
                    state="complete",
                    expanded=False,
                )
                st.success("Dataset uploaded successfully.")

# ─── MAIN CONTENT AREA ────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">AI Data Analyst Agent</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="main-caption">Safe tabular data analysis with FastAPI, Streamlit, pandas tools, Plotly, and optional Gemini.</div>',
    unsafe_allow_html=True,
)

if "profile" not in st.session_state:
    # ─── LANDING STATE: No dataset loaded ──────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 80px 20px;">
            <div style="font-size: 72px; margin-bottom: 20px;">📊</div>
            <div style="font-size: 24px; font-weight: 700; color: #1e293b; margin-bottom: 10px;">
                Upload a dataset to get started
            </div>
            <div style="font-size: 15px; color: #64748b; max-width: 500px; margin: 0 auto;">
                Use the sidebar to upload a CSV or XLSX file. The AI Agent will automatically analyze your data, 
                generate visual insights, and be ready to answer your questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ─── 3-TAB LAYOUT (session-state-backed to persist across reruns) ──────
    profile = st.session_state.profile
    auto_analysis = st.session_state.get("auto_analysis", {})
    suggestions = st.session_state.get("suggestions", {})

    TAB_OPTIONS = ["📊 Dashboard", "🔍 Data Explorer", "💬 Chat"]

    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_OPTIONS[0]

    active_tab = st.radio(
        "Navigation",
        TAB_OPTIONS,
        index=TAB_OPTIONS.index(st.session_state.active_tab),
        horizontal=True,
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state.active_tab = active_tab

    st.markdown("---")

    if active_tab == TAB_OPTIONS[0]:
        render_dashboard_tab(profile, auto_analysis, suggestions)
    elif active_tab == TAB_OPTIONS[1]:
        render_data_explorer_tab(profile)
    elif active_tab == TAB_OPTIONS[2]:
        render_chat_tab(auto_analysis, suggestions)
