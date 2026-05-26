import json
import os
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get(
    "BACKEND_URL",
    "http://localhost:8000",
)


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("session_token")
    return {"X-Session-Token": str(token)} if token else {}


def fetch_profile(session_id: str) -> dict[str, object]:
    response = httpx.get(
        f"{BACKEND_URL}/datasets/{session_id}/profile",
        headers=auth_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_suggestions(session_id: str) -> dict[str, object]:
    response = httpx.get(
        f"{BACKEND_URL}/datasets/{session_id}/suggestions",
        headers=auth_headers(),
        timeout=45.0,
    )
    response.raise_for_status()
    return response.json()


def fetch_auto_analysis(session_id: str) -> dict[str, object]:
    response = httpx.get(
        f"{BACKEND_URL}/datasets/{session_id}/auto-analysis",
        headers=auth_headers(),
        timeout=45.0,
    )
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


def stream_chat_question(session_id: str, question: str):
    with httpx.stream(
        "POST",
        f"{BACKEND_URL}/chat/query/stream",
        json={"session_id": session_id, "question": question},
        headers=auth_headers(),
        timeout=60.0,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            yield json.loads(line)
