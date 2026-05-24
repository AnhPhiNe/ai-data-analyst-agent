import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.agent.suggestions import generate_suggested_content
from backend.main import app
from backend.services.session_store import session_store


class FakeProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.fixture(autouse=True)
def clear_session_store() -> None:
    session_store.clear()


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", None],
            "salary": [1200.0, 900.0, 1500.0, None],
            "tenure_years": [2, 1, 5, 3],
        }
    )


def _upload_dataset(client: TestClient) -> str:
    csv_content = (
        "department,salary,tenure_years\n"
        "Engineering,1200,2\n"
        "Sales,900,1\n"
        "Engineering,1500,5\n"
        ",,3\n"
    ).encode("utf-8")
    response = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_generate_suggested_content_uses_fallback_without_provider() -> None:
    suggested = generate_suggested_content(_sample_dataframe(), provider=None)

    assert suggested.source == "fallback"
    assert suggested.questions
    assert suggested.insights
    assert any("salary" in question for question in suggested.questions)


def test_generate_suggested_content_uses_gemini_and_filters_unknown_structured_columns() -> None:
    provider = FakeProvider(
        '{"questions":["Tính trung bình salary theo department.","Mô tả Unknown_Column."],'
        '"insights":["Dữ liệu cho thấy salary có thể phân tích theo department."]}'
    )

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "gemini"
    assert suggested.questions == ["Tính trung bình salary theo department."]
    assert suggested.insights == ["Dữ liệu cho thấy salary có thể phân tích theo department."]
    assert "department" in provider.prompts[0]


def test_generate_suggested_content_falls_back_on_invalid_gemini_response() -> None:
    provider = FakeProvider("not json")

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "fallback"
    assert suggested.questions


def test_suggestions_endpoint_returns_fallback_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: None)

    response = client.get(f"/datasets/{session_id}/suggestions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["source"] == "fallback"
    assert payload["questions"]
    assert payload["insights"]


def test_suggestions_endpoint_uses_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"questions":["Top giá trị phổ biến nhất của department là gì?"],'
        '"insights":["Dữ liệu cho thấy department có nhiều nhóm khác nhau."]}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.get(f"/datasets/{session_id}/suggestions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "gemini"
    assert payload["questions"] == ["Top giá trị phổ biến nhất của department là gì?"]


def test_suggestions_endpoint_returns_404_for_unknown_session() -> None:
    client = TestClient(app)

    response = client.get("/datasets/missing/suggestions")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset session not found."
