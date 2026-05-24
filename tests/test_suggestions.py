import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.agent.suggestions import (
    _build_profiling_signals,
    _build_suggestions_prompt,
    _validate_insights,
    generate_suggested_content,
)
from backend.main import app
from backend.services.profiling import profile_dataset
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


def test_fallback_insights_are_grounded_with_numbers() -> None:
    suggested = generate_suggested_content(_sample_dataframe(), provider=None)

    assert 3 <= len(suggested.insights) <= 5
    assert all(re.search(r"\d", insight) for insight in suggested.insights)
    assert any(
        "department là cột có tỷ lệ missing cao nhất" in insight and "25%" in insight
        for insight in suggested.insights
    )
    assert any("salary" in insight and "900" in insight and "1.500" in insight for insight in suggested.insights)
    assert any(
        'giá trị phổ biến nhất là "Engineering"' in insight and "2 dòng" in insight and "50%" in insight
        for insight in suggested.insights
    )


def test_generate_suggested_content_uses_gemini_and_filters_unknown_structured_columns() -> None:
    provider = FakeProvider(
        '{"questions":["Tính trung bình salary theo department.","Mô tả Unknown_Column."],'
        '"insights":["Cột salary có trung bình 1200, dao động từ 900 đến 1500."]}'
    )

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "gemini"
    assert suggested.questions == ["Tính trung bình salary theo department."]
    assert suggested.insights == ["Cột salary có trung bình 1200, dao động từ 900 đến 1500."]
    assert "department" in provider.prompts[0]
    assert "PROFILING_SIGNALS" in provider.prompts[0]


def test_gemini_generic_insights_fall_back_to_deterministic_templates() -> None:
    provider = FakeProvider(
        '{"questions":["Tính trung bình salary theo department."],'
        '"insights":["Phần lớn nhân sự thuộc Engineering."]}'
    )

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "gemini"
    assert suggested.questions == ["Tính trung bình salary theo department."]
    assert all(re.search(r"\d", insight) for insight in suggested.insights)
    assert any("Dataset có 4 dòng và 3 cột." == insight for insight in suggested.insights)


def test_generate_suggested_content_falls_back_on_invalid_gemini_response() -> None:
    provider = FakeProvider("not json")

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "fallback"
    assert suggested.questions
    assert suggested.insights


def test_validate_insights_rejects_generic_and_speculative_wording() -> None:
    profile = profile_dataset(_sample_dataframe())

    validated = _validate_insights(
        [
            "Phần lớn nhân sự thuộc Engineering.",
            "Salary cao có thể do tenure_years cao hơn.",
            "Có 78 bản ghi (1.18%) bị thiếu giá trị cho cột Teacher_Quality.",
            "Số buổi Tutoring_Sessions có một giá trị ngoại lệ là 8.",
            "50.91% phụ huynh có mức độ tham gia Medium vào việc học.",
            "Cột salary có trung bình 1200 trên 3 giá trị hợp lệ.",
        ],
        profile,
    )

    assert validated == ["Cột salary có trung bình 1200 trên 3 giá trị hợp lệ."]


def test_outlier_insight_uses_analyst_wording() -> None:
    dataframe = pd.DataFrame(
        {
            "Tutoring_Sessions": [1, 1, 1, 2, 2, 8],
            "Sleep_Hours": [7, 6, 7, 8, 7, 6],
        }
    )

    suggested = generate_suggested_content(dataframe, provider=None)

    assert any(
        "Tutoring_Sessions xuất hiện giá trị cao bất thường" in insight and "max = 8" in insight
        for insight in suggested.insights
    )
    assert all("giá trị ngoại lệ là" not in insight for insight in suggested.insights)


def test_suggested_questions_are_schema_grounded_and_diverse() -> None:
    suggested = generate_suggested_content(_sample_dataframe(), provider=None)

    assert any("department" in question for question in suggested.questions)
    assert any("salary" in question for question in suggested.questions)
    assert any("tương quan" in question for question in suggested.questions)
    assert all("Unknown" not in question for question in suggested.questions)


def test_correlation_signal_and_insight_when_two_numeric_columns_exist() -> None:
    suggested = generate_suggested_content(_sample_dataframe(), provider=None)
    signals = _build_profiling_signals(profile_dataset(_sample_dataframe()), _sample_dataframe())

    assert signals["correlation_candidates"]
    assert any("tương quan" in insight and "r=" in insight for insight in suggested.insights)


def test_gemini_prompt_hardens_insight_contract() -> None:
    dataframe = _sample_dataframe()
    profile = profile_dataset(dataframe)
    signals = _build_profiling_signals(profile, dataframe)

    prompt = _build_suggestions_prompt(profile, signals)

    assert "Mỗi insight bắt buộc có ít nhất một số liệu cụ thể" in prompt
    assert "Không suy diễn nguyên nhân" in prompt
    assert "Top category insight nên theo style" in prompt
    assert "Outlier insight không được viết" in prompt
    assert "PROFILING_SIGNALS" in prompt
    assert "correlation_candidates" in prompt


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
    assert all(re.search(r"\d", insight) for insight in payload["insights"])


def test_suggestions_endpoint_uses_mock_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"questions":["Giá trị nào xuất hiện nhiều nhất trong cột department?"],'
        '"insights":["Trong cột department, Engineering đứng hạng 1 với 2 dòng, chiếm khoảng 50% dữ liệu."]}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.get(f"/datasets/{session_id}/suggestions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "gemini"
    assert payload["questions"] == ["Giá trị nào xuất hiện nhiều nhất trong cột department?"]
    assert payload["insights"] == [
        "Trong cột department, Engineering đứng hạng 1 với 2 dòng, chiếm khoảng 50% dữ liệu."
    ]


def test_suggestions_endpoint_returns_404_for_unknown_session() -> None:
    client = TestClient(app)

    response = client.get("/datasets/missing/suggestions")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset session not found."
