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


from tests.conftest import FakeProvider


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", None],
            "salary": [1200.0, 900.0, 1500.0, None],
            "tenure_years": [2, 1, 5, 3],
        }
    )


def _student_like_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Parental_Education_Level": [
                None,
                "High School",
                "College",
                "College",
                "High School",
                "College",
                "College",
                "High School",
            ],
            "Tutoring_Sessions": [0, 1, 1, 1, 1, 2, 2, 8],
            "Hours_Studied": [1, 10, 12, 15, 18, 21, 24, 28],
            "Internet_Access": ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No"],
            "Attendance": [60, 65, 70, 75, 80, 85, 90, 95],
            "Exam_Score": [55, 58, 61, 65, 68, 72, 76, 80],
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
        "department: missing 1" in insight and "25%" in insight
        for insight in suggested.insights
    )
    assert any(
        "mean=" in insight and "median=" in insight and "min-max=" in insight
        for insight in suggested.insights
    )
    assert any(
        'department="Engineering"' in insight
        and "2 dòng" in insight
        and "50%" in insight
        for insight in suggested.insights
    )


def test_generate_suggested_content_uses_gemini_and_filters_unknown_structured_columns() -> (
    None
):
    provider = FakeProvider(
        '{"questions":["Tính trung bình salary theo department.","Mô tả Unknown_Column."],'
        '"insights":["Cột salary có trung bình 1200, dao động từ 900 đến 1500."]}'
    )
    fallback = generate_suggested_content(_sample_dataframe(), provider=None)

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "gemini"
    assert suggested.questions[0] == "Tính trung bình salary theo department."
    assert len(suggested.questions) > 1
    assert suggested.insights == fallback.insights
    assert "department" in provider.prompts[0]
    assert "PROFILING_SIGNALS" in provider.prompts[0]


def test_gemini_insights_are_ignored_for_deterministic_templates() -> None:
    provider = FakeProvider(
        '{"questions":["Tính trung bình salary theo department."],'
        '"insights":["Phần lớn nhân sự thuộc Engineering.","Cột salary có trung bình 9999."]}'
    )
    fallback = generate_suggested_content(_sample_dataframe(), provider=None)

    suggested = generate_suggested_content(_sample_dataframe(), provider=provider)

    assert suggested.source == "gemini"
    assert suggested.questions[0] == "Tính trung bình salary theo department."
    assert len(suggested.questions) > 1
    assert suggested.insights == fallback.insights
    assert not any("9999" in insight for insight in suggested.insights)


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
        "Tutoring_Sessions: max=8" in insight and "cao bất thường" in insight
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
    signals = _build_profiling_signals(
        profile_dataset(_sample_dataframe()), _sample_dataframe()
    )

    assert signals["correlation_candidates"]
    assert any(" vs " in insight and "r=" in insight for insight in suggested.insights)


def test_student_like_dataset_returns_five_meaningful_deterministic_insights() -> None:
    suggested = generate_suggested_content(_student_like_dataframe(), provider=None)

    assert len(suggested.insights) == 5
    assert any(
        "Parental_Education_Level: missing" in insight for insight in suggested.insights
    )
    assert any(
        "mean=" in insight and "min-max=" in insight for insight in suggested.insights
    )
    assert any('Internet_Access="Yes"' in insight for insight in suggested.insights)
    assert any("bất thường" in insight for insight in suggested.insights)
    assert any(
        "Attendance vs Exam_Score" in insight and "r=" in insight
        for insight in suggested.insights
    )


def test_weak_correlation_is_not_highlighted() -> None:
    dataframe = pd.DataFrame(
        {
            "Sleep_Hours": [6, 6, 8, 8],
            "Previous_Scores": [70, 80, 70, 80],
            "Attendance": [90, 80, 80, 90],
        }
    )

    suggested = generate_suggested_content(dataframe, provider=None)
    signals = _build_profiling_signals(profile_dataset(dataframe), dataframe)

    assert signals["correlation_candidates"] == []
    assert not any("tương quan" in insight for insight in suggested.insights)
    assert not any("tương quan" in question for question in suggested.questions)


def test_moderate_correlation_below_threshold_is_not_highlighted() -> None:
    dataframe = pd.DataFrame(
        {
            "Study_Hours": [1, 2, 3, 4, 5, 6],
            "Sleep_Hours": [1, 6, 2, 5, 3, 4],
        }
    )

    suggested = generate_suggested_content(dataframe, provider=None)
    signals = _build_profiling_signals(profile_dataset(dataframe), dataframe)

    assert signals["correlation_candidates"] == []
    assert not any("tương quan" in insight for insight in suggested.insights)
    assert not any("tương quan" in question for question in suggested.questions)


def test_gemini_prompt_hardens_insight_contract() -> None:
    dataframe = _sample_dataframe()
    profile = profile_dataset(dataframe)
    signals = _build_profiling_signals(profile, dataframe)

    prompt = _build_suggestions_prompt(profile, signals)

    assert "Không tự tạo insights" in prompt
    assert "|r| >= 0.50" in prompt
    assert "PROFILING_SIGNALS" in prompt
    assert "correlation_candidates" in prompt


def test_suggestions_endpoint_returns_fallback_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_suggestions_endpoint_uses_mock_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert (
        payload["questions"][0]
        == "Giá trị nào xuất hiện nhiều nhất trong cột department?"
    )
    assert len(payload["questions"]) > 1
    assert all(re.search(r"\d", insight) for insight in payload["insights"])
    assert not any("đứng hạng 1" in insight for insight in payload["insights"])


def test_suggestions_endpoint_returns_404_for_unknown_session() -> None:
    client = TestClient(app)

    response = client.get("/datasets/missing/suggestions")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset session not found."
