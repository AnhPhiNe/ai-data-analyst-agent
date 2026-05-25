import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.auto_analysis import generate_auto_analysis
from backend.services.session_store import session_store


@pytest.fixture(autouse=True)
def clear_session_store() -> None:
    session_store.clear()


def _upload_dataset(client: TestClient) -> str:
    csv_content = (
        "department,salary,tenure_years,performance_score\n"
        "Engineering,1200,2,4.5\n"
        "Sales,900,1,3.8\n"
        "Engineering,1500,5,4.9\n"
        "HR,,3,4.1\n"
    ).encode("utf-8")
    response = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_auto_analysis_endpoint_returns_workflow_report() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.get(f"/datasets/{session_id}/auto-analysis")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert "profile_dataset" in payload["workflow_steps"]
    assert payload["overview"] == {
        "rows": 4,
        "columns": 4,
        "column_names": ["department", "salary", "tenure_years", "performance_score"],
    }
    assert payload["data_quality"]["total_missing_cells"] == 1
    assert payload["numeric_highlights"]
    assert payload["categorical_highlights"][0]["column"] == "department"
    assert payload["recommended_charts"]
    assert payload["next_questions"]


def test_generate_auto_analysis_handles_categorical_only_dataset() -> None:
    import pandas as pd

    dataframe = pd.DataFrame(
        {
            "region": ["North", "South", "North"],
            "segment": ["SMB", "Enterprise", "SMB"],
        }
    )

    analysis = generate_auto_analysis(dataframe)

    assert analysis["numeric_highlights"] == []
    assert analysis["correlation_highlights"] == []
    assert analysis["categorical_highlights"][0]["column"] == "region"
    assert analysis["recommended_charts"][0]["chart_spec"] == {"chart_type": "pie", "names": "region"}
