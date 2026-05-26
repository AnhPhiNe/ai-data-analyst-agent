from fastapi.testclient import TestClient

from backend.main import app
from backend.services.auto_analysis import generate_auto_analysis


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
    assert analysis["recommended_charts"][0]["chart_spec"] == {
        "chart_type": "pie",
        "names": "region",
    }


def test_generate_auto_analysis_limits_charts_and_deduplicates_types() -> None:
    import pandas as pd

    # Create a dataframe with 3 numeric columns and 2 categorical columns
    dataframe = pd.DataFrame(
        {
            "salary": [1000, 2000, 3000],
            "age": [25, 30, 35],
            "tenure": [1, 3, 5],
            "department": ["HR", "Engineering", "Sales"],
            "city": ["Hanoi", "HCM", "DaNang"],
        }
    )

    analysis = generate_auto_analysis(dataframe)
    recommended = analysis["recommended_charts"]

    # 1. Total charts must be at most 4
    assert len(recommended) <= 4

    # 2. Each chart must have a unique chart_type
    seen_types = set()
    for item in recommended:
        chart_type = item["chart_spec"]["chart_type"]
        assert chart_type not in seen_types
        seen_types.add(chart_type)

    # 3. Since there are multiple numeric columns, correlation_heatmap MUST be present
    assert "correlation_heatmap" in seen_types


def test_generate_auto_analysis_with_gemini_duplicate_fallback() -> None:
    import pandas as pd
    from unittest.mock import MagicMock

    dataframe = pd.DataFrame(
        {
            "salary": [1000, 2000, 3000],
            "age": [25, 30, 35],
            "department": ["HR", "Engineering", "Sales"],
        }
    )

    # Mock provider to return duplicate bar chart specs
    mock_provider = MagicMock()
    mock_provider.generate.return_value = """
    [
      {
        "title": "Salary by Dept",
        "chart_spec": {
          "chart_type": "bar",
          "x": "department",
          "y": "salary"
        },
        "reason": "Compare salary across departments."
      },
      {
        "title": "Age by Dept",
        "chart_spec": {
          "chart_type": "bar",
          "x": "department",
          "y": "age"
        },
        "reason": "Compare age across departments."
      }
    ]
    """

    analysis = generate_auto_analysis(dataframe, provider=mock_provider)
    recommended = analysis["recommended_charts"]

    # The second duplicate bar chart must be filtered out, and a correlation heatmap must be automatically injected
    seen_types = [item["chart_spec"]["chart_type"] for item in recommended]
    assert seen_types.count("bar") <= 1
    assert "correlation_heatmap" in seen_types
    assert len(recommended) <= 4


def test_eta_squared_filters_flat_bar_charts() -> None:
    """Bar charts where group averages are identical must be discarded."""
    import pandas as pd

    # Intentionally create a dataset where Hours_Studied has identical mean across Parental groups
    dataframe = pd.DataFrame(
        {
            "hours": [20, 20, 20, 20, 20, 20],
            "score": [60, 70, 80, 65, 75, 85],
            "group": ["Low", "Low", "Medium", "Medium", "High", "High"],
        }
    )

    analysis = generate_auto_analysis(dataframe)
    recommended = analysis["recommended_charts"]

    # No bar chart should appear for hours × group because eta² ≈ 0
    for item in recommended:
        spec = item["chart_spec"]
        if spec.get("chart_type") == "bar":
            assert not (
                spec.get("x") == "group" and spec.get("y") == "hours"
            ), "Flat bar chart (hours by group) should have been filtered out"


def test_scatter_always_uses_top_correlation_pair() -> None:
    """Scatter plot must always use the pair with strongest |r|."""
    import pandas as pd

    # Create dataset where score-attendance has r≈1.0, but score-hours has r≈0.5
    dataframe = pd.DataFrame(
        {
            "attendance": [60, 70, 80, 90, 100],
            "hours": [10, 20, 15, 25, 12],
            "score": [60, 70, 80, 90, 100],  # perfectly correlated with attendance
            "group": ["A", "B", "C", "A", "B"],
        }
    )

    analysis = generate_auto_analysis(dataframe)
    recommended = analysis["recommended_charts"]

    scatter_charts = [
        item
        for item in recommended
        if item["chart_spec"].get("chart_type") == "scatter"
    ]
    if scatter_charts:
        spec = scatter_charts[0]["chart_spec"]
        pair = {spec.get("x"), spec.get("y")}
        assert (
            "attendance" in pair and "score" in pair
        ), f"Scatter should use attendance vs score (strongest r), got {pair}"


def test_histogram_selects_most_variable_column() -> None:
    """Histogram should prefer the column with highest coefficient of variation."""
    import pandas as pd

    dataframe = pd.DataFrame(
        {
            "stable": [100, 100, 100, 100, 100],  # CV = 0
            "variable": [10, 50, 200, 500, 1000],  # CV >> 0
        }
    )

    analysis = generate_auto_analysis(dataframe)
    recommended = analysis["recommended_charts"]

    hist_charts = [
        item
        for item in recommended
        if item["chart_spec"].get("chart_type") == "histogram"
    ]
    assert len(hist_charts) >= 1
    assert (
        hist_charts[0]["chart_spec"]["x"] == "variable"
    ), "Histogram should prefer the column with highest CV"
