import pandas as pd

from backend.agent.tool_validation import validate_tool_call


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "HR"],
            "salary": [1200.0, 900.0, 1000.0],
            "tenure_years": [2, 1, 3],
        }
    )


def test_validate_tool_call_accepts_valid_aggregate() -> None:
    result = validate_tool_call(
        _sample_dataframe(),
        "aggregate_metric",
        {"metric_column": "salary", "group_by": "department", "operation": "mean"},
    )

    assert result.is_valid is True
    assert result.normalized_arguments == {
        "metric_column": "salary",
        "group_by": "department",
        "operation": "mean",
        "limit": 20,
    }


def test_validate_tool_call_rejects_unknown_tool() -> None:
    result = validate_tool_call(
        _sample_dataframe(), "run_python", {"code": "print('no')"}
    )

    assert result.is_valid is False
    assert "not allowed" in result.message


def test_validate_tool_call_rejects_dangerous_nested_key() -> None:
    result = validate_tool_call(
        _sample_dataframe(),
        "value_counts",
        {"column": "department", "nested": {"exec": "x"}},
    )

    assert result.is_valid is False
    assert "not allowed" in result.message


def test_validate_tool_call_rejects_missing_column() -> None:
    result = validate_tool_call(
        _sample_dataframe(), "value_counts", {"column": "unknown"}
    )

    assert result.is_valid is False
    assert "does not exist" in result.message


def test_validate_tool_call_rejects_non_numeric_metric() -> None:
    result = validate_tool_call(
        _sample_dataframe(),
        "aggregate_metric",
        {"metric_column": "department", "group_by": "salary"},
    )

    assert result.is_valid is False
    assert "must be numeric" in result.message


def test_validate_tool_call_accepts_chart_spec() -> None:
    result = validate_tool_call(
        _sample_dataframe(),
        "generate_chart_spec",
        {"chart_type": "bar", "x": "department", "y": "salary"},
    )

    assert result.is_valid is True
    assert result.normalized_arguments == {
        "chart_type": "bar",
        "x": "department",
        "y": "salary",
    }
