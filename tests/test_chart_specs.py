import pandas as pd
import pytest

from backend.visualization.chart_specs import ChartSpecValidationError, validate_chart_spec


def _chart_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", "HR"],
            "salary": [1200.0, 900.0, 1500.0, 1000.0],
            "tenure_years": [2, 1, 5, 3],
            "performance_score": [4.5, 3.8, 4.9, 4.1],
            "month": ["Jan", "Feb", "Mar", "Apr"],
        }
    )


def test_validate_bar_chart_spec() -> None:
    spec = {"chart_type": "bar", "x": "department", "y": "salary", "title": "Salary by department"}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == spec


def test_validate_line_chart_spec() -> None:
    spec = {"chart_type": "line", "x": "month", "y": "salary"}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == spec


def test_validate_histogram_chart_spec_with_bins() -> None:
    spec = {"chart_type": "histogram", "x": "salary", "bins": 5}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == spec


def test_validate_scatter_requires_numeric_x() -> None:
    spec = {"chart_type": "scatter", "x": "department", "y": "salary"}

    with pytest.raises(ChartSpecValidationError, match="must be numeric"):
        validate_chart_spec(spec, _chart_dataframe())


def test_validate_correlation_heatmap_defaults_to_numeric_columns() -> None:
    spec = {"chart_type": "correlation_heatmap"}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == {
        "chart_type": "correlation_heatmap",
        "columns": ["salary", "tenure_years", "performance_score"],
    }


def test_validate_box_chart_allows_optional_group_column() -> None:
    spec = {"chart_type": "box", "x": "department", "y": "salary"}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == spec


def test_validate_pie_chart_normalizes_x_y_aliases() -> None:
    spec = {"chart_type": "pie", "x": "department", "y": "salary"}

    validated = validate_chart_spec(spec, _chart_dataframe())

    assert validated == {"chart_type": "pie", "names": "department", "values": "salary"}


def test_validate_pie_chart_rejects_high_cardinality_categories() -> None:
    dataframe = pd.DataFrame({"category": [f"item_{index}" for index in range(11)], "amount": range(11)})

    with pytest.raises(ChartSpecValidationError, match="10 or fewer"):
        validate_chart_spec({"chart_type": "pie", "names": "category", "values": "amount"}, dataframe)


def test_validate_chart_spec_rejects_unknown_fields() -> None:
    spec = {"chart_type": "bar", "x": "department", "y": "salary", "code": "print('no')"}

    with pytest.raises(ChartSpecValidationError, match="Invalid chart spec field 'code'"):
        validate_chart_spec(spec, _chart_dataframe())


def test_validate_chart_spec_rejects_missing_required_axis() -> None:
    spec = {"chart_type": "bar", "x": "department"}

    with pytest.raises(ChartSpecValidationError, match="'y' is required"):
        validate_chart_spec(spec, _chart_dataframe())
