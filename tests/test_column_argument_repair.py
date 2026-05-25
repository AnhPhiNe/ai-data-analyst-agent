import pandas as pd

from backend.agent.column_argument_repair import repair_tool_column_arguments


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "HR"],
            "performance_score": [4.5, 3.8, 4.1],
            "Monthly_Revenue": [1200, 900, 1000],
            "Customer_Age": [30, 24, 41],
        }
    )


def test_repairs_single_column_arguments() -> None:
    dataframe = _sample_dataframe()

    assert repair_tool_column_arguments(dataframe, "describe_numeric", {"column": "diem"}) == {
        "column": "performance_score"
    }
    assert repair_tool_column_arguments(dataframe, "value_counts", {"column": "phong ban"}) == {
        "column": "department"
    }
    assert repair_tool_column_arguments(dataframe, "sort_values", {"column": "doanh thu"}) == {
        "column": "Monthly_Revenue"
    }


def test_repairs_aggregate_metric_arguments() -> None:
    repaired = repair_tool_column_arguments(
        _sample_dataframe(),
        "aggregate_metric",
        {"metric_column": "diem", "group_by": "phong ban", "operation": "mean"},
    )

    assert repaired == {
        "metric_column": "performance_score",
        "group_by": "department",
        "operation": "mean",
    }


def test_repairs_filter_and_percentage_arguments() -> None:
    dataframe = _sample_dataframe()

    assert repair_tool_column_arguments(
        dataframe,
        "filter_rows",
        {"column": "doanh thu", "operator": "gt", "value": 1000},
    ) == {"column": "Monthly_Revenue", "operator": "gt", "value": 1000}
    assert repair_tool_column_arguments(
        dataframe,
        "conditional_percentage",
        {"column": "tuoi khach hang", "operator": "lt", "value": 30},
    ) == {"column": "Customer_Age", "operator": "lt", "value": 30}


def test_repairs_chart_and_correlation_arguments() -> None:
    dataframe = _sample_dataframe()

    assert repair_tool_column_arguments(
        dataframe,
        "generate_chart_spec",
        {"chart_type": "bar", "x": "phong ban", "y": "diem"},
    ) == {"chart_type": "bar", "x": "department", "y": "performance_score"}
    assert repair_tool_column_arguments(
        dataframe,
        "correlation_analysis",
        {"columns": ["diem", "doanh thu", "tuoi khach hang"]},
    ) == {"columns": ["performance_score", "Monthly_Revenue", "Customer_Age"]}


def test_repairs_chart_axis_aliases_from_llm() -> None:
    assert repair_tool_column_arguments(
        _sample_dataframe(),
        "generate_chart_spec",
        {"chart_type": "bar", "x_axis": "phong ban", "y_axis": "diem"},
    ) == {"chart_type": "bar", "x": "department", "y": "performance_score"}


def test_leaves_unresolved_columns_for_validation_to_reject() -> None:
    assert repair_tool_column_arguments(_sample_dataframe(), "describe_numeric", {"column": "unknown"}) == {
        "column": "unknown"
    }
