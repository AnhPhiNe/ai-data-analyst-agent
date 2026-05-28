import pandas as pd

from backend.tools.safe_pandas import TOOL_REGISTRY, execute_tool


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
            "salary": [1200.0, 900.0, 1500.0, None, 950.0],
            "tenure_years": [2, 1, 5, 3, 2],
            "performance_score": [4.5, 3.8, 4.9, 4.1, None],
            "is_manager": [True, False, True, False, False],
        }
    )


def test_registry_contains_only_expected_mvp_tools() -> None:
    assert set(TOOL_REGISTRY) == {
        "list_columns",
        "profile_dataset",
        "describe_numeric",
        "detect_missing_values",
        "data_quality_report",
        "value_counts",
        "aggregate_metric",
        "compare_groups",
        "sort_values",
        "outlier_detection",
        "filter_rows",
        "conditional_percentage",
        "correlation_analysis",
        "generate_chart_spec",
    }


def test_execute_tool_rejects_unknown_tool() -> None:
    result = execute_tool(_sample_dataframe(), "run_python", {"code": "print('nope')"})

    assert result.status == "error"
    assert "not allowed" in result.message


def test_execute_tool_rejects_dangerous_argument_keys() -> None:
    result = execute_tool(_sample_dataframe(), "list_columns", {"__class__": "bad"})

    assert result.status == "error"
    assert "not allowed" in result.message


def test_list_columns_returns_columns_and_dtypes() -> None:
    result = execute_tool(_sample_dataframe(), "list_columns")

    assert result.status == "success"
    assert result.data == {
        "columns": [
            "department",
            "salary",
            "tenure_years",
            "performance_score",
            "is_manager",
        ]
    }
    assert result.table[0] == {"column": "department", "dtype": "object"}


def test_profile_dataset_returns_profile_summary() -> None:
    result = execute_tool(_sample_dataframe(), "profile_dataset")

    assert result.status == "success"
    assert result.data["rows"] == 5
    assert result.data["columns"] == 5
    assert "numeric_summary" in result.data


def test_describe_numeric_for_one_column() -> None:
    result = execute_tool(_sample_dataframe(), "describe_numeric", {"column": "salary"})

    assert result.status == "success"
    assert result.table == [
        {
            "column": "salary",
            "count": 4,
            "mean": 1137.5,
            "std": 275.0,
            "min": 900.0,
            "median": 1075.0,
            "max": 1500.0,
        }
    ]


def test_describe_numeric_rejects_non_numeric_column() -> None:
    result = execute_tool(
        _sample_dataframe(), "describe_numeric", {"column": "department"}
    )

    assert result.status == "error"
    assert "must be numeric" in result.message


def test_detect_missing_values_returns_all_columns() -> None:
    result = execute_tool(_sample_dataframe(), "detect_missing_values")

    assert result.status == "success"
    salary_row = next(row for row in result.table if row["column"] == "salary")
    assert salary_row == {
        "column": "salary",
        "missing_count": 1,
        "missing_percent": 20.0,
    }


def test_data_quality_report_returns_quality_signals() -> None:
    dataframe = pd.DataFrame(
        {
            "user_id": [f"user_{index}" for index in range(25)] + ["user_24"],
            "department": ["Engineering"] * 26,
            "salary": [1000.0, None, *([1200.0] * 24)],
            "free_text": [f"note_{index}" for index in range(26)],
        }
    )

    result = execute_tool(dataframe, "data_quality_report")

    assert result.status == "success"
    assert result.data["duplicate_rows"] == 0
    assert "salary" in result.data["missing_columns"]
    assert "department" in result.data["constant_columns"]
    assert "user_id" in result.data["possible_id_columns"]
    assert "free_text" in result.data["high_cardinality_columns"]
    assert any(row["check"] == "missing_values" for row in result.table)


def test_value_counts_returns_top_categories() -> None:
    result = execute_tool(
        _sample_dataframe(), "value_counts", {"column": "department", "top_n": 2}
    )

    assert result.status == "success"
    assert result.table == [
        {"value": "Engineering", "count": 2, "percent": 40.0},
        {"value": "Sales", "count": 2, "percent": 40.0},
    ]
    assert result.data == {
        "column": "department",
        "unique_count": 3,
        "non_null_count": 5,
        "top_n": 2,
    }


def test_aggregate_metric_groups_numeric_metric() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "aggregate_metric",
        {"metric_column": "salary", "group_by": "department", "operation": "mean"},
    )

    assert result.status == "success"
    assert result.table[0] == {"department": "Engineering", "mean_salary": 1350.0}


def test_compare_groups_returns_group_statistics() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "compare_groups",
        {"metric_column": "salary", "group_by": "department", "operation": "mean"},
    )

    assert result.status == "success"
    assert result.data["overall_mean"] == 1137.5
    assert result.table[0]["department"] == "Engineering"
    assert result.table[0]["mean_salary"] == 1350.0
    assert result.table[0]["median_salary"] == 1350.0
    assert result.table[0]["diff_from_overall_mean"] == 212.5


def test_aggregate_metric_rejects_missing_column() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "aggregate_metric",
        {"metric_column": "unknown", "group_by": "department", "operation": "mean"},
    )

    assert result.status == "error"
    assert "does not exist" in result.message


def test_sort_values_returns_ordered_rows() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "sort_values",
        {"column": "salary", "ascending": False, "limit": 2},
    )

    assert result.status == "success"
    assert [row["salary"] for row in result.table] == [1500.0, 1200.0]


def test_outlier_detection_returns_iqr_outliers() -> None:
    dataframe = pd.DataFrame(
        {
            "department": ["A", "A", "B", "B", "C", "C"],
            "salary": [10, 11, 12, 13, 14, 100],
        }
    )

    result = execute_tool(dataframe, "outlier_detection", {"column": "salary"})

    assert result.status == "success"
    assert result.data["method"] == "iqr"
    assert result.data["outlier_count"] == 1
    assert result.table[0]["salary"] == 100
    assert result.table[0]["row_index"] == 5


def test_outlier_detection_rejects_non_numeric_column() -> None:
    result = execute_tool(
        _sample_dataframe(), "outlier_detection", {"column": "department"}
    )

    assert result.status == "error"
    assert "must be numeric" in result.message


def test_filter_rows_supports_numeric_operator() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "filter_rows",
        {"column": "salary", "operator": "gt", "value": 1000},
    )

    assert result.status == "success"
    assert result.data == {"matched_rows": 2, "returned_rows": 2, "total_rows": 5}
    assert {row["department"] for row in result.table} == {"Engineering"}


def test_conditional_percentage_returns_percent_of_valid_rows() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "conditional_percentage",
        {"column": "salary", "operator": "lt", "value": 1000},
    )

    assert result.status == "success"
    assert result.data == {
        "column": "salary",
        "operator": "lt",
        "value": 1000,
        "matched_rows": 2,
        "valid_rows": 4,
        "total_rows": 5,
        "percent_of_valid": 50.0,
        "percent_of_rows": 40.0,
    }
    assert result.table is None


def test_filter_rows_supports_contains_operator() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "filter_rows",
        {"column": "department", "operator": "contains", "value": "eng"},
    )

    assert result.status == "success"
    assert result.data["matched_rows"] == 2


def test_correlation_analysis_returns_matrix() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "correlation_analysis",
        {"columns": ["salary", "tenure_years", "performance_score"]},
    )

    assert result.status == "success"
    assert result.table[0]["column"] == "salary"
    assert "tenure_years" in result.table[0]


def test_generate_chart_spec_validates_bar_chart_columns() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "generate_chart_spec",
        {"chart_type": "bar", "x": "department", "y": "salary"},
    )

    assert result.status == "success"
    assert result.chart_spec == {"chart_type": "bar", "x": "department", "y": "salary"}


def test_generate_chart_spec_rejects_scatter_with_non_numeric_x() -> None:
    result = execute_tool(
        _sample_dataframe(),
        "generate_chart_spec",
        {"chart_type": "scatter", "x": "department", "y": "salary"},
    )

    assert result.status == "error"
    assert "must be numeric" in result.message


def test_generate_chart_spec_rejects_high_cardinality_pie() -> None:
    dataframe = pd.DataFrame(
        {"category": [f"item_{index}" for index in range(11)], "amount": range(11)}
    )

    result = execute_tool(
        dataframe,
        "generate_chart_spec",
        {"chart_type": "pie", "x": "category", "y": "amount"},
    )

    assert result.status == "error"
    assert "10 or fewer" in result.message
