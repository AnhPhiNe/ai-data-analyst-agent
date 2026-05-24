import pandas as pd

from backend.agent.router import ROUTER_CONFIDENCE_THRESHOLD, route_question


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", "HR"],
            "salary": [1200.0, 900.0, 1500.0, 1000.0],
            "tenure_years": [2, 1, 5, 3],
            "performance_score": [4.5, 3.8, 4.9, 4.1],
        }
    )


def test_router_routes_row_count_to_profile_dataset() -> None:
    decision = route_question(_sample_dataframe(), "Dataset có bao nhiêu dòng?")

    assert decision.should_use_tool
    assert decision.tool_name == "profile_dataset"
    assert decision.arguments == {}


def test_router_routes_list_columns() -> None:
    decision = route_question(_sample_dataframe(), "Dataset có những cột nào?")

    assert decision.should_use_tool
    assert decision.tool_name == "list_columns"


def test_router_routes_missing_values() -> None:
    decision = route_question(_sample_dataframe(), "Cột nào thiếu dữ liệu?")

    assert decision.should_use_tool
    assert decision.tool_name == "detect_missing_values"


def test_router_routes_describe_numeric_column() -> None:
    decision = route_question(_sample_dataframe(), "Mô tả cột salary")

    assert decision.should_use_tool
    assert decision.tool_name == "describe_numeric"
    assert decision.arguments == {"column": "salary"}


def test_router_clarifies_describe_non_numeric_column() -> None:
    decision = route_question(_sample_dataframe(), "Mô tả cột department")

    assert decision.route_type == "clarify"
    assert decision.should_use_tool is False


def test_router_routes_value_counts() -> None:
    decision = route_question(_sample_dataframe(), "Top 2 department phổ biến nhất")

    assert decision.should_use_tool
    assert decision.tool_name == "value_counts"
    assert decision.arguments == {"column": "department", "top_n": 2}


def test_router_routes_average_metric_by_group() -> None:
    decision = route_question(_sample_dataframe(), "Tính trung bình salary theo department")

    assert decision.should_use_tool
    assert decision.tool_name == "aggregate_metric"
    assert decision.arguments == {
        "metric_column": "salary",
        "group_by": "department",
        "operation": "mean",
    }


def test_router_routes_sum_metric_by_group() -> None:
    decision = route_question(_sample_dataframe(), "Tổng salary theo department")

    assert decision.should_use_tool
    assert decision.tool_name == "aggregate_metric"
    assert decision.arguments["operation"] == "sum"


def test_router_routes_chart_metric_by_group() -> None:
    decision = route_question(_sample_dataframe(), "Vẽ biểu đồ salary theo department")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "bar", "x": "department", "y": "salary"}


def test_router_routes_histogram() -> None:
    decision = route_question(_sample_dataframe(), "Vẽ histogram phân phối salary")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "histogram", "x": "salary"}


def test_router_clarifies_ambiguous_aggregate_question() -> None:
    decision = route_question(_sample_dataframe(), "Tính trung bình theo nhóm")

    assert decision.route_type == "clarify"
    assert decision.confidence < ROUTER_CONFIDENCE_THRESHOLD


def test_router_falls_back_for_low_confidence_question() -> None:
    decision = route_question(_sample_dataframe(), "Có insight gì thú vị không?")

    assert decision.route_type == "fallback"
    assert decision.should_use_tool is False
