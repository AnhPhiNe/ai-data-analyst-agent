import pandas as pd

from backend.agent.router import ROUTER_CONFIDENCE_THRESHOLD, route_question


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "Engineering", "HR"],
            "salary": [1200.0, 900.0, 1500.0, 1000.0],
            "tenure_years": [2, 1, 5, 3],
            "performance_score": [4.5, 3.8, 4.9, 4.1],
            "Extracurricular_Activities": ["Yes", "No", "Yes", "Yes"],
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


def test_router_prioritizes_missing_over_list_columns_phrase() -> None:
    decision = route_question(_sample_dataframe(), "Nhung cot nao thieu du lieu?")

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
    assert decision.arguments == {"chart_type": "histogram", "x": "salary", "bins": 4}


def test_router_routes_distribution_question_to_histogram() -> None:
    decision = route_question(_sample_dataframe(), "Phân phối của salary trông như thế nào?")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "histogram", "x": "salary", "bins": 4}


def test_router_fuzzy_matches_typo_for_distribution_column() -> None:
    decision = route_question(_sample_dataframe(), "Phân phối của performance_scor thế nào?")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "histogram", "x": "performance_score", "bins": 4}


def test_router_uses_dynamic_histogram_bins_for_larger_distribution() -> None:
    dataframe = pd.DataFrame({"score": list(range(100))})

    decision = route_question(dataframe, "Phân phối của score trông như thế nào?")

    assert decision.should_use_tool
    assert decision.arguments == {"chart_type": "histogram", "x": "score", "bins": 10}


def test_router_routes_pairwise_correlation_question() -> None:
    dataframe = pd.DataFrame(
        {
            "Attendance": [70, 80, 90, 95],
            "Exam_Score": [60, 70, 85, 90],
            "Hours_Studied": [8, 12, 15, 20],
        }
    )

    decision = route_question(dataframe, "Attendance co tuong quan voi Exam_Score khong?")

    assert decision.should_use_tool
    assert decision.tool_name == "correlation_analysis"
    assert decision.arguments == {"columns": ["Attendance", "Exam_Score"]}


def test_router_prioritizes_correlation_question_over_list_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "Hours_Studied": [1, 2, 3, 4],
            "Sleep_Hours": [9, 8, 7, 6],
            "Exam_Score": [60, 70, 80, 90],
        }
    )

    decision = route_question(dataframe, "Nhung cot nao co tuong quan am voi cot diem?")

    assert decision.should_use_tool
    assert decision.tool_name == "correlation_analysis"
    assert decision.arguments == {"columns": ["Exam_Score", "Hours_Studied", "Sleep_Hours"]}


def test_router_keeps_correlation_heatmap_as_chart_request() -> None:
    dataframe = pd.DataFrame(
        {
            "Attendance": [70, 80, 90, 95],
            "Exam_Score": [60, 70, 85, 90],
        }
    )

    decision = route_question(dataframe, "Ve heatmap tuong quan")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "correlation_heatmap"}


def test_router_falls_back_when_chart_and_aggregate_intents_conflict() -> None:
    decision = route_question(_sample_dataframe(), "Ve bieu do salary trung binh theo department")

    assert decision.route_type == "fallback"
    assert decision.should_use_tool is False
    assert "conflicting intents" in str(decision.message)


def test_router_routes_numeric_percentage_condition() -> None:
    decision = route_question(_sample_dataframe(), "Tỷ lệ nhân viên có salary dưới 1000 là bao nhiêu?")

    assert decision.should_use_tool
    assert decision.tool_name == "conditional_percentage"
    assert decision.arguments == {"column": "salary", "operator": "lt", "value": 1000}


def test_router_routes_score_alias_for_grouped_average() -> None:
    dataframe = pd.DataFrame(
        {
            "Parental_Involvement": ["Low", "High", "High", "Medium"],
            "Exam_Score": [60, 80, 90, 75],
            "Hours_Studied": [10, 20, 25, 15],
        }
    )

    decision = route_question(dataframe, "Điểm trung bình theo Parental_Involvement là bao nhiêu?")

    assert decision.should_use_tool
    assert decision.tool_name == "aggregate_metric"
    assert decision.arguments == {
        "metric_column": "Exam_Score",
        "group_by": "Parental_Involvement",
        "operation": "mean",
    }


def test_router_resolves_vietnamese_metric_tokens_for_non_student_schema() -> None:
    dataframe = pd.DataFrame(
        {
            "Region": ["North", "South", "North", "West"],
            "Monthly_Revenue": [1200, 900, 1500, 1000],
            "Customer_Age": [22, 35, 41, 29],
        }
    )

    decision = route_question(dataframe, "Doanh thu hang thang trung binh theo Region la bao nhieu?")

    assert decision.should_use_tool
    assert decision.tool_name == "aggregate_metric"
    assert decision.arguments == {
        "metric_column": "Monthly_Revenue",
        "group_by": "Region",
        "operation": "mean",
    }


def test_router_resolves_vietnamese_single_metric_when_multiple_numeric_columns_exist() -> None:
    dataframe = pd.DataFrame(
        {
            "Customer_Age": [22, 35, 41, 29],
            "Monthly_Revenue": [1200, 900, 1500, 1000],
            "Order_Count": [3, 1, 4, 2],
        }
    )

    decision = route_question(dataframe, "Do tuoi khach hang trung binh la bao nhieu?")

    assert decision.should_use_tool
    assert decision.tool_name == "describe_numeric"
    assert decision.arguments == {"column": "Customer_Age"}


def test_router_resolves_vietnamese_distribution_column_for_generic_schema() -> None:
    dataframe = pd.DataFrame(
        {
            "Sleep_Duration": [5.5, 6.0, 7.5, 8.0, 6.5],
            "Screen_Time": [2, 5, 4, 6, 3],
        }
    )

    decision = route_question(dataframe, "Phan phoi giac ngu the nao?")

    assert decision.should_use_tool
    assert decision.tool_name == "generate_chart_spec"
    assert decision.arguments == {"chart_type": "histogram", "x": "Sleep_Duration", "bins": 5}


def test_router_routes_binary_category_percentage_condition() -> None:
    decision = route_question(
        _sample_dataframe(),
        'Tỷ lệ phần trăm học sinh tham gia "Extracurricular_Activities" là bao nhiêu?',
    )

    assert decision.should_use_tool
    assert decision.tool_name == "conditional_percentage"
    assert decision.arguments == {"column": "Extracurricular_Activities", "operator": "eq", "value": "Yes"}


def test_router_routes_negative_binary_category_percentage_condition() -> None:
    decision = route_question(
        _sample_dataframe(),
        'Tỷ lệ phần trăm học sinh không tham gia "Extracurricular_Activities" là bao nhiêu?',
    )

    assert decision.should_use_tool
    assert decision.tool_name == "conditional_percentage"
    assert decision.arguments == {"column": "Extracurricular_Activities", "operator": "eq", "value": "No"}


def test_router_routes_single_column_average_to_numeric_summary() -> None:
    decision = route_question(_sample_dataframe(), "Tỷ lệ phần trăm performance_score trung bình là bao nhiêu?")

    assert decision.should_use_tool
    assert decision.tool_name == "describe_numeric"
    assert decision.arguments == {"column": "performance_score"}


def test_router_clarifies_ambiguous_aggregate_question() -> None:
    decision = route_question(_sample_dataframe(), "Tính trung bình theo nhóm")

    assert decision.route_type == "clarify"
    assert decision.confidence < ROUTER_CONFIDENCE_THRESHOLD


def test_router_falls_back_for_low_confidence_question() -> None:
    decision = route_question(_sample_dataframe(), "Có insight gì thú vị không?")

    assert decision.route_type == "fallback"
    assert decision.should_use_tool is False
