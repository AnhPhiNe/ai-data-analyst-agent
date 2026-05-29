import pandas as pd

from backend.agent.multi_step_planner import (
    MultiStepPlan,
    MultiStepToolCall,
    plan_multi_step_question,
    validate_multi_step_plan,
)
from backend.agent.gemini_runtime import LLMRuntimeError


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "department": ["Engineering", "Sales", "Engineering", "HR", "Sales"],
            "salary": [1000, 1100, 1200, 900, 10000],
            "Exam_Score": [80, 75, 85, 70, 95],
            "Attendance": [90, 85, 92, 80, 98],
        }
    )


def test_deterministic_planner_detects_compare_and_outlier() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "Nhóm nào có salary trung bình cao nhất và có outlier không?",
    )

    assert result.status == "plan"
    assert result.plan is not None
    assert [step.tool_name for step in result.plan.steps] == [
        "compare_groups",
        "outlier_detection",
    ]
    assert result.plan.steps[0].arguments["metric_column"] == "salary"
    assert result.plan.steps[0].arguments["group_by"] == "department"


def test_deterministic_planner_detects_compare_and_chart() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "So sánh salary theo department và vẽ biểu đồ",
    )

    assert result.status == "plan"
    assert result.plan is not None
    assert [step.tool_name for step in result.plan.steps] == [
        "compare_groups",
        "generate_chart_spec",
    ]


def test_deterministic_planner_skips_chart_only_aggregate_request() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "Vẽ biểu đồ salary trung bình theo department",
    )

    assert result.status == "skip"


def test_deterministic_planner_skips_scatter_comparison_request() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "So sánh salary và Exam_Score bằng scatter",
    )

    assert result.status == "skip"


def test_deterministic_planner_clarifies_vague_compare_chart_request() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "So sánh dữ liệu này và vẽ biểu đồ",
    )

    assert result.status == "clarify"


def test_deterministic_planner_detects_data_quality_recommendation() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "Cột nào giống ID và cột nào nên dùng để phân tích?",
    )

    assert result.status == "plan"
    assert result.plan is not None
    assert result.plan.steps[0].tool_name == "data_quality_report"


def test_deterministic_planner_skips_single_step_target_correlation() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "Cột nào tương quan mạnh với Exam_Score?",
    )

    assert result.status == "skip"


def test_plan_validation_rejects_unsupported_tool() -> None:
    plan = MultiStepPlan(
        source="llm",
        confidence=0.95,
        message="bad",
        steps=[
            MultiStepToolCall(
                tool_name="run_python",
                arguments={"code": "print('x')"},
                purpose="bad",
            )
        ],
    )

    is_valid, message = validate_multi_step_plan(_sample_dataframe(), plan)

    assert is_valid is False
    assert "unsupported tool" in message


def test_plan_validation_rejects_too_many_steps() -> None:
    plan = MultiStepPlan(
        source="llm",
        confidence=0.95,
        message="too many",
        steps=[
            MultiStepToolCall(tool_name="list_columns", arguments={}, purpose="1"),
            MultiStepToolCall(tool_name="profile_dataset", arguments={}, purpose="2"),
            MultiStepToolCall(
                tool_name="detect_missing_values", arguments={}, purpose="3"
            ),
            MultiStepToolCall(
                tool_name="data_quality_report", arguments={}, purpose="4"
            ),
        ],
    )

    is_valid, message = validate_multi_step_plan(_sample_dataframe(), plan)

    assert is_valid is False
    assert "at most 3" in message


def test_plan_validation_rejects_invalid_arguments() -> None:
    plan = MultiStepPlan(
        source="llm",
        confidence=0.95,
        message="invalid",
        steps=[
            MultiStepToolCall(
                tool_name="outlier_detection",
                arguments={"column": "department"},
                purpose="wrong dtype",
            )
        ],
    )

    is_valid, message = validate_multi_step_plan(_sample_dataframe(), plan)

    assert is_valid is False
    assert "must be numeric" in message


class SchemaFailingProvider:
    def generate_structured(self, prompt: str, response_schema: object) -> str:
        raise LLMRuntimeError(
            "Schema properties.steps.items.properties.arguments.additionalProperties "
            "Extra inputs are not permitted"
        )

    def generate(self, prompt: str) -> str:
        return (
            '{"action":"plan","confidence":0.91,"steps":['
            '{"tool_name":"describe_numeric","arguments":{"column":"salary"},'
            '"purpose":"Describe salary"},'
            '{"tool_name":"generate_chart_spec",'
            '"arguments":{"chart_type":"histogram","x":"salary","bins":5},'
            '"purpose":"Chart salary"}],"message":"ok"}'
        )


def test_llm_multi_step_planner_falls_back_when_structured_schema_is_rejected() -> None:
    result = plan_multi_step_question(
        _sample_dataframe(),
        "Mo ta salary roi ve histogram",
        provider=SchemaFailingProvider(),
    )

    assert result.status == "plan"
    assert result.plan is not None
    assert [step.tool_name for step in result.plan.steps] == [
        "describe_numeric",
        "generate_chart_spec",
    ]
