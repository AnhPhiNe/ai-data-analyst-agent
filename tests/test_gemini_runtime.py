import pandas as pd
import pytest

from backend.agent.gemini_runtime import (
    TransientLLMError,
    build_tool_selection_prompt,
    choose_tool_with_gemini,
    parse_tool_selection_response,
)


class FakeProvider:
    def __init__(self, responses: list[str] | None = None, errors: list[Exception] | None = None) -> None:
        self.responses = responses or []
        self.errors = errors or []
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        return self.responses.pop(0)


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "HR"],
            "salary": [1200.0, 900.0, 1000.0],
            "tenure_years": [2, 1, 3],
        }
    )


def test_build_tool_selection_prompt_contains_schema_and_tools() -> None:
    prompt = build_tool_selection_prompt(_sample_dataframe(), "Tính trung bình salary theo department")

    assert "salary" in prompt
    assert "department" in prompt
    assert "aggregate_metric" in prompt
    assert "Không sinh hoặc yêu cầu chạy code Python" in prompt


def test_parse_tool_selection_response_accepts_plain_json() -> None:
    selection = parse_tool_selection_response(
        '{"action":"tool_call","confidence":0.91,"tool_name":"value_counts","arguments":{"column":"department"}}'
    )

    assert selection.action == "tool_call"
    assert selection.tool_name == "value_counts"
    assert selection.arguments == {"column": "department"}


def test_parse_tool_selection_response_accepts_markdown_json_block() -> None:
    selection = parse_tool_selection_response(
        '```json\n{"action":"clarify","confidence":0.4,"message":"Bạn muốn dùng cột nào?"}\n```'
    )

    assert selection.action == "clarify"
    assert selection.message == "Bạn muốn dùng cột nào?"


def test_parse_tool_selection_response_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_tool_selection_response("Tôi sẽ dùng tool aggregate_metric")


def test_choose_tool_with_gemini_returns_validated_tool_call() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.93,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}'
        ]
    )

    result = choose_tool_with_gemini(_sample_dataframe(), "Tính trung bình salary theo department", provider)

    assert result.status == "tool_call"
    assert result.tool_name == "aggregate_metric"
    assert result.arguments == {
        "metric_column": "salary",
        "group_by": "department",
        "operation": "mean",
        "limit": 20,
    }


def test_choose_tool_with_gemini_returns_clarify_for_invalid_tool_call() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.9,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"unknown","group_by":"department"}}'
        ]
    )

    result = choose_tool_with_gemini(_sample_dataframe(), "Tính trung bình unknown theo department", provider)

    assert result.status == "clarify"
    assert "does not exist" in result.message


def test_choose_tool_with_gemini_returns_clarify_action() -> None:
    provider = FakeProvider(
        responses=['{"action":"clarify","confidence":0.3,"message":"Bạn muốn tính metric nào?"}']
    )

    result = choose_tool_with_gemini(_sample_dataframe(), "Tính trung bình theo nhóm", provider)

    assert result.status == "clarify"
    assert result.message == "Bạn muốn tính metric nào?"


def test_choose_tool_with_gemini_retries_transient_errors() -> None:
    provider = FakeProvider(
        responses=['{"action":"answer","confidence":0.7,"message":"Không cần gọi tool."}'],
        errors=[TransientLLMError("503 unavailable"), None],
    )
    sleeps: list[float] = []

    result = choose_tool_with_gemini(
        _sample_dataframe(),
        "Có cần gọi tool không?",
        provider,
        sleep_fn=sleeps.append,
    )

    assert result.status == "answer"
    assert len(provider.prompts) == 2
    assert sleeps == [0.25]


def test_choose_tool_with_gemini_returns_friendly_error_after_retries() -> None:
    provider = FakeProvider(errors=[TransientLLMError("429"), TransientLLMError("429"), TransientLLMError("429")])

    result = choose_tool_with_gemini(_sample_dataframe(), "Test", provider, sleep_fn=lambda _: None, max_retries=1)

    assert result.status == "error"
    assert "quá tải" in result.message
