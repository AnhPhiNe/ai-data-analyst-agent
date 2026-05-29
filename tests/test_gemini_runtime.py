import json

import httpx
import pandas as pd
import pytest

from backend.agent.gemini_runtime import (
    GroqProvider,
    LLMRuntimeError,
    TransientLLMError,
    build_tool_selection_prompt,
    choose_tool_with_gemini,
    parse_tool_selection_response,
)


from tests.conftest import FakeProvider


class StructuredSchemaFallbackProvider:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.structured_calls = 0

    def generate_structured(self, prompt: str, response_schema: object) -> str:
        self.structured_calls += 1
        raise LLMRuntimeError(
            "Schema properties.arguments.additionalProperties Extra inputs are not permitted"
        )

    def generate(self, prompt: str) -> str:
        self.generate_calls += 1
        return (
            '{"action":"tool_call","confidence":0.91,"tool_name":"value_counts",'
            '"arguments":{"column":"department","top_n":2}}'
        )


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales", "HR"],
            "salary": [1200.0, 900.0, 1000.0],
            "tenure_years": [2, 1, 3],
        }
    )


def test_build_tool_selection_prompt_contains_schema_and_tools() -> None:
    prompt = build_tool_selection_prompt(
        _sample_dataframe(), "Tính trung bình salary theo department"
    )

    assert "salary" in prompt
    assert "department" in prompt
    assert "column_data_dictionary" in prompt
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


def test_groq_provider_uses_json_object_mode_for_structured_generation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"action":"clarify","confidence":0.4,'
                                '"message":"Ban muon cot nao?"}'
                            )
                        }
                    }
                ]
            },
        )

    provider = GroqProvider(
        api_key="test-key",
        model="qwen/qwen3-32b",
        transport=httpx.MockTransport(handler),
    )

    response = provider.generate_structured("Return JSON", object)  # type: ignore[arg-type]
    payload = json.loads(requests[0].content)

    assert "Ban muon cot nao" in response
    assert payload["model"] == "qwen/qwen3-32b"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_format"] == "hidden"
    assert payload["reasoning_effort"] == "none"


def test_groq_provider_omits_reasoning_controls_for_llama_models() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"action":"clarify","confidence":0.4,'
                                '"message":"Ban muon cot nao?"}'
                            )
                        }
                    }
                ]
            },
        )

    provider = GroqProvider(
        api_key="test-key",
        model="llama-3.3-70b-versatile",
        transport=httpx.MockTransport(handler),
    )

    provider.generate_structured("Return JSON", object)  # type: ignore[arg-type]
    payload = json.loads(requests[0].content)

    assert payload["model"] == "llama-3.3-70b-versatile"
    assert payload["response_format"] == {"type": "json_object"}
    assert "reasoning_format" not in payload
    assert "reasoning_effort" not in payload


def test_groq_provider_maps_rate_limit_to_transient_error() -> None:
    provider = GroqProvider(
        api_key="test-key",
        model="qwen/qwen3-32b",
        transport=httpx.MockTransport(lambda _: httpx.Response(429, text="rate limit")),
    )

    with pytest.raises(TransientLLMError):
        provider.generate("hello")


def test_choose_tool_with_gemini_returns_validated_tool_call() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.93,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}'
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Tính trung bình salary theo department", provider
    )

    assert result.status == "tool_call"
    assert result.tool_name == "aggregate_metric"
    assert result.arguments == {
        "metric_column": "salary",
        "group_by": "department",
        "operation": "mean",
        "limit": 20,
    }


def test_choose_tool_with_gemini_falls_back_when_structured_schema_is_rejected() -> (
    None
):
    provider = StructuredSchemaFallbackProvider()

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Co bao nhieu department?", provider
    )

    assert result.status == "tool_call"
    assert result.tool_name == "value_counts"
    assert result.arguments == {"column": "department", "top_n": 2}
    assert provider.structured_calls == 1
    assert provider.generate_calls == 1


def test_choose_tool_with_gemini_retries_invalid_tool_call_with_feedback() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.88,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"unknown","group_by":"department"}}',
            '{"action":"tool_call","confidence":0.93,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}',
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(),
        "Tính trung bình salary theo department",
        provider,
        max_validation_retries=1,
    )

    assert result.status == "tool_call"
    assert result.tool_name == "aggregate_metric"
    assert result.arguments == {
        "metric_column": "salary",
        "group_by": "department",
        "operation": "mean",
        "limit": 20,
    }
    assert result.validation_retry_count == 1
    assert len(provider.prompts) == 2
    assert "validation_error" in provider.prompts[1]


def test_choose_tool_with_gemini_returns_clarify_for_invalid_tool_call() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.9,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"unknown","group_by":"department"}}'
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Tính trung bình unknown theo department", provider
    )

    assert result.status == "clarify"
    assert "does not exist" in result.message


def test_choose_tool_with_gemini_returns_clarify_action() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"clarify","confidence":0.3,"message":"Bạn muốn tính metric nào?"}'
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Tính trung bình theo nhóm", provider
    )

    assert result.status == "clarify"
    assert result.message == "Bạn muốn tính metric nào?"


def test_choose_tool_with_gemini_retries_transient_errors() -> None:
    provider = FakeProvider(
        responses=[
            '{"action":"answer","confidence":0.7,"message":"Không cần gọi tool."}'
        ],
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
    provider = FakeProvider(
        errors=[
            TransientLLMError(
                '{"error":{"message":"Rate limit reached for model qwen/qwen3-32b",'
                '"type":"tokens","code":"rate_limit_exceeded"}}'
            ),
            TransientLLMError(
                '{"error":{"message":"Rate limit reached for model qwen/qwen3-32b",'
                '"type":"tokens","code":"rate_limit_exceeded"}}'
            ),
            TransientLLMError(
                '{"error":{"message":"Rate limit reached for model qwen/qwen3-32b",'
                '"type":"tokens","code":"rate_limit_exceeded"}}'
            ),
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Test", provider, sleep_fn=lambda _: None, max_retries=1
    )

    assert result.status == "error"
    assert "rate_limit" in result.message
    assert "Rate limit reached" in result.message
    assert "qwen/qwen3-32b" in result.message


def test_choose_tool_with_gemini_redacts_provider_error_secrets() -> None:
    provider = FakeProvider(
        errors=[
            TransientLLMError("429 API key AIza123456789012345678901234567890 leaked"),
            TransientLLMError("429 API key AIza123456789012345678901234567890 leaked"),
            TransientLLMError("429 API key AIza123456789012345678901234567890 leaked"),
        ]
    )

    result = choose_tool_with_gemini(
        _sample_dataframe(), "Test", provider, sleep_fn=lambda _: None, max_retries=1
    )

    assert result.status == "error"
    assert "[REDACTED_API_KEY]" in result.message
    assert "AIza123456" not in result.message
