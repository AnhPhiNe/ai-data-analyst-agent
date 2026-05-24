from collections.abc import Callable
from dataclasses import dataclass
import json
import time
from typing import Any, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.tool_validation import validate_tool_call
from backend.tools.safe_pandas import TOOL_REGISTRY


AgentAction = Literal["tool_call", "clarify", "answer"]


class LLMRuntimeError(RuntimeError):
    """Raised when the LLM provider cannot return a usable response."""


class TransientLLMError(LLMRuntimeError):
    """Raised for retryable LLM provider failures."""


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Return raw model text for a prompt."""


class AgentToolSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    confidence: float = Field(ge=0.0, le=1.0)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


@dataclass(frozen=True)
class GeminiRuntimeResult:
    status: Literal["tool_call", "clarify", "answer", "error"]
    message: str
    confidence: float = 0.0
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    raw_response: str | None = None


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            from google import genai
        except ImportError as exc:
            raise LLMRuntimeError("google-genai is not installed.") from exc

        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(model=self.model, contents=prompt)
        except Exception as exc:
            if _is_retryable_exception(exc):
                raise TransientLLMError(str(exc)) from exc
            raise LLMRuntimeError(str(exc)) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMRuntimeError("Gemini returned an empty response.")
        return str(text)


def choose_tool_with_gemini(
    dataframe: pd.DataFrame,
    question: str,
    provider: LLMProvider,
    profile_summary: dict[str, Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = 2,
) -> GeminiRuntimeResult:
    prompt = build_tool_selection_prompt(dataframe, question, profile_summary)

    try:
        raw_response = _generate_with_retry(provider, prompt, sleep_fn=sleep_fn, max_retries=max_retries)
        selection = parse_tool_selection_response(raw_response)
    except TransientLLMError:
        return GeminiRuntimeResult(
            status="error",
            message="Gemini đang tạm thời quá tải. Vui lòng thử lại sau.",
        )
    except (LLMRuntimeError, ValueError) as exc:
        return GeminiRuntimeResult(status="error", message=f"Không thể xử lý phản hồi Gemini: {exc}")

    if selection.action == "clarify":
        return GeminiRuntimeResult(
            status="clarify",
            message=selection.message or "Bạn có thể nói rõ hơn metric hoặc cột muốn phân tích không?",
            confidence=selection.confidence,
            raw_response=raw_response,
        )

    if selection.action == "answer":
        return GeminiRuntimeResult(
            status="answer",
            message=selection.message or "Câu hỏi này không cần gọi tool.",
            confidence=selection.confidence,
            raw_response=raw_response,
        )

    if not selection.tool_name:
        return GeminiRuntimeResult(
            status="error",
            message="Gemini chọn tool_call nhưng không cung cấp tool_name.",
            confidence=selection.confidence,
            raw_response=raw_response,
        )

    validation = validate_tool_call(dataframe, selection.tool_name, selection.arguments)
    if not validation.is_valid:
        return GeminiRuntimeResult(
            status="clarify",
            message=f"Tool call chưa hợp lệ: {validation.message}",
            confidence=selection.confidence,
            raw_response=raw_response,
        )

    return GeminiRuntimeResult(
        status="tool_call",
        message=selection.message or "Gemini selected a validated tool call.",
        confidence=selection.confidence,
        tool_name=selection.tool_name,
        arguments=validation.normalized_arguments,
        raw_response=raw_response,
    )


def build_tool_selection_prompt(
    dataframe: pd.DataFrame,
    question: str,
    profile_summary: dict[str, Any] | None = None,
) -> str:
    schema = [
        {"name": str(column), "dtype": str(dataframe[column].dtype)}
        for column in dataframe.columns
    ]
    tools = [
        {"name": name, "description": definition.description}
        for name, definition in TOOL_REGISTRY.items()
    ]
    safe_profile = profile_summary or {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": [str(column) for column in dataframe.columns],
    }

    context = {
        "user_question": question,
        "dataset_schema": schema,
        "profile_summary": safe_profile,
        "available_tools": tools,
        "response_contract": {
            "action": "tool_call | clarify | answer",
            "confidence": "float from 0 to 1",
            "tool_name": "tool name when action is tool_call, otherwise null",
            "arguments": "JSON object, empty object if not needed",
            "message": "short Vietnamese message",
        },
    }

    return (
        "Bạn là runtime planner cho AI Data Analyst Agent.\n"
        "Chỉ chọn tool trong available_tools. Không tự bịa số liệu. Không sinh hoặc yêu cầu chạy code Python.\n"
        "Không truy cập internet, file hệ thống, API key hoặc biến môi trường.\n"
        "Nếu thiếu metric/group/cột hoặc confidence thấp, trả action='clarify'.\n"
        "Nếu cần dữ liệu thật để trả lời, trả action='tool_call'.\n"
        "Chỉ trả về một JSON object hợp lệ, không markdown.\n\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )


def parse_tool_selection_response(raw_response: str) -> AgentToolSelection:
    payload = _extract_json_object(raw_response)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not valid JSON.") from exc

    try:
        return AgentToolSelection.model_validate(data)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(item) for item in first_error["loc"])
        raise ValueError(f"Invalid selection field '{field}': {first_error['msg']}") from exc


def _generate_with_retry(
    provider: LLMProvider,
    prompt: str,
    sleep_fn: Callable[[float], None],
    max_retries: int,
) -> str:
    attempt = 0
    while True:
        try:
            return provider.generate(prompt)
        except TransientLLMError:
            if attempt >= max_retries:
                raise
            sleep_fn(0.25 * (2**attempt))
            attempt += 1


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Response does not contain a JSON object.")
    return text[start : end + 1]


def _is_retryable_exception(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("429", "503", "timeout", "temporarily", "unavailable"))
