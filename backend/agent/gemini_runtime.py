from collections.abc import Callable
from dataclasses import dataclass
import json
import time
from typing import Any, Literal, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.column_argument_repair import repair_tool_column_arguments
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
    validation_retry_count: int = 0


class GeminiProvider:
    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        try:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        except ImportError as exc:
            raise LLMRuntimeError("google-genai is not installed.") from exc

    def generate(self, prompt: str) -> str:
        try:
            from google import genai

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.0),
            )
        except Exception as exc:
            if _is_retryable_exception(exc):
                raise TransientLLMError(str(exc)) from exc
            raise LLMRuntimeError(str(exc)) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMRuntimeError("Gemini returned an empty response.")
        return str(text)

    def generate_structured(self, prompt: str, response_schema: type[BaseModel]) -> str:
        try:
            from google import genai

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
        except TypeError:
            return self.generate(prompt)
        except Exception as exc:
            if _is_retryable_exception(exc):
                raise TransientLLMError(str(exc)) from exc
            raise LLMRuntimeError(str(exc)) from exc

        text = getattr(response, "text", None)
        if not text:
            raise LLMRuntimeError("Gemini returned an empty structured response.")
        return str(text)


def choose_tool_with_gemini(
    dataframe: pd.DataFrame,
    question: str,
    provider: LLMProvider,
    profile_summary: dict[str, Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_retries: int = 2,
    max_validation_retries: int = 1,
) -> GeminiRuntimeResult:
    prompt = build_tool_selection_prompt(dataframe, question, profile_summary)

    try:
        raw_response = _generate_structured_with_retry(
            provider,
            prompt,
            AgentToolSelection,
            sleep_fn=sleep_fn,
            max_retries=max_retries,
        )
        selection = parse_tool_selection_response(raw_response)
        retry_count = 0
    except TransientLLMError:
        return GeminiRuntimeResult(
            status="error",
            message="Gemini đang tạm thời quá tải. Vui lòng thử lại sau.",
        )
    except (LLMRuntimeError, ValueError) as exc:
        return GeminiRuntimeResult(
            status="error", message=f"Không thể xử lý phản hồi Gemini: {exc}"
        )

    if selection.action == "clarify":
        return GeminiRuntimeResult(
            status="clarify",
            message=selection.message
            or "Bạn có thể nói rõ hơn metric hoặc cột muốn phân tích không?",
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

    repaired_arguments = repair_tool_column_arguments(
        dataframe, selection.tool_name, selection.arguments
    )
    validation = validate_tool_call(dataframe, selection.tool_name, repaired_arguments)

    while not validation.is_valid and retry_count < max_validation_retries:
        retry_count += 1
        retry_prompt = build_tool_selection_retry_prompt(
            dataframe=dataframe,
            question=question,
            profile_summary=profile_summary,
            previous_selection=selection,
            validation_message=validation.message,
        )
        try:
            raw_response = _generate_structured_with_retry(
                provider,
                retry_prompt,
                AgentToolSelection,
                sleep_fn=sleep_fn,
                max_retries=max_retries,
            )
            selection = parse_tool_selection_response(raw_response)
        except TransientLLMError:
            return GeminiRuntimeResult(
                status="error",
                message="Gemini đang tạm thời quá tải. Vui lòng thử lại sau.",
                validation_retry_count=retry_count,
            )
        except (LLMRuntimeError, ValueError) as exc:
            return GeminiRuntimeResult(
                status="error",
                message=f"Không thể xử lý phản hồi Gemini sau retry: {exc}",
                validation_retry_count=retry_count,
            )

        if selection.action != "tool_call" or not selection.tool_name:
            return _selection_to_runtime_result(
                selection, raw_response, validation_retry_count=retry_count
            )

        repaired_arguments = repair_tool_column_arguments(
            dataframe, selection.tool_name, selection.arguments
        )
        validation = validate_tool_call(
            dataframe, selection.tool_name, repaired_arguments
        )

    if not validation.is_valid:
        return GeminiRuntimeResult(
            status="clarify",
            message=f"Tool call chưa hợp lệ sau retry: {validation.message}",
            confidence=selection.confidence,
            raw_response=raw_response,
            validation_retry_count=retry_count,
        )

    return GeminiRuntimeResult(
        status="tool_call",
        message=(
            selection.message
            or (
                "Gemini selected a validated tool call after validation retry."
                if retry_count
                else "Gemini selected a validated tool call."
            )
        ),
        confidence=selection.confidence,
        tool_name=selection.tool_name,
        arguments=validation.normalized_arguments,
        raw_response=raw_response,
        validation_retry_count=retry_count,
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
        "column_data_dictionary": _column_data_dictionary(dataframe, safe_profile),
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


def build_tool_selection_retry_prompt(
    dataframe: pd.DataFrame,
    question: str,
    profile_summary: dict[str, Any] | None,
    previous_selection: AgentToolSelection,
    validation_message: str,
) -> str:
    retry_context = {
        "previous_invalid_selection": previous_selection.model_dump(),
        "validation_error": validation_message,
        "repair_instruction": (
            "Return one corrected JSON object. Fix only the tool_name/arguments. "
            "If the request is ambiguous or cannot be mapped to a valid tool, return action='clarify'."
        ),
    }
    return (
        build_tool_selection_prompt(dataframe, question, profile_summary)
        + "\n\nThe previous tool call failed validation. Use this feedback before answering:\n"
        + json.dumps(retry_context, ensure_ascii=False)
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
        raise ValueError(
            f"Invalid selection field '{field}': {first_error['msg']}"
        ) from exc


def _selection_to_runtime_result(
    selection: AgentToolSelection,
    raw_response: str,
    validation_retry_count: int = 0,
) -> GeminiRuntimeResult:
    if selection.action == "clarify":
        return GeminiRuntimeResult(
            status="clarify",
            message=selection.message
            or "Bạn có thể nói rõ hơn metric hoặc cột muốn phân tích không?",
            confidence=selection.confidence,
            raw_response=raw_response,
            validation_retry_count=validation_retry_count,
        )
    if selection.action == "answer":
        return GeminiRuntimeResult(
            status="answer",
            message=selection.message or "Câu hỏi này không cần gọi tool.",
            confidence=selection.confidence,
            raw_response=raw_response,
            validation_retry_count=validation_retry_count,
        )
    return GeminiRuntimeResult(
        status="error",
        message="Gemini chọn tool_call nhưng không cung cấp tool_name.",
        confidence=selection.confidence,
        raw_response=raw_response,
        validation_retry_count=validation_retry_count,
    )


def _column_data_dictionary(
    dataframe: pd.DataFrame, profile_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    metadata = profile_summary.get("column_metadata")
    if isinstance(metadata, list) and metadata:
        return [
            {
                "name": item.get("name"),
                "dtype": item.get("dtype"),
                "analysis_role": item.get("analysis_role") or item.get("inferred_kind"),
                "semantic_aliases": item.get("semantic_aliases", []),
                "sample_values": item.get("sample_values", []),
                "missing_percent": item.get("missing_percent"),
                "unique_count": item.get("unique_count"),
            }
            for item in metadata
            if isinstance(item, dict)
        ]
    return [
        {
            "name": str(column),
            "dtype": str(dataframe[column].dtype),
            "analysis_role": "numeric_metric"
            if pd.api.types.is_numeric_dtype(dataframe[column])
            else "categorical_dimension",
            "semantic_aliases": [],
            "sample_values": [],
            "missing_percent": None,
            "unique_count": None,
        }
        for column in dataframe.columns
    ]


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


def _generate_structured_with_retry(
    provider: LLMProvider,
    prompt: str,
    response_schema: type[BaseModel],
    sleep_fn: Callable[[float], None],
    max_retries: int,
) -> str:
    generate_structured = getattr(provider, "generate_structured", None)
    if not callable(generate_structured):
        return _generate_with_retry(
            provider, prompt, sleep_fn=sleep_fn, max_retries=max_retries
        )

    attempt = 0
    while True:
        try:
            return str(generate_structured(prompt, response_schema))
        except TransientLLMError:
            if attempt >= max_retries:
                raise
            sleep_fn(0.25 * (2**attempt))
            attempt += 1
        except LLMRuntimeError as exc:
            if _is_structured_schema_error(exc):
                return _generate_with_retry(
                    provider, prompt, sleep_fn=sleep_fn, max_retries=max_retries
                )
            raise


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
    return any(
        token in text
        for token in ("429", "503", "timeout", "temporarily", "unavailable")
    )


def _is_structured_schema_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "additionalproperties" in text or "response_schema" in text
