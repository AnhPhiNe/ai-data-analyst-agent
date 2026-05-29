from typing import Any, Literal
import json

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.column_resolver import normalize_text
from backend.agent.gemini_runtime import (
    LLMProvider,
    LLMRuntimeError,
    TransientLLMError,
    format_llm_provider_error,
)
from backend.agent.helpers import detect_aggregation, get_histogram_bins
from backend.agent.router import (
    _find_column,
    _find_group_column,
    _find_metric_column,
    _is_numeric_column,
)
from backend.agent.tool_validation import validate_tool_call
from backend.tools.safe_pandas import TOOL_REGISTRY


MultiStepAction = Literal["plan", "clarify", "answer"]
MultiStepSource = Literal["deterministic", "llm"]
PlannerStatus = Literal["plan", "clarify", "answer", "skip", "error"]

MAX_MULTI_STEP_STEPS = 3
MIN_LLM_PLAN_CONFIDENCE = 0.75

PHASE3_MULTI_STEP_TOOLS = {
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


class MultiStepToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    purpose: str


class MultiStepPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: MultiStepSource
    confidence: float = Field(ge=0.0, le=1.0)
    steps: list[MultiStepToolCall]
    message: str


class MultiStepPlannerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PlannerStatus
    message: str | None = None
    plan: MultiStepPlan | None = None
    confidence: float = 0.0


class LLMMultiStepSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: MultiStepAction
    confidence: float = Field(ge=0.0, le=1.0)
    steps: list[MultiStepToolCall] = Field(default_factory=list)
    message: str | None = None


class PlannerIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requires_multi_step: bool
    primary_intents: list[str]
    metric_column: str | None = None
    group_column: str | None = None
    chart_requested: bool = False
    quality_requested: bool = False
    outlier_requested: bool = False
    describe_requested: bool = False
    correlation_requested: bool = False
    column_recommendation_requested: bool = False
    confidence: float
    reason: str


def plan_multi_step_question(
    dataframe: pd.DataFrame,
    question: str,
    provider: LLMProvider | None = None,
    profile_summary: dict[str, Any] | None = None,
) -> MultiStepPlannerResult:
    normalized = normalize_text(question)
    deterministic = _deterministic_plan(dataframe, question, normalized)
    if deterministic.status != "skip":
        return deterministic

    if provider is None or not _looks_like_multi_step_request(normalized):
        return MultiStepPlannerResult(status="skip")

    return _llm_multi_step_plan(dataframe, question, provider, profile_summary)


def validate_multi_step_plan(
    dataframe: pd.DataFrame,
    plan: MultiStepPlan,
) -> tuple[bool, str]:
    if len(plan.steps) < 1:
        return False, "Multi-step plan must contain at least one step."
    if len(plan.steps) > MAX_MULTI_STEP_STEPS:
        return (
            False,
            f"Multi-step plan can contain at most {MAX_MULTI_STEP_STEPS} steps.",
        )
    if plan.source == "llm" and plan.confidence < MIN_LLM_PLAN_CONFIDENCE:
        return False, "LLM multi-step plan confidence is too low."

    seen_tool_names: set[str] = set()
    for index, step in enumerate(plan.steps, start=1):
        if step.tool_name not in PHASE3_MULTI_STEP_TOOLS:
            return False, f"Step {index} uses unsupported tool '{step.tool_name}'."
        if step.tool_name not in TOOL_REGISTRY:
            return False, f"Step {index} uses unavailable tool '{step.tool_name}'."
        if step.tool_name in seen_tool_names:
            return False, f"Step {index} repeats tool '{step.tool_name}'."
        seen_tool_names.add(step.tool_name)
        validation = validate_tool_call(dataframe, step.tool_name, step.arguments)
        if not validation.is_valid:
            return False, f"Step {index} is invalid: {validation.message}"

    return True, "Multi-step plan is valid."


def _deterministic_plan(
    dataframe: pd.DataFrame,
    question: str,
    normalized: str,
) -> MultiStepPlannerResult:
    intent = _classify_planner_intent(dataframe, normalized)
    if (
        _has_compare_intent(normalized)
        and intent.chart_requested
        and "scatter" not in normalized
        and (intent.metric_column is None or intent.group_column is None)
    ):
        return _clarify(
            "Ban muon so sanh metric numeric nao theo cot nhom nao va ve bieu do nao?"
        )

    if not intent.requires_multi_step and not intent.column_recommendation_requested:
        return MultiStepPlannerResult(status="skip")

    if intent.column_recommendation_requested:
        return _plan(
            [
                MultiStepToolCall(
                    tool_name="data_quality_report",
                    arguments={},
                    purpose="Đánh giá chất lượng dữ liệu và xác định cột nên dùng.",
                )
            ],
            "Đánh giá chất lượng dữ liệu và gợi ý cột phân tích.",
            confidence=0.94,
        )

    metric_column = intent.metric_column
    group_column = intent.group_column

    if _has_intents(intent, "compare_groups", "outlier_detection"):
        if not metric_column or not group_column:
            return _clarify(
                "Mình cần biết metric numeric và cột nhóm để vừa so sánh vừa kiểm tra outlier."
            )
        operation = detect_aggregation(question) or "mean"
        return _plan(
            [
                MultiStepToolCall(
                    tool_name="compare_groups",
                    arguments={
                        "metric_column": metric_column,
                        "group_by": group_column,
                        "operation": operation if operation != "sum" else "mean",
                    },
                    purpose="So sánh metric numeric giữa các nhóm.",
                ),
                MultiStepToolCall(
                    tool_name="outlier_detection",
                    arguments={"column": metric_column, "limit": 20},
                    purpose="Kiểm tra outlier có thể ảnh hưởng đến kết quả so sánh.",
                ),
            ],
            "So sánh nhóm và kiểm tra outlier.",
            confidence=0.93,
        )

    if _has_intents(intent, "compare_groups", "generate_chart_spec"):
        if not metric_column or not group_column:
            return _clarify(
                "Mình cần biết metric numeric và cột nhóm để so sánh và vẽ biểu đồ."
            )
        return _plan(
            [
                MultiStepToolCall(
                    tool_name="compare_groups",
                    arguments={
                        "metric_column": metric_column,
                        "group_by": group_column,
                        "operation": detect_aggregation(question) or "mean",
                    },
                    purpose="Tính thống kê so sánh giữa các nhóm.",
                ),
                MultiStepToolCall(
                    tool_name="generate_chart_spec",
                    arguments={
                        "chart_type": "bar",
                        "x": group_column,
                        "y": metric_column,
                    },
                    purpose="Tạo biểu đồ so sánh metric theo nhóm.",
                ),
            ],
            "So sánh nhóm và tạo biểu đồ.",
            confidence=0.91,
        )

    if _has_intents(intent, "data_quality_report", "outlier_detection"):
        column = metric_column or _find_column(dataframe, normalized)
        if not column or not _is_numeric_column(dataframe, column):
            return _clarify(
                "Mình cần biết cột numeric để kiểm tra outlier sau báo cáo chất lượng dữ liệu."
            )
        return _plan(
            [
                MultiStepToolCall(
                    tool_name="data_quality_report",
                    arguments={},
                    purpose="Kiểm tra các tín hiệu chất lượng dữ liệu.",
                ),
                MultiStepToolCall(
                    tool_name="outlier_detection",
                    arguments={"column": column, "limit": 20},
                    purpose="Kiểm tra outlier trên cột numeric được hỏi.",
                ),
            ],
            "Kiểm tra chất lượng dữ liệu và outlier.",
            confidence=0.9,
        )

    if _has_intents(intent, "describe_numeric", "generate_chart_spec"):
        column = metric_column or _find_column(dataframe, normalized)
        if not column or not _is_numeric_column(dataframe, column):
            return _clarify("Mình cần biết một cột numeric để mô tả và vẽ histogram.")
        return _plan(
            [
                MultiStepToolCall(
                    tool_name="describe_numeric",
                    arguments={"column": column},
                    purpose="Mô tả thống kê cột numeric.",
                ),
                MultiStepToolCall(
                    tool_name="generate_chart_spec",
                    arguments={
                        "chart_type": "histogram",
                        "x": column,
                        "bins": get_histogram_bins(dataframe, column),
                    },
                    purpose="Tạo histogram cho cột numeric.",
                ),
            ],
            "Mô tả cột numeric và tạo histogram.",
            confidence=0.9,
        )

    return MultiStepPlannerResult(status="skip")


def _classify_planner_intent(dataframe: pd.DataFrame, normalized: str) -> PlannerIntent:
    quality_requested = _has_any(
        normalized,
        (
            "chat luong du lieu",
            "data quality",
            "van de du lieu",
            "missing values",
            "thieu du lieu",
        ),
    )
    column_recommendation_requested = _has_any(
        normalized, ("giong id", "cot id", "nen dung de phan tich")
    )
    outlier_requested = _has_any(normalized, ("outlier", "ngoai lai", "bat thuong"))
    chart_requested = _has_any(
        normalized, ("ve bieu do", "bieu do", "chart", "histogram", "scatter")
    )
    describe_requested = _has_any(normalized, ("mo ta", "describe", "thong ke"))
    correlation_requested = _has_any(
        normalized, ("tuong quan", "correlation", "lien he")
    )
    compare_requested = _has_compare_intent(normalized)
    group_relation_requested = _has_group_relation_intent(normalized)

    metric_column = _find_metric_column(dataframe, normalized)
    group_column = _find_group_column(
        dataframe, normalized, exclude={metric_column} if metric_column else set()
    )
    if (
        compare_requested
        and outlier_requested
        and group_column is None
        and not group_relation_requested
    ):
        group_column = _find_preferred_group_column(dataframe, exclude={metric_column})

    primary_intents: list[str] = []
    if column_recommendation_requested:
        primary_intents.append("data_quality_report")
    elif quality_requested:
        primary_intents.append("data_quality_report")
    if describe_requested:
        primary_intents.append("describe_numeric")
    if compare_requested and (group_relation_requested or outlier_requested):
        primary_intents.append("compare_groups")
    if outlier_requested:
        primary_intents.append("outlier_detection")
    if chart_requested:
        primary_intents.append("generate_chart_spec")
    if correlation_requested:
        primary_intents.append("correlation_analysis")

    primary_intents = _dedupe(primary_intents)
    requires_multi_step = _requires_multi_step(
        primary_intents=primary_intents,
        column_recommendation_requested=column_recommendation_requested,
        metric_column=metric_column,
        group_column=group_column,
        chart_requested=chart_requested,
        compare_requested=compare_requested,
        group_relation_requested=group_relation_requested,
    )

    return PlannerIntent(
        requires_multi_step=requires_multi_step,
        primary_intents=primary_intents,
        metric_column=metric_column,
        group_column=group_column,
        chart_requested=chart_requested,
        quality_requested=quality_requested,
        outlier_requested=outlier_requested,
        describe_requested=describe_requested,
        correlation_requested=correlation_requested,
        column_recommendation_requested=column_recommendation_requested,
        confidence=_intent_confidence(primary_intents, metric_column, group_column),
        reason="Classified analytical intents before deciding whether multi-step is needed.",
    )


def _requires_multi_step(
    primary_intents: list[str],
    column_recommendation_requested: bool,
    metric_column: str | None,
    group_column: str | None,
    chart_requested: bool,
    compare_requested: bool,
    group_relation_requested: bool,
) -> bool:
    if column_recommendation_requested:
        return True
    intent_set = set(primary_intents)
    if {"compare_groups", "outlier_detection"}.issubset(intent_set):
        return True
    if {"data_quality_report", "outlier_detection"}.issubset(intent_set):
        return True
    if {"describe_numeric", "generate_chart_spec"}.issubset(intent_set):
        return True
    if {"compare_groups", "generate_chart_spec"}.issubset(intent_set):
        return (
            chart_requested
            and compare_requested
            and group_relation_requested
            and metric_column is not None
            and group_column is not None
        )
    return False


def _llm_multi_step_plan(
    dataframe: pd.DataFrame,
    question: str,
    provider: LLMProvider,
    profile_summary: dict[str, Any] | None,
) -> MultiStepPlannerResult:
    prompt = _build_multi_step_prompt(dataframe, question, profile_summary)
    try:
        raw_response = _generate_structured(provider, prompt)
        selection = _parse_llm_multi_step_response(raw_response)
    except TransientLLMError as exc:
        return MultiStepPlannerResult(
            status="error",
            message=format_llm_provider_error(exc),
        )
    except (LLMRuntimeError, ValueError) as exc:
        return MultiStepPlannerResult(
            status="error",
            message=f"Không thể xử lý multi-step plan từ LLM provider: {exc}",
        )

    if selection.action == "clarify":
        return MultiStepPlannerResult(
            status="clarify",
            message=selection.message
            or "Bạn hãy nêu rõ các cột/metric cần dùng cho phân tích nhiều bước.",
            confidence=selection.confidence,
        )
    if selection.action == "answer":
        return MultiStepPlannerResult(
            status="answer",
            message=selection.message or "Câu hỏi này không cần chạy nhiều bước.",
            confidence=selection.confidence,
        )

    plan = MultiStepPlan(
        source="llm",
        confidence=selection.confidence,
        steps=selection.steps,
        message=selection.message or "LLM planner created a bounded multi-step plan.",
    )
    is_valid, validation_message = validate_multi_step_plan(dataframe, plan)
    if not is_valid:
        return MultiStepPlannerResult(
            status="clarify",
            message=f"Multi-step plan chưa đủ an toàn hoặc chưa rõ: {validation_message}",
            confidence=selection.confidence,
        )
    return MultiStepPlannerResult(status="plan", plan=plan, confidence=plan.confidence)


def _build_multi_step_prompt(
    dataframe: pd.DataFrame,
    question: str,
    profile_summary: dict[str, Any] | None,
) -> str:
    schema = [
        {"name": str(column), "dtype": str(dataframe[column].dtype)}
        for column in dataframe.columns
    ]
    context = {
        "user_question": question,
        "dataset_schema": schema,
        "profile_summary": profile_summary
        or {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "column_names": [str(column) for column in dataframe.columns],
        },
        "allowed_tools": sorted(PHASE3_MULTI_STEP_TOOLS),
        "max_steps": MAX_MULTI_STEP_STEPS,
        "response_contract": {
            "action": "plan | clarify | answer",
            "confidence": "float from 0 to 1",
            "steps": [
                {
                    "tool_name": "one allowed tool",
                    "arguments": "JSON object",
                    "purpose": "short Vietnamese purpose",
                }
            ],
            "message": "short Vietnamese message",
        },
    }
    return (
        "Bạn là bounded multi-step planner cho AI Data Analyst Agent.\n"
        "Chỉ lập plan khi câu hỏi thật sự cần nhiều thao tác phân tích dữ liệu.\n"
        "Chỉ dùng allowed_tools, tối đa 3 steps, không sinh SQL, không sinh Python/pandas code.\n"
        "Nếu thiếu tên cột/metric/group hoặc không chắc, trả action='clarify'.\n"
        "Nếu một specialist tool giải quyết đủ câu hỏi, không lập multi-step plan.\n"
        "Chỉ trả về một JSON object hợp lệ, không markdown.\n\n"
        f"{json.dumps(context, ensure_ascii=False)}"
    )


def _generate_structured(provider: LLMProvider, prompt: str) -> str:
    structured = getattr(provider, "generate_structured", None)
    if callable(structured):
        try:
            return str(structured(prompt, LLMMultiStepSelection))
        except LLMRuntimeError as exc:
            if _is_structured_schema_error(exc):
                return provider.generate(prompt)
            raise
    return provider.generate(prompt)


def _parse_llm_multi_step_response(raw_response: str) -> LLMMultiStepSelection:
    try:
        data = json.loads(_extract_json_object(raw_response))
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not valid JSON.") from exc
    try:
        return LLMMultiStepSelection.model_validate(data)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(item) for item in first_error["loc"])
        raise ValueError(f"Invalid multi-step response field '{field}'.") from exc


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Response does not contain a JSON object.")
    return stripped[start : end + 1]


def _looks_like_multi_step_request(normalized: str) -> bool:
    if not _has_any(normalized, (" va ", " roi", "sau do", "dong thoi", "kem", "and")):
        return False
    intent_count = 0
    intent_groups = (
        ("chat luong du lieu", "data quality", "missing", "thieu"),
        ("outlier", "ngoai lai", "bat thuong"),
        ("bieu do", "chart", "histogram", "scatter"),
        ("so sanh", "trung binh", "tong", "nhom"),
        ("tuong quan", "correlation"),
        ("mo ta", "describe", "thong ke"),
    )
    for group in intent_groups:
        if _has_any(normalized, group):
            intent_count += 1
    return intent_count >= 2


def _has_intents(intent: PlannerIntent, *tool_names: str) -> bool:
    intent_set = set(intent.primary_intents)
    return all(tool_name in intent_set for tool_name in tool_names)


def _has_compare_intent(normalized: str) -> bool:
    return _has_any(
        normalized,
        ("so sanh", "compare", "comparison", "nhom nao", "cao nhat", "thap nhat"),
    )


def _has_group_relation_intent(normalized: str) -> bool:
    return _has_any(normalized, (" theo ", "theo nhom", "group by", " by "))


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _intent_confidence(
    primary_intents: list[str],
    metric_column: str | None,
    group_column: str | None,
) -> float:
    if not primary_intents:
        return 0.0
    confidence = 0.75 + min(len(primary_intents), 3) * 0.05
    if metric_column is not None:
        confidence += 0.05
    if group_column is not None:
        confidence += 0.05
    return min(confidence, 0.95)


def _plan(
    steps: list[MultiStepToolCall],
    message: str,
    confidence: float,
) -> MultiStepPlannerResult:
    return MultiStepPlannerResult(
        status="plan",
        plan=MultiStepPlan(
            source="deterministic",
            confidence=confidence,
            steps=steps,
            message=message,
        ),
        confidence=confidence,
    )


def _clarify(message: str) -> MultiStepPlannerResult:
    return MultiStepPlannerResult(status="clarify", message=message)


def _find_preferred_group_column(
    dataframe: pd.DataFrame, exclude: set[str | None] | None = None
) -> str | None:
    exclude_values = {value for value in (exclude or set()) if isinstance(value, str)}
    preferred_tokens = (
        "department",
        "region",
        "gender",
        "category",
        "segment",
        "group",
        "class",
        "team",
        "city",
        "country",
    )
    candidates = []
    for column in dataframe.columns:
        column_name = str(column)
        if column_name in exclude_values or _is_numeric_column(dataframe, column_name):
            continue
        non_null_count = int(dataframe[column_name].notna().sum())
        unique_count = int(dataframe[column_name].dropna().nunique())
        if non_null_count <= 0 or unique_count > 20:
            continue
        normalized_column = normalize_text(column_name.replace("_", " "))
        for index, token in enumerate(preferred_tokens):
            if token in normalized_column:
                candidates.append((index, column_name))
                break
    if not candidates:
        return None
    return sorted(candidates)[0][1]


def _has_any(normalized: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in normalized for phrase in phrases)


def _is_structured_schema_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "additionalproperties" in text or "response_schema" in text
