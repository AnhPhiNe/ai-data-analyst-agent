from collections.abc import Callable
from typing import Any

import pandas as pd

from backend.agent.clarification_memory import (
    column_options,
    set_pending_from_question,
    set_pending_from_tool_call,
    try_resolve_pending_clarification,
)
from backend.agent.column_argument_repair import repair_tool_column_arguments
from backend.agent.column_resolver import normalize_text, resolve_column
from backend.agent.correlation_helpers import (
    correlation_target_issue,
    find_mentioned_numeric_column,
)
from backend.agent.gemini_runtime import LLMProvider, choose_tool_with_gemini
from backend.agent.guardrails import check_guardrails
from backend.agent.response_composer import (
    build_answer,
    clarification_response,
    response_type,
    validation_clarification_message,
)
from backend.agent.router import route_question
from backend.agent.tool_validation import validate_tool_call
from backend.schemas import ChatResponse, ToolTraceItem
from backend.services.profiling import profile_dataset
from backend.services.session_store import DatasetSession, session_store
from backend.tools.safe_pandas import execute_tool

TraceCallback = Callable[[ToolTraceItem], None]
MAX_GEMINI_VALIDATION_REPAIR_ATTEMPTS = 1


def run_agent_turn(
    session: DatasetSession,
    question: str,
    provider: LLMProvider | None = None,
    event_callback: TraceCallback | None = None,
) -> ChatResponse:
    traces: list[ToolTraceItem] = []

    guardrail = check_guardrails(
        question, column_names=[str(column) for column in session.dataframe.columns]
    )
    if not guardrail.is_allowed:
        guardrail_trace = ToolTraceItem(
            source="guardrails",
            status="blocked",
            message=guardrail.message,
        )
        _record_trace(traces, guardrail_trace, event_callback)
        response = ChatResponse(
            session_id=session.session_id,
            answer=guardrail.message,
            response_type="blocked",
            tool_trace=traces,
            is_blocked=True,
        )
        _remember(session.session_id, question, response.answer, "guardrails")
        return response

    def execute_tool_with_events(
        session: DatasetSession,
        question: str,
        tool_name: str,
        arguments: dict[str, Any],
        traces: list[ToolTraceItem],
        source: str,
    ) -> ChatResponse:
        return execute_validated_tool(
            session=session,
            question=question,
            tool_name=tool_name,
            arguments=arguments,
            traces=traces,
            source=source,
            event_callback=event_callback,
        )

    if session.pending_clarification is not None:
        pending_response = try_resolve_pending_clarification(
            session,
            question,
            traces,
            execute_tool_with_events,
        )
        if pending_response is not None:
            _remember(
                session.session_id,
                question,
                pending_response.answer,
                "clarification_followup",
            )
            return pending_response

    router_decision = route_question(session.dataframe, question)
    _record_trace(
        traces,
        ToolTraceItem(
            source="router",
            tool_name=router_decision.tool_name,
            arguments=router_decision.arguments,
            status=router_decision.route_type,
            message=router_decision.message or "Router decision completed.",
            confidence=router_decision.confidence,
        ),
        event_callback,
    )

    if router_decision.route_type == "clarify":
        response = clarification_response(
            session.session_id,
            router_decision.message or "Bạn có thể nói rõ hơn không?",
            traces,
            options=column_options(session),
        )
        set_pending_from_question(session, question, response.answer)
        _remember(session.session_id, question, response.answer, "router_clarify")
        return response

    if router_decision.should_use_tool and router_decision.tool_name:
        response = execute_validated_tool(
            session=session,
            question=question,
            tool_name=router_decision.tool_name,
            arguments=router_decision.arguments,
            traces=traces,
            source="router",
            event_callback=event_callback,
        )
        _remember(session.session_id, question, response.answer, "router_tool")
        return response

    if provider is None:
        skipped_trace = ToolTraceItem(
            source="gemini",
            status="skipped",
            message="Gemini provider is not configured.",
        )
        _record_trace(traces, skipped_trace, event_callback)
        response = ChatResponse(
            session_id=session.session_id,
            answer=(
                "Mình chưa đủ tự tin để chọn công cụ phân tích cho câu hỏi này. "
                "Bạn có thể hỏi rõ hơn bằng cách nêu tên cột/metric trong dataset, "
                "hoặc cấu hình GEMINI_API_KEY để bật lớp hiểu ngôn ngữ tự nhiên nâng cao."
            ),
            response_type="error",
            tool_trace=traces,
            clarification_options=column_options(session),
        )
        _remember(session.session_id, question, response.answer, "missing_gemini")
        return response

    gemini_result = choose_tool_with_gemini(
        dataframe=session.dataframe,
        question=question,
        provider=provider,
        profile_summary=_safe_profile_summary(session),
    )
    _record_trace(
        traces,
        ToolTraceItem(
            source="gemini",
            tool_name=gemini_result.tool_name,
            arguments=gemini_result.arguments,
            status=gemini_result.status,
            message=gemini_result.message,
            confidence=gemini_result.confidence,
        ),
        event_callback,
    )

    if gemini_result.status == "clarify":
        response = clarification_response(
            session.session_id,
            gemini_result.message,
            traces,
            options=column_options(session),
        )
        set_pending_from_question(session, question, response.answer)
        _remember(session.session_id, question, response.answer, "gemini_clarify")
        return response

    if gemini_result.status == "answer":
        response = ChatResponse(
            session_id=session.session_id,
            answer=gemini_result.message,
            response_type="answer",
            tool_trace=traces,
        )
        _remember(session.session_id, question, response.answer, "gemini_answer")
        return response

    if gemini_result.status == "error" or not gemini_result.tool_name:
        response = ChatResponse(
            session_id=session.session_id,
            answer=gemini_result.message,
            response_type="error",
            tool_trace=traces,
        )
        _remember(session.session_id, question, response.answer, "gemini_error")
        return response

    response = execute_validated_tool(
        session=session,
        question=question,
        tool_name=gemini_result.tool_name,
        arguments=gemini_result.arguments or {},
        traces=traces,
        source="gemini",
        event_callback=event_callback,
    )
    _remember(session.session_id, question, response.answer, "gemini_tool")
    return response


def execute_validated_tool(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    traces: list[ToolTraceItem],
    source: str,
    event_callback: TraceCallback | None = None,
) -> ChatResponse:
    arguments = _repair_tool_arguments(
        session, question, tool_name, arguments, traces, event_callback
    )
    target_issue = correlation_target_issue(session, question, tool_name)
    if target_issue is not None:
        _record_trace(
            traces,
            ToolTraceItem(
                source="agent_validation",
                tool_name=tool_name,
                arguments=arguments,
                status="clarify",
                message=target_issue,
            ),
            event_callback,
        )
        return clarification_response(
            session.session_id,
            target_issue,
            traces,
            options=column_options(session, numeric_only=True),
        )

    validation = validate_tool_call(session.dataframe, tool_name, arguments)
    _record_trace(
        traces,
        ToolTraceItem(
            source="tool_validation",
            tool_name=tool_name,
            arguments=arguments,
            status="success" if validation.is_valid else "error",
            message=validation.message,
        ),
        event_callback,
    )
    if (
        not validation.is_valid
        and source == "gemini"
        and MAX_GEMINI_VALIDATION_REPAIR_ATTEMPTS > 0
    ):
        repaired_arguments = _repair_after_validation_failure(
            session.dataframe, question, tool_name, arguments, validation.message
        )
        if repaired_arguments != arguments:
            _record_trace(
                traces,
                ToolTraceItem(
                    source="agent_repair",
                    tool_name=tool_name,
                    arguments=repaired_arguments,
                    status="success",
                    message=(
                        "Retried Gemini-selected tool call once after fixing "
                        "a validation issue."
                    ),
                ),
                event_callback,
            )
            arguments = repaired_arguments
            validation = validate_tool_call(session.dataframe, tool_name, arguments)
            _record_trace(
                traces,
                ToolTraceItem(
                    source="tool_validation",
                    tool_name=tool_name,
                    arguments=arguments,
                    status="success" if validation.is_valid else "error",
                    message=validation.message,
                ),
                event_callback,
            )
    if not validation.is_valid:
        response = clarification_response(
            session.session_id,
            validation_clarification_message(tool_name, validation.message),
            traces,
            options=_validation_options(session, validation.message),
        )
        set_pending_from_tool_call(
            session, question, tool_name, arguments, response.answer
        )
        return response

    tool_result = execute_tool(
        session.dataframe, tool_name, validation.normalized_arguments
    )
    _record_trace(
        traces,
        ToolTraceItem(
            source="tool_executor",
            tool_name=tool_name,
            arguments=validation.normalized_arguments,
            status=tool_result.status,
            message=tool_result.message,
        ),
        event_callback,
    )
    if tool_result.status == "error":
        session_store.clear_pending_clarification(session.session_id)
        return ChatResponse(
            session_id=session.session_id,
            answer=f"Could not complete this analysis: {tool_result.message}",
            response_type="error",
            tool_trace=traces,
        )

    answer = build_answer(question, tool_result)
    session_store.clear_pending_clarification(session.session_id)
    return ChatResponse(
        session_id=session.session_id,
        answer=answer,
        response_type=response_type(tool_result),
        table=tool_result.table,
        chart_spec=tool_result.chart_spec,
        tool_trace=traces,
    )


def _repair_after_validation_failure(
    dataframe: pd.DataFrame,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    validation_message: str,
) -> dict[str, Any]:
    if "not allowed" in validation_message or not isinstance(arguments, dict):
        return arguments

    repaired = dict(arguments)
    _coerce_integer_argument(repaired, "top_n")
    _coerce_integer_argument(repaired, "limit")
    _coerce_integer_argument(repaired, "bins")
    _normalize_operation_argument(repaired)
    _normalize_operator_argument(repaired)

    if "does not exist" in validation_message:
        _repair_missing_column_from_question(dataframe, question, tool_name, repaired)

    return repaired


def _coerce_integer_argument(arguments: dict[str, Any], key: str) -> None:
    value = arguments.get(key)
    if isinstance(value, str) and value.strip().isdigit():
        arguments[key] = int(value.strip())


def _normalize_operation_argument(arguments: dict[str, Any]) -> None:
    value = arguments.get("operation")
    if not isinstance(value, str):
        return
    operation_aliases = {
        "avg": "mean",
        "average": "mean",
        "trung binh": "mean",
        "mean": "mean",
        "tong": "sum",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "median": "median",
        "count": "count",
    }
    normalized = normalize_text(value)
    if normalized in operation_aliases:
        arguments["operation"] = operation_aliases[normalized]


def _normalize_operator_argument(arguments: dict[str, Any]) -> None:
    value = arguments.get("operator")
    if not isinstance(value, str):
        return
    operator_aliases = {
        ">": "gt",
        ">=": "gte",
        "<": "lt",
        "<=": "lte",
        "=": "eq",
        "==": "eq",
        "gt": "gt",
        "greater than": "gt",
        "lon hon": "gt",
        "gte": "gte",
        "greater or equal": "gte",
        "lt": "lt",
        "less than": "lt",
        "duoi": "lt",
        "nho hon": "lt",
        "lte": "lte",
        "less or equal": "lte",
        "eq": "eq",
        "bang": "eq",
        "ne": "ne",
        "khac": "ne",
        "contains": "contains",
        "chua": "contains",
    }
    normalized = normalize_text(value)
    if value in operator_aliases:
        arguments["operator"] = operator_aliases[value]
    elif normalized in operator_aliases:
        arguments["operator"] = operator_aliases[normalized]


def _repair_missing_column_from_question(
    dataframe: pd.DataFrame,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    column_expectations = {
        "describe_numeric": {"column": "numeric"},
        "value_counts": {"column": None},
        "outlier_detection": {"column": "numeric"},
        "aggregate_metric": {
            "metric_column": "numeric",
            "group_by": "categorical",
        },
        "compare_groups": {
            "metric_column": "numeric",
            "group_by": "categorical",
        },
        "sort_values": {"column": None},
        "filter_rows": {"column": None},
        "conditional_percentage": {"column": None},
        "generate_chart_spec": {"x": None, "y": "numeric", "names": None},
    }
    for key, expected_type in column_expectations.get(tool_name, {}).items():
        value = arguments.get(key)
        if not isinstance(value, str) or value in dataframe.columns:
            continue
        resolved = resolve_column(dataframe, question, expected_type=expected_type)
        if resolved is not None:
            arguments[key] = resolved


def _safe_profile_summary(session: DatasetSession) -> dict[str, Any]:
    cached_profile = session.profile_cache
    profile = (
        cached_profile
        if cached_profile is not None
        else profile_dataset(session.dataframe)
    )
    return {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "dtypes": profile["dtypes"],
        "column_metadata": profile["column_metadata"],
        "missing_values": profile["missing_values"],
        "numeric_summary": profile["numeric_summary"],
    }


def _remember(session_id: str, question: str, answer: str, route: str) -> None:
    session_store.add_chat_turn(
        session_id=session_id, question=question, answer=answer, route=route
    )


def _repair_tool_arguments(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    traces: list[ToolTraceItem],
    event_callback: TraceCallback | None = None,
) -> dict[str, Any]:
    repaired_arguments = repair_tool_column_arguments(
        session.dataframe, tool_name, arguments
    )
    if repaired_arguments != arguments:
        _record_trace(
            traces,
            ToolTraceItem(
                source="agent_column_resolver",
                tool_name=tool_name,
                arguments=repaired_arguments,
                status="success",
                message="Đã chuẩn hóa tên cột trong tool arguments theo schema dataset.",
            ),
            event_callback,
        )
        arguments = repaired_arguments

    if tool_name != "correlation_analysis":
        return arguments

    target_column = find_mentioned_numeric_column(session, question)
    if target_column is None:
        return arguments

    columns = arguments.get("columns")
    if columns is None:
        return arguments
    if not isinstance(columns, list) or target_column in columns:
        return arguments

    repaired_arguments = {**arguments, "columns": [target_column, *columns]}
    _record_trace(
        traces,
        ToolTraceItem(
            source="agent_repair",
            tool_name=tool_name,
            arguments=repaired_arguments,
            status="success",
            message=f"Đã thêm cột mục tiêu '{target_column}' vào correlation_analysis.",
        ),
        event_callback,
    )
    return repaired_arguments


def _validation_options(session: DatasetSession, validation_message: str) -> list[str]:
    return column_options(session, numeric_only="must be numeric" in validation_message)


def _record_trace(
    traces: list[ToolTraceItem],
    trace: ToolTraceItem,
    event_callback: TraceCallback | None = None,
) -> None:
    traces.append(trace)
    if event_callback is not None:
        event_callback(trace)
