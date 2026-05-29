from collections.abc import Callable
from typing import Any

import pandas as pd

from backend.agent.clarification_memory import (
    try_resolve_pending_clarification,
    set_pending_from_question,
    set_pending_from_tool_call,
)
from backend.agent.column_argument_repair import repair_tool_column_arguments
from backend.agent.column_resolver import normalize_text, resolve_column
from backend.agent.correlation_helpers import (
    correlation_target_issue,
    find_mentioned_numeric_column,
)
from backend.agent.gemini_runtime import LLMProvider, choose_tool_with_gemini
from backend.agent.guardrails import check_guardrails
from backend.agent.multi_step_planner import (
    MultiStepPlan,
    plan_multi_step_question,
    validate_multi_step_plan,
)
from backend.agent.response_composer import (
    build_answer,
    build_multi_step_answer,
    clarification_response,
    response_type,
    validation_clarification_message,
)
from backend.agent.router import route_question
from backend.agent.tool_validation import validate_tool_call
from backend.schemas import ChatResponse, ToolTraceItem
from backend.services.profiling import profile_dataset
from backend.services.session_store import DatasetSession, session_store
from backend.tools.safe_pandas import ToolResult, execute_tool

TraceCallback = Callable[[ToolTraceItem], None]
MAX_GEMINI_VALIDATION_REPAIR_ATTEMPTS = 1


def run_agent_turn(
    session: DatasetSession,
    question: str,
    provider: LLMProvider | None = None,
    event_callback: TraceCallback | None = None,
    max_planner_validation_retries: int = 1,
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

    multi_step_result = plan_multi_step_question(
        dataframe=session.dataframe,
        question=question,
        provider=provider,
        profile_summary=_safe_profile_summary(session),
    )
    if multi_step_result.status != "skip":
        _record_trace(
            traces,
            ToolTraceItem(
                source="multi_step_planner",
                status=multi_step_result.status,
                message=multi_step_result.message
                or (
                    multi_step_result.plan.message
                    if multi_step_result.plan is not None
                    else "Multi-step planner evaluated the question."
                ),
                confidence=multi_step_result.confidence,
                arguments=(
                    {
                        "steps": [
                            {
                                "tool_name": step.tool_name,
                                "arguments": step.arguments,
                                "purpose": step.purpose,
                            }
                            for step in multi_step_result.plan.steps
                        ]
                    }
                    if multi_step_result.plan is not None
                    else None
                ),
            ),
            event_callback,
        )
        if multi_step_result.status == "clarify":
            session_store.clear_pending_clarification(session.session_id)
            set_pending_from_question(
                session=session,
                question=question,
                message=multi_step_result.message or "Bạn hãy nói rõ hơn yêu cầu phân tích.",
                initial_arguments=None,
            )
            response = clarification_response(
                session.session_id,
                multi_step_result.message or "Bạn hãy nói rõ hơn yêu cầu phân tích.",
                traces,
            )
            _remember(
                session.session_id, question, response.answer, "multi_step_clarify"
            )
            return response
        if multi_step_result.status == "answer":
            response = ChatResponse(
                session_id=session.session_id,
                answer=multi_step_result.message or "Câu hỏi này không cần chạy tool.",
                response_type="answer",
                tool_trace=traces,
            )
            _remember(
                session.session_id, question, response.answer, "multi_step_answer"
            )
            return response
        if multi_step_result.status == "error":
            response = ChatResponse(
                session_id=session.session_id,
                answer=multi_step_result.message or "Không thể lập multi-step plan.",
                response_type="error",
                tool_trace=traces,
            )
            _remember(session.session_id, question, response.answer, "multi_step_error")
            return response
        if multi_step_result.plan is not None:
            response = execute_multi_step_plan(
                session=session,
                question=question,
                plan=multi_step_result.plan,
                traces=traces,
                event_callback=event_callback,
            )
            _remember(session.session_id, question, response.answer, "multi_step_tool")
            return response

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
        session_store.clear_pending_clarification(session.session_id)
        set_pending_from_question(
            session=session,
            question=question,
            message=router_decision.message or "Bạn có thể nói rõ hơn không?",
            initial_arguments=router_decision.arguments,
        )
        response = clarification_response(
            session.session_id,
            router_decision.message or "Bạn có thể nói rõ hơn không?",
            traces,
        )
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
            source="llm",
            status="skipped",
            message="LLM provider is not configured.",
        )
        _record_trace(traces, skipped_trace, event_callback)
        response = ChatResponse(
            session_id=session.session_id,
            answer=(
                "Mình chưa đủ tự tin để chọn công cụ phân tích cho câu hỏi này. "
                "Bạn có thể hỏi rõ hơn bằng cách nêu tên cột/metric trong dataset, "
                "hoặc cấu hình GEMINI_API_KEY/GROQ_API_KEY để bật lớp hiểu ngôn ngữ tự nhiên nâng cao."
            ),
            response_type="error",
            tool_trace=traces,
        )
        _remember(session.session_id, question, response.answer, "missing_llm")
        return response

    gemini_result = choose_tool_with_gemini(
        dataframe=session.dataframe,
        question=question,
        provider=provider,
        profile_summary=_safe_profile_summary(session),
        max_validation_retries=max_planner_validation_retries,
    )
    gemini_message = gemini_result.message
    if gemini_result.validation_retry_count:
        gemini_message = (
            f"{gemini_message} | planner_validation_retries="
            f"{gemini_result.validation_retry_count}"
        )
    _record_trace(
        traces,
        ToolTraceItem(
            source="llm",
            tool_name=gemini_result.tool_name,
            arguments=gemini_result.arguments,
            status=gemini_result.status,
            message=gemini_message,
            confidence=gemini_result.confidence,
        ),
        event_callback,
    )

    if gemini_result.status == "clarify":
        session_store.clear_pending_clarification(session.session_id)
        set_pending_from_question(
            session=session,
            question=question,
            message=gemini_result.message,
            initial_arguments=gemini_result.arguments or {},
        )
        response = clarification_response(
            session.session_id,
            gemini_result.message,
            traces,
        )
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
        source="llm",
        event_callback=event_callback,
    )
    _remember(session.session_id, question, response.answer, "gemini_tool")
    return response


def execute_multi_step_plan(
    session: DatasetSession,
    question: str,
    plan: MultiStepPlan,
    traces: list[ToolTraceItem],
    event_callback: TraceCallback | None = None,
) -> ChatResponse:
    is_valid_plan, plan_message = validate_multi_step_plan(session.dataframe, plan)
    _record_trace(
        traces,
        ToolTraceItem(
            source="multi_step_validation",
            status="success" if is_valid_plan else "error",
            message=plan_message,
            confidence=plan.confidence,
        ),
        event_callback,
    )
    if not is_valid_plan:
        session_store.clear_pending_clarification(session.session_id)
        return clarification_response(
            session.session_id,
            f"Multi-step plan chưa hợp lệ: {plan_message}",
            traces,
        )

    tool_results: list[ToolResult] = []
    warnings: list[str] = []
    for index, step in enumerate(plan.steps, start=1):
        arguments = _repair_tool_arguments(
            session=session,
            question=question,
            tool_name=step.tool_name,
            arguments=step.arguments,
            traces=traces,
            event_callback=event_callback,
        )
        validation = validate_tool_call(session.dataframe, step.tool_name, arguments)
        _record_trace(
            traces,
            ToolTraceItem(
                source="multi_step_validation",
                tool_name=step.tool_name,
                arguments=arguments,
                status="success" if validation.is_valid else "error",
                message=f"Step {index}: {validation.message}",
            ),
            event_callback,
        )
        if not validation.is_valid:
            session_store.clear_pending_clarification(session.session_id)
            set_pending_from_tool_call(
                session=session,
                question=question,
                tool_name=step.tool_name,
                arguments=arguments,
                message=validation_clarification_message(
                    step.tool_name, validation.message
                ),
            )
            if not tool_results:
                return clarification_response(
                    session.session_id,
                    validation_clarification_message(
                        step.tool_name, validation.message
                    ),
                    traces,
                )
            warnings.append(f"Bỏ qua bước {index} vì tham số chưa hợp lệ.")
            break

        tool_result = execute_tool(
            session.dataframe, step.tool_name, validation.normalized_arguments
        )
        _record_trace(
            traces,
            ToolTraceItem(
                source="multi_step_executor",
                tool_name=step.tool_name,
                arguments=validation.normalized_arguments,
                status=tool_result.status,
                message=f"Step {index}: {tool_result.message}",
            ),
            event_callback,
        )
        if tool_result.status == "error":
            session_store.clear_pending_clarification(session.session_id)
            if not tool_results:
                return ChatResponse(
                    session_id=session.session_id,
                    answer=f"Could not complete this multi-step analysis: {tool_result.message}",
                    response_type="error",
                    tool_trace=traces,
                )
            warnings.append(f"Bỏ qua bước {index}: {tool_result.message}")
            break
        tool_results.append(tool_result)

    answer = build_multi_step_answer(question, tool_results, warnings)
    table = _select_multi_step_table(tool_results)
    chart_spec = _select_multi_step_chart(tool_results)
    session_store.clear_pending_clarification(session.session_id)
    return ChatResponse(
        session_id=session.session_id,
        answer=answer,
        response_type="chart" if chart_spec else "table" if table else "answer",
        table=table,
        chart_spec=chart_spec,
        tool_trace=traces,
    )


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
        set_pending_from_tool_call(
            session=session,
            question=question,
            tool_name=tool_name,
            arguments=arguments,
            message=target_issue,
        )
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
        and source == "llm"
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
                        "Retried LLM-selected tool call once after fixing "
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
        session_store.clear_pending_clarification(session.session_id)
        set_pending_from_tool_call(
            session=session,
            question=question,
            tool_name=tool_name,
            arguments=arguments,
            message=validation_clarification_message(tool_name, validation.message),
        )
        response = clarification_response(
            session.session_id,
            validation_clarification_message(tool_name, validation.message),
            traces,
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


def _select_multi_step_table(
    tool_results: list[ToolResult],
) -> list[dict[str, Any]] | None:
    priority = (
        "query_table_sql",
        "compare_groups",
        "correlation_analysis",
        "data_quality_report",
        "outlier_detection",
        "detect_missing_values",
        "describe_numeric",
        "value_counts",
    )
    for tool_name in priority:
        for result in tool_results:
            if result.tool_name == tool_name and result.table is not None:
                return result.table
    for result in tool_results:
        if result.table is not None:
            return result.table
    return None


def _select_multi_step_chart(tool_results: list[ToolResult]) -> dict[str, Any] | None:
    for result in reversed(tool_results):
        if result.chart_spec is not None:
            return result.chart_spec
    return None


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
        "equals": "eq",
        "equal": "eq",
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


def _record_trace(
    traces: list[ToolTraceItem],
    trace: ToolTraceItem,
    event_callback: TraceCallback | None = None,
) -> None:
    traces.append(trace)
    if event_callback is not None:
        event_callback(trace)
