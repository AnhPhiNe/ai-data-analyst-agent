from typing import Any

from backend.agent.clarification_memory import (
    column_options,
    set_pending_from_question,
    set_pending_from_tool_call,
    try_resolve_pending_clarification,
)
from backend.agent.column_argument_repair import repair_tool_column_arguments
from backend.agent.correlation_helpers import correlation_target_issue, find_mentioned_numeric_column
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


def run_agent_turn(
    session: DatasetSession,
    question: str,
    provider: LLMProvider | None = None,
) -> ChatResponse:
    traces: list[ToolTraceItem] = []

    guardrail = check_guardrails(question)
    if not guardrail.is_allowed:
        response = ChatResponse(
            session_id=session.session_id,
            answer=guardrail.message,
            response_type="blocked",
            tool_trace=[
                ToolTraceItem(
                    source="guardrails",
                    status="blocked",
                    message=guardrail.message,
                )
            ],
            is_blocked=True,
        )
        _remember(session.session_id, question, response.answer, "guardrails")
        return response

    if session.pending_clarification is not None:
        pending_response = try_resolve_pending_clarification(
            session,
            question,
            traces,
            execute_validated_tool,
        )
        if pending_response is not None:
            _remember(session.session_id, question, pending_response.answer, "clarification_followup")
            return pending_response

    router_decision = route_question(session.dataframe, question)
    traces.append(
        ToolTraceItem(
            source="router",
            tool_name=router_decision.tool_name,
            arguments=router_decision.arguments,
            status=router_decision.route_type,
            message=router_decision.message or "Router decision completed.",
            confidence=router_decision.confidence,
        )
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
        )
        _remember(session.session_id, question, response.answer, "router_tool")
        return response

    if provider is None:
        response = ChatResponse(
            session_id=session.session_id,
            answer=(
                "Mình chưa đủ tự tin để chọn công cụ phân tích cho câu hỏi này. "
                "Bạn có thể hỏi rõ hơn bằng cách nêu tên cột/metric trong dataset, "
                "hoặc cấu hình GEMINI_API_KEY để bật lớp hiểu ngôn ngữ tự nhiên nâng cao."
            ),
            response_type="error",
            tool_trace=traces
            + [
                ToolTraceItem(
                    source="gemini",
                    status="skipped",
                    message="Gemini provider is not configured.",
                )
            ],
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
    traces.append(
        ToolTraceItem(
            source="gemini",
            tool_name=gemini_result.tool_name,
            arguments=gemini_result.arguments,
            status=gemini_result.status,
            message=gemini_result.message,
            confidence=gemini_result.confidence,
        )
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
) -> ChatResponse:
    arguments = _repair_tool_arguments(session, question, tool_name, arguments, traces)
    target_issue = correlation_target_issue(session, question, tool_name)
    if target_issue is not None:
        traces.append(
            ToolTraceItem(
                source="agent_validation",
                tool_name=tool_name,
                arguments=arguments,
                status="clarify",
                message=target_issue,
            )
        )
        return clarification_response(
            session.session_id,
            target_issue,
            traces,
            options=column_options(session, numeric_only=True),
        )

    validation = validate_tool_call(session.dataframe, tool_name, arguments)
    traces.append(
        ToolTraceItem(
            source="tool_validation",
            tool_name=tool_name,
            arguments=arguments,
            status="success" if validation.is_valid else "error",
            message=validation.message,
        )
    )
    if not validation.is_valid:
        response = clarification_response(
            session.session_id,
            validation_clarification_message(tool_name, validation.message),
            traces,
            options=_validation_options(session, validation.message),
        )
        set_pending_from_tool_call(session, question, tool_name, arguments, response.answer)
        return response

    tool_result = execute_tool(session.dataframe, tool_name, validation.normalized_arguments)
    traces.append(
        ToolTraceItem(
            source="tool_executor",
            tool_name=tool_name,
            arguments=validation.normalized_arguments,
            status=tool_result.status,
            message=tool_result.message,
        )
    )

    if tool_result.status == "error":
        session_store.clear_pending_clarification(session.session_id)
        return ChatResponse(
            session_id=session.session_id,
            answer=f"Mình chưa thể hoàn tất phân tích này: {tool_result.message}",
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


def _safe_profile_summary(session: DatasetSession) -> dict[str, Any]:
    cached_profile = session.profile_cache
    profile = cached_profile if cached_profile is not None else profile_dataset(session.dataframe)
    return {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "dtypes": profile["dtypes"],
        "missing_values": profile["missing_values"],
        "numeric_summary": profile["numeric_summary"],
    }


def _remember(session_id: str, question: str, answer: str, route: str) -> None:
    session_store.add_chat_turn(session_id=session_id, question=question, answer=answer, route=route)


def _repair_tool_arguments(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    traces: list[ToolTraceItem],
) -> dict[str, Any]:
    repaired_arguments = repair_tool_column_arguments(session.dataframe, tool_name, arguments)
    if repaired_arguments != arguments:
        traces.append(
            ToolTraceItem(
                source="agent_column_resolver",
                tool_name=tool_name,
                arguments=repaired_arguments,
                status="success",
                message="Đã chuẩn hóa tên cột trong tool arguments theo schema dataset.",
            )
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
    traces.append(
        ToolTraceItem(
            source="agent_repair",
            tool_name=tool_name,
            arguments=repaired_arguments,
            status="success",
            message=f"Đã thêm cột mục tiêu '{target_column}' vào correlation_analysis.",
        )
    )
    return repaired_arguments


def _validation_options(session: DatasetSession, validation_message: str) -> list[str]:
    return column_options(session, numeric_only="must be numeric" in validation_message)
