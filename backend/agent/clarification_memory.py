from collections.abc import Callable
from typing import Any

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.agent.column_resolver import (
    contains_normalized_column,
    normalize_text,
    resolve_column,
)
from backend.agent.correlation_helpers import find_mentioned_numeric_column
from backend.agent.helpers import (
    detect_aggregation,
    detect_chart_type,
    get_histogram_bins,
    has_group_intent,
)
from backend.agent.router import route_question
from backend.agent.response_composer import clarification_response
from backend.schemas import ChatResponse, ToolTraceItem
from backend.services.session_store import DatasetSession, session_store


ExecuteValidatedTool = Callable[
    [DatasetSession, str, str, dict[str, Any], list[ToolTraceItem], str],
    ChatResponse,
]


def try_resolve_pending_clarification(
    session: DatasetSession,
    question: str,
    traces: list[ToolTraceItem],
    execute_validated_tool: ExecuteValidatedTool,
) -> ChatResponse | None:
    pending = session.pending_clarification
    if pending is None:
        return None

    direct_route = route_question(session.dataframe, question)
    if direct_route.should_use_tool:
        session_store.clear_pending_clarification(session.session_id)
        return None

    if _is_new_standalone_intent(question, str(pending.get("intent"))):
        session_store.clear_pending_clarification(session.session_id)
        return None

    traces.append(
        ToolTraceItem(
            source="memory",
            tool_name=str(pending.get("intent")) if pending.get("intent") else None,
            arguments=pending,
            status="pending",
            message="Đang nối câu trả lời mới với yêu cầu cần làm rõ trước đó.",
        )
    )

    intent = pending.get("intent")
    if intent == "aggregate_metric":
        resolved = _resolve_pending_aggregate(session, question, pending)
    elif intent == "generate_chart_spec":
        resolved = _resolve_pending_chart(session, question, pending)
    elif intent == "correlation_analysis":
        resolved = _resolve_pending_correlation(session, question, pending)
    else:
        session_store.clear_pending_clarification(session.session_id)
        return None

    if resolved is None:
        response = clarification_response(
            session.session_id,
            "Mình vẫn chưa xác định đủ cột cần dùng. Bạn hãy nêu rõ metric và nhóm, ví dụ: salary và department.",
            traces,
            options=column_options(session),
        )
        session_store.set_pending_clarification(session.session_id, pending)
        return response

    traces.append(
        ToolTraceItem(
            source="memory",
            tool_name=resolved["tool_name"],
            arguments=resolved["arguments"],
            status="resolved",
            message="Đã điền đủ thông tin từ follow-up.",
        )
    )
    return execute_validated_tool(
        session,
        str(pending.get("original_question", question)),
        resolved["tool_name"],
        resolved["arguments"],
        traces,
        "memory",
    )


def set_pending_from_question(
    session: DatasetSession, question: str, message: str
) -> None:
    pending = _build_pending_from_question(session, question, message)
    if pending is not None:
        session_store.set_pending_clarification(session.session_id, pending)


def set_pending_from_tool_call(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    if tool_name == "aggregate_metric":
        pending = {
            "intent": "aggregate_metric",
            "operation": str(
                arguments.get("operation", detect_aggregation(question) or "mean")
            ),
            "metric_column": arguments.get("metric_column"),
            "group_by": arguments.get("group_by"),
            "original_question": question,
            "message": message,
        }
        session_store.set_pending_clarification(session.session_id, pending)
    elif tool_name == "correlation_analysis":
        session_store.set_pending_clarification(
            session.session_id,
            {
                "intent": "correlation_analysis",
                "target_column": None,
                "original_question": question,
                "message": message,
            },
        )


def column_options(
    session: DatasetSession, numeric_only: bool = False, limit: int = 12
) -> list[str]:
    columns: list[str] = []
    for column in session.dataframe.columns:
        column_name = str(column)
        if numeric_only and not _is_numeric_dataset_column(session, column_name):
            continue
        columns.append(column_name)
    return columns[:limit]


def _build_pending_from_question(
    session: DatasetSession, question: str, message: str
) -> dict[str, object] | None:
    normalized = normalize_text(question)
    operation = detect_aggregation(question)
    if operation is not None:
        metric_column, group_by = _infer_metric_and_group(session, question)
        if metric_column is not None and not has_group_intent(normalized):
            return None
        return {
            "intent": "aggregate_metric",
            "operation": operation,
            "metric_column": metric_column,
            "group_by": group_by,
            "original_question": question,
            "message": message,
        }

    if any(
        token in normalized
        for token in (
            "bieu do",
            "chart",
            "plot",
            "histogram",
            "phan phoi",
            "scatter",
            "heatmap",
        )
    ):
        metric_column, group_by = _infer_metric_and_group(session, question)
        return {
            "intent": "generate_chart_spec",
            "chart_type": detect_chart_type(question),
            "chart_type_explicit": _has_explicit_chart_type(question),
            "metric_column": metric_column,
            "group_by": group_by,
            "original_question": question,
            "message": message,
        }

    if any(token in normalized for token in ("tuong quan", "lien quan", "correlation")):
        return {
            "intent": "correlation_analysis",
            "target_column": find_mentioned_numeric_column(session, question),
            "original_question": question,
            "message": message,
        }
    return None


def _is_new_standalone_intent(question: str, pending_intent: str) -> bool:
    normalized = normalize_text(question)
    chart_tokens = (
        "phan phoi",
        "bieu do",
        "chart",
        "plot",
        "histogram",
        "scatter",
        "heatmap",
    )
    if pending_intent != "generate_chart_spec" and any(
        token in normalized for token in chart_tokens
    ):
        return True
    return False


def _resolve_pending_aggregate(
    session: DatasetSession,
    follow_up: str,
    pending: dict[str, object],
) -> dict[str, Any] | None:
    metric_column = pending.get("metric_column")
    group_by = pending.get("group_by")
    follow_metric, follow_group = _infer_metric_and_group(session, follow_up)

    if metric_column is None and follow_metric is not None:
        metric_column = follow_metric
    if group_by is None and follow_group is not None:
        group_by = follow_group

    if metric_column is None or group_by is None:
        columns = _mentioned_columns(session, follow_up)
        for column in columns:
            if metric_column is None and _is_numeric_dataset_column(session, column):
                metric_column = column
            elif group_by is None and column != metric_column:
                group_by = column

    pending["metric_column"] = metric_column
    pending["group_by"] = group_by
    if not isinstance(metric_column, str) or not isinstance(group_by, str):
        return None

    return {
        "tool_name": "aggregate_metric",
        "arguments": {
            "metric_column": metric_column,
            "group_by": group_by,
            "operation": str(pending.get("operation", "mean")),
        },
    }


def _resolve_pending_chart(
    session: DatasetSession,
    follow_up: str,
    pending: dict[str, object],
) -> dict[str, Any] | None:
    metric_column = pending.get("metric_column")
    group_by = pending.get("group_by")
    follow_metric, follow_group = _infer_metric_and_group(session, follow_up)
    mentioned_columns = _mentioned_columns(session, follow_up)
    if metric_column is None and follow_metric is not None:
        metric_column = follow_metric
    if group_by is None and follow_group is not None:
        group_by = follow_group

    chart_type = str(pending.get("chart_type", "bar"))
    chart_type_explicit = bool(pending.get("chart_type_explicit", False))
    if not chart_type_explicit:
        follow_chart_type = detect_chart_type(follow_up)
        if follow_chart_type != "bar" or _has_explicit_chart_type(follow_up):
            chart_type = follow_chart_type
            chart_type_explicit = _has_explicit_chart_type(follow_up)

    pending["metric_column"] = metric_column
    pending["group_by"] = group_by
    pending["chart_type"] = chart_type
    pending["chart_type_explicit"] = chart_type_explicit

    numeric_mentions = [
        column
        for column in mentioned_columns
        if _is_numeric_dataset_column(session, column)
    ]

    if chart_type == "scatter" and len(numeric_mentions) >= 2:
        return {
            "tool_name": "generate_chart_spec",
            "arguments": {
                "chart_type": "scatter",
                "x": numeric_mentions[0],
                "y": numeric_mentions[1],
            },
        }

    if (
        chart_type in {"histogram", "bar"}
        and not chart_type_explicit
        and len(numeric_mentions) >= 2
    ):
        return {
            "tool_name": "generate_chart_spec",
            "arguments": {
                "chart_type": "scatter",
                "x": numeric_mentions[0],
                "y": numeric_mentions[1],
            },
        }

    if chart_type == "pie" and isinstance(group_by, str):
        return {
            "tool_name": "generate_chart_spec",
            "arguments": {"chart_type": "pie", "names": group_by},
        }

    if (
        not chart_type_explicit
        and not isinstance(metric_column, str)
        and isinstance(group_by, str)
    ):
        return {
            "tool_name": "generate_chart_spec",
            "arguments": {"chart_type": "pie", "names": group_by},
        }

    if isinstance(metric_column, str) and (
        chart_type == "histogram" or (not chart_type_explicit and group_by is None)
    ):
        return {
            "tool_name": "generate_chart_spec",
            "arguments": {
                "chart_type": "histogram",
                "x": metric_column,
                "bins": get_histogram_bins(session.dataframe, metric_column),
            },
        }
    if not isinstance(metric_column, str) or not isinstance(group_by, str):
        return None
    return {
        "tool_name": "generate_chart_spec",
        "arguments": {"chart_type": chart_type, "x": group_by, "y": metric_column},
    }


def _resolve_pending_correlation(
    session: DatasetSession,
    follow_up: str,
    pending: dict[str, object],
) -> dict[str, Any] | None:
    target_column = pending.get("target_column")
    if target_column is None:
        target_column = find_mentioned_numeric_column(session, follow_up)
    pending["target_column"] = target_column
    if not isinstance(target_column, str):
        return None

    columns = [
        str(column)
        for column in session.dataframe.columns
        if _is_numeric_dataset_column(session, str(column))
    ]
    if target_column not in columns:
        columns.insert(0, target_column)
    return {"tool_name": "correlation_analysis", "arguments": {"columns": columns}}


def _infer_metric_and_group(
    session: DatasetSession, text: str
) -> tuple[str | None, str | None]:
    metric_column: str | None = None
    group_by: str | None = None
    for column in _mentioned_columns(session, text):
        if _is_numeric_dataset_column(session, column):
            if metric_column is None:
                metric_column = column
        elif group_by is None:
            group_by = column
    if metric_column is None:
        metric_column = resolve_column(session.dataframe, text, expected_type="numeric")
    if group_by is None:
        group_by = resolve_column(session.dataframe, text, expected_type="categorical")
    return metric_column, group_by


def _mentioned_columns(session: DatasetSession, text: str) -> list[str]:
    normalized = normalize_text(text)
    matches = []
    for column in session.dataframe.columns:
        column_name = str(column)
        normalized_column = normalize_text(column_name.replace("_", " "))
        if _contains_normalized_column(normalized, normalized_column):
            matches.append(column_name)
    if matches:
        return matches

    resolved = resolve_column(session.dataframe, text)
    return [resolved] if resolved else []


def _is_numeric_dataset_column(session: DatasetSession, column: str) -> bool:
    return is_numeric_dtype(session.dataframe[column]) and not is_bool_dtype(
        session.dataframe[column]
    )


def _has_explicit_chart_type(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        token in normalized
        for token in (
            "histogram",
            "phan phoi",
            "scatter",
            "phan tan",
            "line",
            "duong",
            "pie",
            "tron",
            "box",
            "boxplot",
            "bar",
        )
    )


def _contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    return contains_normalized_column(normalized_text, normalized_column)
