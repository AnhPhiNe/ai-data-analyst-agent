from typing import Any
import difflib
import math
import re
import unicodedata

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.agent.gemini_runtime import LLMProvider, choose_tool_with_gemini
from backend.agent.guardrails import check_guardrails
from backend.agent.router import route_question
from backend.agent.tool_validation import validate_tool_call
from backend.schemas import ChatResponse, ToolTraceItem
from backend.services.profiling import profile_dataset
from backend.services.session_store import DatasetSession, session_store
from backend.tools.safe_pandas import ToolResult, execute_tool


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
        pending_response = _try_resolve_pending_clarification(session, question, traces)
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
        response = _clarification_response(session.session_id, router_decision.message or "Ban co the noi ro hon khong?", traces)
        _set_pending_from_question(session, question, response.answer)
        _remember(session.session_id, question, response.answer, "router_clarify")
        return response

    if router_decision.should_use_tool and router_decision.tool_name:
        response = _execute_validated_tool(
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
                "Cau hoi nay chua khop rule chac chan. Hay cau hinh GEMINI_API_KEY de agent hieu ngon ngu tu nhien hon, "
                "hoac hoi ro hon ve cot/metric trong dataset."
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
        response = _clarification_response(session.session_id, gemini_result.message, traces)
        _set_pending_from_question(session, question, response.answer)
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

    response = _execute_validated_tool(
        session=session,
        question=question,
        tool_name=gemini_result.tool_name,
        arguments=gemini_result.arguments or {},
        traces=traces,
        source="gemini",
    )
    _remember(session.session_id, question, response.answer, "gemini_tool")
    return response


def _execute_validated_tool(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    traces: list[ToolTraceItem],
    source: str,
) -> ChatResponse:
    arguments = _repair_tool_arguments(session, question, tool_name, arguments, traces)
    target_issue = _correlation_target_issue(session, question, tool_name)
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
        return _clarification_response(session.session_id, target_issue, traces)

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
        response = _clarification_response(session.session_id, f"Tool call chua hop le: {validation.message}", traces)
        _set_pending_from_tool_call(session, question, tool_name, arguments, response.answer)
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
            answer=f"Khong the thuc thi tool '{tool_name}': {tool_result.message}",
            response_type="error",
            tool_trace=traces,
        )

    answer = _generate_answer(question, tool_result)
    session_store.clear_pending_clarification(session.session_id)
    return ChatResponse(
        session_id=session.session_id,
        answer=answer,
        response_type=_response_type(tool_result),
        table=tool_result.table,
        chart_spec=tool_result.chart_spec,
        tool_trace=traces,
    )


def _generate_answer(question: str, tool_result: ToolResult) -> str:
    if tool_result.tool_name == "profile_dataset" and isinstance(tool_result.data, dict):
        rows = tool_result.data.get("rows")
        columns = tool_result.data.get("columns")
        return f"Dữ liệu hiện có {_format_number(rows)} dòng và {_format_number(columns)} cột."

    if tool_result.tool_name == "list_columns" and isinstance(tool_result.data, dict):
        columns = tool_result.data.get("columns", [])
        return "Dataset có các cột: " + ", ".join(str(column) for column in columns) + "."

    if tool_result.tool_name == "detect_missing_values":
        missing_rows = [row for row in tool_result.table or [] if row.get("missing_count", 0) > 0]
        if not missing_rows:
            return "Dữ liệu không có giá trị thiếu trong các cột đã kiểm tra."
        details = ", ".join(f"{row['column']}: {row['missing_count']}" for row in missing_rows)
        return f"Dữ liệu có giá trị thiếu ở các cột sau: {details}."

    if tool_result.tool_name == "correlation_analysis":
        answer = _correlation_answer(question, tool_result.table or [])
        if answer:
            return answer

    if tool_result.tool_name == "generate_chart_spec":
        chart_type = (tool_result.chart_spec or {}).get("chart_type", "chart")
        return f"Đã tạo chart spec dạng {chart_type}. Frontend có thể render trực tiếp spec này."

    if tool_result.table:
        return f"Đã tính xong bảng kết quả bằng tool '{tool_result.tool_name}'."

    return tool_result.message


def _response_type(tool_result: ToolResult) -> str:
    if tool_result.chart_spec is not None:
        return "chart"
    if tool_result.table is not None:
        return "table"
    return "answer"


def _clarification_response(session_id: str, message: str, traces: list[ToolTraceItem]) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=message,
        response_type="clarification",
        tool_trace=traces,
        should_clarify=True,
    )


def _safe_profile_summary(session: DatasetSession) -> dict[str, Any]:
    profile = profile_dataset(session.dataframe)
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


def _try_resolve_pending_clarification(
    session: DatasetSession,
    question: str,
    traces: list[ToolTraceItem],
) -> ChatResponse | None:
    pending = session.pending_clarification
    if pending is None:
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
        response = _clarification_response(
            session.session_id,
            "Mình vẫn chưa xác định đủ cột cần dùng. Bạn hãy nêu rõ metric và nhóm, ví dụ: salary và department.",
            traces,
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
    return _execute_validated_tool(
        session=session,
        question=str(pending.get("original_question", question)),
        tool_name=resolved["tool_name"],
        arguments=resolved["arguments"],
        traces=traces,
        source="memory",
    )


def _set_pending_from_question(session: DatasetSession, question: str, message: str) -> None:
    pending = _build_pending_from_question(session, question, message)
    if pending is not None:
        session_store.set_pending_clarification(session.session_id, pending)


def _set_pending_from_tool_call(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    message: str,
) -> None:
    if tool_name == "aggregate_metric":
        pending = {
            "intent": "aggregate_metric",
            "operation": str(arguments.get("operation", _detect_aggregation_operation(question) or "mean")),
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


def _build_pending_from_question(session: DatasetSession, question: str, message: str) -> dict[str, object] | None:
    normalized = _normalize_text(question)
    operation = _detect_aggregation_operation(question)
    if operation is not None:
        metric_column, group_by = _infer_metric_and_group(session, question)
        return {
            "intent": "aggregate_metric",
            "operation": operation,
            "metric_column": metric_column,
            "group_by": group_by,
            "original_question": question,
            "message": message,
        }

    if any(token in normalized for token in ("bieu do", "chart", "plot", "histogram", "scatter", "heatmap")):
        metric_column, group_by = _infer_metric_and_group(session, question)
        return {
            "intent": "generate_chart_spec",
            "chart_type": _detect_pending_chart_type(question),
            "metric_column": metric_column,
            "group_by": group_by,
            "original_question": question,
            "message": message,
        }

    if any(token in normalized for token in ("tuong quan", "lien quan", "correlation")):
        return {
            "intent": "correlation_analysis",
            "target_column": _find_mentioned_numeric_column(session, question),
            "original_question": question,
            "message": message,
        }
    return None


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
    if metric_column is None and follow_metric is not None:
        metric_column = follow_metric
    if group_by is None and follow_group is not None:
        group_by = follow_group

    chart_type = str(pending.get("chart_type", "bar"))
    pending["metric_column"] = metric_column
    pending["group_by"] = group_by
    if chart_type == "histogram" and isinstance(metric_column, str):
        return {"tool_name": "generate_chart_spec", "arguments": {"chart_type": "histogram", "x": metric_column}}
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
        target_column = _find_mentioned_numeric_column(session, follow_up)
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


def _infer_metric_and_group(session: DatasetSession, text: str) -> tuple[str | None, str | None]:
    metric_column: str | None = None
    group_by: str | None = None
    for column in _mentioned_columns(session, text):
        if _is_numeric_dataset_column(session, column):
            if metric_column is None:
                metric_column = column
        elif group_by is None:
            group_by = column
    return metric_column, group_by


def _mentioned_columns(session: DatasetSession, text: str) -> list[str]:
    normalized = _normalize_text(text)
    matches = []
    for column in session.dataframe.columns:
        column_name = str(column)
        normalized_column = _normalize_text(column_name.replace("_", " "))
        if _contains_normalized_column(normalized, normalized_column):
            matches.append(column_name)
    if matches:
        return matches

    close_match = _find_close_column_name(text, [str(column) for column in session.dataframe.columns])
    return [close_match] if close_match else []


def _is_numeric_dataset_column(session: DatasetSession, column: str) -> bool:
    return is_numeric_dtype(session.dataframe[column]) and not is_bool_dtype(session.dataframe[column])


def _detect_aggregation_operation(text: str) -> str | None:
    normalized = _normalize_text(text)
    if any(token in normalized for token in ("trung binh", "average", "mean", "avg")):
        return "mean"
    if any(token in normalized for token in ("tong", "sum", "total")):
        return "sum"
    if "median" in normalized or "trung vi" in normalized:
        return "median"
    if "min" in normalized or "nho nhat" in normalized:
        return "min"
    if "max" in normalized or "lon nhat" in normalized or "cao nhat" in normalized:
        return "max"
    return None


def _detect_pending_chart_type(text: str) -> str:
    normalized = _normalize_text(text)
    if "histogram" in normalized or "phan phoi" in normalized:
        return "histogram"
    if "scatter" in normalized or "phan tan" in normalized:
        return "scatter"
    if "line" in normalized or "duong" in normalized:
        return "line"
    if "pie" in normalized or "tron" in normalized:
        return "pie"
    if "box" in normalized:
        return "box"
    return "bar"


def _repair_tool_arguments(
    session: DatasetSession,
    question: str,
    tool_name: str,
    arguments: dict[str, Any],
    traces: list[ToolTraceItem],
) -> dict[str, Any]:
    if tool_name != "correlation_analysis":
        return arguments

    target_column = _find_mentioned_numeric_column(session, question)
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


def _find_mentioned_numeric_column(session: DatasetSession, question: str) -> str | None:
    normalized_question = _normalize_text(question)
    for column in session.dataframe.columns:
        column_name = str(column)
        if not is_numeric_dtype(session.dataframe[column_name]) or is_bool_dtype(session.dataframe[column_name]):
            continue
        normalized_column = _normalize_text(column_name.replace("_", " "))
        if _contains_normalized_column(normalized_question, normalized_column):
            return column_name

    target_phrase = _extract_correlation_target_phrase(question)
    if target_phrase is not None:
        return _find_close_numeric_column(session, target_phrase)
    return None


def _correlation_target_issue(session: DatasetSession, question: str, tool_name: str) -> str | None:
    if tool_name != "correlation_analysis":
        return None

    if _find_mentioned_numeric_column(session, question) is not None:
        return None

    target_phrase = _extract_correlation_target_phrase(question)
    if target_phrase is None:
        return None

    matched_numeric_column = _find_close_numeric_column(session, target_phrase)
    if matched_numeric_column is not None:
        return None

    matched_column = _find_mentioned_column(session, target_phrase)
    if matched_column is not None:
        return f"Cột '{matched_column}' không phải numeric nên không thể dùng làm target để tính tương quan."

    return (
        f"Mình không tìm thấy cột numeric tương ứng với '{target_phrase}' trong dataset. "
        "Bạn muốn dùng cột nào làm target?"
    )


def _extract_correlation_target_phrase(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not any(phrase in normalized for phrase in ("tuong quan", "lien quan", "correlation", "related")):
        return None

    explicit_target = _extract_explicit_target_phrase(normalized)
    if explicit_target is not None:
        return explicit_target

    for marker in (" voi ", " with ", " to "):
        if marker in f" {normalized} ":
            target = f" {normalized} ".split(marker, 1)[1].strip()
            target = re.sub(r"\b(nhat|manh nhat|cao nhat|khong|khong)\b", "", target).strip()
            if target and not _is_generic_correlation_target_phrase(target):
                return target
    return None


def _extract_explicit_target_phrase(normalized_question: str) -> str | None:
    patterns = (
        r"(?:lay\s+)?(?:cot|bien)\s+(.+?)\s+lam\s+target",
        r"target\s+(?:la\s+)?(.+?)(?:\s+va\s+|\s+de\s+|\s+tinh\s+|\s+voi\s+|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_question)
        if match:
            target = match.group(1).strip()
            if target and not _is_generic_correlation_target_phrase(target):
                return target
    return None


def _is_generic_correlation_target_phrase(phrase: str) -> bool:
    normalized = _normalize_text(phrase)
    generic_phrases = {
        "cot nao",
        "bien nao",
        "nhom nao",
        "yeu to nao",
        "cac cot",
        "cac cot con lai",
        "cac cot numeric",
        "cac cot numeric con lai",
        "nhung cot numeric con lai",
        "nhung cot con lai",
        "numeric con lai",
        "all numeric columns",
        "remaining numeric columns",
    }
    return normalized in generic_phrases or normalized.endswith("con lai")


def _find_mentioned_column(session: DatasetSession, phrase: str) -> str | None:
    normalized_phrase = _normalize_text(phrase)
    for column in session.dataframe.columns:
        column_name = str(column)
        normalized_column = _normalize_text(column_name.replace("_", " "))
        if _contains_normalized_column(normalized_phrase, normalized_column):
            return column_name
    return None


def _find_close_numeric_column(session: DatasetSession, phrase: str) -> str | None:
    numeric_columns = [
        str(column)
        for column in session.dataframe.columns
        if is_numeric_dtype(session.dataframe[str(column)]) and not is_bool_dtype(session.dataframe[str(column)])
    ]
    return _find_close_column_name(phrase, numeric_columns)


def _correlation_answer(question: str, table: list[dict[str, Any]]) -> str | None:
    target_column = _find_target_in_correlation_table(question, table)
    if target_column is None:
        return "Đã tính xong ma trận tương quan cho các cột numeric đã chọn."

    target_row = next((row for row in table if row.get("column") == target_column), None)
    if target_row is None:
        return None

    candidates: list[tuple[str, float]] = []
    for column, value in target_row.items():
        if column == "column" or column == target_column:
            continue
        if isinstance(value, int | float) and not isinstance(value, bool) and not math.isnan(float(value)):
            candidates.append((column, float(value)))

    if not candidates:
        return None

    strongest_column, coefficient = max(candidates, key=lambda item: abs(item[1]))
    direction = "dương" if coefficient >= 0 else "âm"
    return (
        f"Dữ liệu cho thấy '{strongest_column}' có tương quan {direction} mạnh nhất với '{target_column}' "
        f"trong các cột numeric đã kiểm tra, với hệ số tương quan khoảng {coefficient:.3f}. "
        "Lưu ý: tương quan không khẳng định quan hệ nhân quả."
    )


def _find_target_in_correlation_table(question: str, table: list[dict[str, Any]]) -> str | None:
    normalized_question = _normalize_text(question)
    columns = [str(row.get("column")) for row in table if row.get("column") is not None]
    for column in columns:
        normalized_column = _normalize_text(column.replace("_", " "))
        if _contains_normalized_column(normalized_question, normalized_column):
            return column

    target_phrase = _extract_correlation_target_phrase(question)
    if target_phrase is not None:
        return _find_close_column_name(target_phrase, columns)
    return None


def _contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    if re.search(rf"(?<!\w){re.escape(normalized_column)}(?!\w)", normalized_text):
        return True
    return normalized_column.replace(" ", "") in normalized_text.replace(" ", "")


def _find_close_column_name(phrase: str, columns: list[str]) -> str | None:
    normalized_phrase = _normalize_text(phrase)
    lookup = {_normalize_text(column.replace("_", " ")): column for column in columns}
    matches = difflib.get_close_matches(normalized_phrase, list(lookup), n=1, cutoff=0.86)
    if not matches:
        return None
    return lookup[matches[0]]


def _normalize_text(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in stripped if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d").replace("_", " ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.strip().split())


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}".replace(",", ".")
    return str(value)
