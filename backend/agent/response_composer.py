from typing import Any
import re

from backend.agent.correlation_helpers import correlation_answer
from backend.schemas import ChatResponse, ToolTraceItem
from backend.tools.safe_pandas import ToolResult


def build_answer(question: str, tool_result: ToolResult) -> str:
    if tool_result.tool_name == "profile_dataset" and isinstance(
        tool_result.data, dict
    ):
        rows = tool_result.data.get("rows")
        columns = tool_result.data.get("columns")
        return f"Dữ liệu hiện có {_format_number(rows)} dòng và {_format_number(columns)} cột."

    if tool_result.tool_name == "list_columns" and isinstance(tool_result.data, dict):
        columns = tool_result.data.get("columns", [])
        return (
            "Dataset có các cột: " + ", ".join(str(column) for column in columns) + "."
        )

    if tool_result.tool_name == "detect_missing_values":
        missing_rows = [
            row for row in tool_result.table or [] if row.get("missing_count", 0) > 0
        ]
        if not missing_rows:
            return "Dữ liệu không có giá trị thiếu trong các cột đã kiểm tra."
        details = ", ".join(
            f"{row['column']}: {row['missing_count']}" for row in missing_rows
        )
        return f"Dữ liệu có giá trị thiếu ở các cột sau: {details}."

    if tool_result.tool_name == "describe_numeric":
        answer = _describe_numeric_answer(question, tool_result.table or [])
        if answer:
            return answer

    if tool_result.tool_name == "correlation_analysis":
        answer = correlation_answer(question, tool_result.table or [])
        if answer:
            return answer

    if tool_result.tool_name == "conditional_percentage" and isinstance(
        tool_result.data, dict
    ):
        column = tool_result.data.get("column")
        condition = _condition_answer_label(
            str(column),
            str(tool_result.data.get("operator")),
            tool_result.data.get("value"),
        )
        matched_rows = tool_result.data.get("matched_rows")
        valid_rows = tool_result.data.get("valid_rows")
        percent = tool_result.data.get("percent_of_valid")
        return (
            f"{_format_number(matched_rows)} / {_format_number(valid_rows)} giá trị hợp lệ của {column} "
            f"{condition}, chiếm khoảng {_format_number(percent)}%."
        )

    if tool_result.tool_name == "aggregate_metric":
        answer = _aggregate_metric_answer(tool_result.table or [])
        if answer:
            return answer

    if tool_result.tool_name == "generate_chart_spec":
        chart_type = (tool_result.chart_spec or {}).get("chart_type", "chart")
        if chart_type == "correlation_heatmap":
            columns = (tool_result.chart_spec or {}).get("columns", [])
            return f"Đã tạo heatmap tương quan cho {_format_number(len(columns))} cột numeric."
        if chart_type == "histogram":
            column = (tool_result.chart_spec or {}).get("x", "cột đã chọn")
            bins = (tool_result.chart_spec or {}).get("bins", 20)
            return (
                f"Histogram của {column} hiển thị tần suất các giá trị theo khoảng; "
                f"trục X là {column}, trục Y là số bản ghi, chia khoảng {bins} bins."
            )
        return f"Đã tạo biểu đồ dạng {chart_type} cho các cột đã chọn."

    if tool_result.table:
        return f"Đã tính xong bảng kết quả bằng tool '{tool_result.tool_name}'."

    return tool_result.message


def response_type(tool_result: ToolResult) -> str:
    if tool_result.chart_spec is not None:
        return "chart"
    if tool_result.table is not None:
        return "table"
    return "answer"


def clarification_response(
    session_id: str,
    message: str,
    traces: list[ToolTraceItem],
    options: list[str] | None = None,
) -> ChatResponse:
    return ChatResponse(
        session_id=session_id,
        answer=message,
        response_type="clarification",
        tool_trace=traces,
        should_clarify=True,
        clarification_options=options or None,
    )


def validation_clarification_message(tool_name: str, validation_message: str) -> str:
    required_match = re.search(r"'([^']+)' is required", validation_message)
    if required_match:
        missing_field = required_match.group(1)
        return (
            f"Mình còn thiếu thông tin `{missing_field}` để chạy phân tích này. "
            "Bạn hãy nêu rõ cột hoặc điều kiện cần dùng trong dataset."
        )
    if "does not exist" in validation_message:
        return (
            f"Mình chưa tìm thấy cột phù hợp trong dataset: {validation_message} "
            "Bạn có thể kiểm tra lại tên cột hoặc dùng một tên gần với schema hiện có."
        )
    if "must be numeric" in validation_message:
        return (
            f"Cột được chọn chưa phù hợp cho phân tích numeric: {validation_message} "
            "Bạn hãy chọn một cột số khác."
        )
    return (
        f"Mình chưa thể chạy `{tool_name}` vì tham số chưa hợp lệ: {validation_message}"
    )


def _aggregate_metric_answer(table: list[dict[str, Any]]) -> str | None:
    if not table:
        return None
    result_columns = [
        column
        for column in table[0]
        if column.startswith(("mean_", "sum_", "min_", "max_", "median_", "count_"))
    ]
    group_columns = [column for column in table[0] if column not in result_columns]
    if len(result_columns) != 1 or len(group_columns) != 1:
        return None

    result_column = result_columns[0]
    group_column = group_columns[0]
    operation, metric_column = result_column.split("_", 1)
    operation_label = {
        "mean": "trung bình",
        "sum": "tổng",
        "min": "nhỏ nhất",
        "max": "lớn nhất",
        "median": "median",
        "count": "số lượng",
    }.get(operation, operation)
    details = ", ".join(
        f"{row.get(group_column)} = {_format_number(row.get(result_column))}"
        for row in table[:5]
    )
    return f"{metric_column} {operation_label} theo {group_column}: {details}."


def _describe_numeric_answer(question: str, table: list[dict[str, Any]]) -> str | None:
    if len(table) != 1:
        return None
    row = table[0]
    column = row.get("column")
    normalized = _normalize_answer_text(question)
    mean = row.get("mean")
    median = row.get("median")
    min_value = row.get("min")
    max_value = row.get("max")
    count = row.get("count")
    suffix = (
        "%"
        if any(
            token in normalized
            for token in ("ty le phan tram", "phan tram", "percent", "percentage")
        )
        else ""
    )

    if any(token in normalized for token in ("trung binh", "average", "mean", "avg")):
        return f"{column} trung bình là {_format_number(mean)}{suffix} trên {_format_number(count)} giá trị hợp lệ."

    return (
        f"{column}: mean={_format_number(mean)}{suffix}, median={_format_number(median)}, "
        f"min-max={_format_number(min_value)}-{_format_number(max_value)} "
        f"(n={_format_number(count)})."
    )


def _format_number(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    if isinstance(value, float) and value.is_integer():
        return f"{int(value):,}".replace(",", ".")
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _condition_answer_label(column: str, operator: str, value: Any) -> str:
    labels = {
        "lt": f"dưới {value}",
        "lte": f"nhỏ hơn hoặc bằng {value}",
        "gt": f"trên {value}",
        "gte": f"lớn hơn hoặc bằng {value}",
        "eq": f"bằng {value}",
        "ne": f"khác {value}",
        "is_missing": "bị thiếu",
        "is_not_missing": "không bị thiếu",
    }
    return labels.get(operator, f"thỏa điều kiện {operator} {value}")


def _normalize_answer_text(text: str) -> str:
    import unicodedata

    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in stripped if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d").replace("\u0111", "d")
    return " ".join(ascii_text.strip().split())
