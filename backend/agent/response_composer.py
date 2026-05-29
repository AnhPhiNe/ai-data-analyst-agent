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
            f"{row['column']}: {row['missing_count']} dòng ({row['missing_percent']}%)"
            for row in missing_rows
        )
        return f"Dữ liệu có giá trị thiếu ở các cột sau: {details}."

    if tool_result.tool_name == "data_quality_report" and isinstance(
        tool_result.data, dict
    ):
        return _data_quality_answer(tool_result.data)

    if tool_result.tool_name == "describe_numeric":
        answer = _describe_numeric_answer(question, tool_result.table or [])
        if answer:
            return answer

    if tool_result.tool_name == "outlier_detection" and isinstance(
        tool_result.data, dict
    ):
        return _outlier_answer(tool_result.data)

    if tool_result.tool_name == "value_counts" and isinstance(tool_result.data, dict):
        answer = _value_counts_answer(
            question, tool_result.data, tool_result.table or []
        )
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
            return _append_outlier_context(question, answer, tool_result.data)

    if tool_result.tool_name == "compare_groups":
        answer = _compare_groups_answer(tool_result.table or [], tool_result.data)
        if answer:
            return _append_outlier_context(question, answer, tool_result.data)

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

    if tool_result.tool_name == "query_table_sql":
        row_count = len(tool_result.table or [])
        sql = ""
        if isinstance(tool_result.data, dict):
            sql_value = tool_result.data.get("sql")
            if isinstance(sql_value, str) and sql_value.strip():
                sql = f"\n\n```sql\n{sql_value.strip()}\n```"
        return (
            "Đã chạy truy vấn SQL read-only đã được validate và trả về "
            f"{_format_number(row_count)} dòng kết quả.{sql}"
        )

    if tool_result.table:
        return f"Đã tính xong bảng kết quả bằng tool '{tool_result.tool_name}'."

    return tool_result.message


def response_type(tool_result: ToolResult) -> str:
    if tool_result.chart_spec is not None:
        return "chart"
    if tool_result.table is not None:
        return "table"
    return "answer"


def build_multi_step_answer(
    question: str,
    tool_results: list[ToolResult],
    warnings: list[str] | None = None,
) -> str:
    if not tool_results:
        return "Mình chưa chạy được bước phân tích nào cho câu hỏi này."

    normalized = _normalize_answer_text(question)
    if (
        len(tool_results) == 1
        and tool_results[0].tool_name == "data_quality_report"
        and isinstance(tool_results[0].data, dict)
        and any(
            token in normalized for token in ("nen dung", "de phan tich", "giong id")
        )
    ):
        answer = _data_quality_recommendation_answer(tool_results[0].data)
    else:
        answer_parts = []
        for result in tool_results:
            part = build_answer(question, result)
            if part and part not in answer_parts:
                answer_parts.append(part)
        answer = " ".join(answer_parts)

    if warnings:
        answer = answer + " Lưu ý: " + " ".join(warnings)
    return answer


def _data_quality_recommendation_answer(data: dict[str, Any]) -> str:
    possible_id_columns = [
        str(column) for column in data.get("possible_id_columns") or []
    ]
    constant_columns = [str(column) for column in data.get("constant_columns") or []]
    high_cardinality_columns = [
        str(column) for column in data.get("high_cardinality_columns") or []
    ]
    candidates = [
        str(column) for column in data.get("analysis_candidate_columns") or []
    ]

    parts = []
    if possible_id_columns:
        parts.append("cột giống ID: " + ", ".join(possible_id_columns[:5]))
    if constant_columns:
        parts.append("cột hằng số nên tránh: " + ", ".join(constant_columns[:5]))
    if high_cardinality_columns:
        parts.append(
            "cột high-cardinality cần cẩn thận: "
            + ", ".join(high_cardinality_columns[:5])
        )
    if candidates:
        parts.append("cột nên ưu tiên phân tích: " + ", ".join(candidates[:8]))
    if not parts:
        return "Không phát hiện cột giống ID hoặc cột cần tránh rõ ràng; có thể phân tích các cột còn lại theo mục tiêu câu hỏi."
    return "Tóm tắt lựa chọn cột: " + "; ".join(parts) + "."


def clarification_response(
    session_id: str,
    message: str,
    traces: list[ToolTraceItem],
    options: list[str] | None = None,
) -> ChatResponse:
    del options
    return ChatResponse(
        session_id=session_id,
        answer=_full_question_clarification_message(message),
        response_type="clarification",
        tool_trace=traces,
        should_clarify=True,
        clarification_options=None,
    )


def _full_question_clarification_message(message: str) -> str:
    example = (
        "Bạn hãy nhập lại thành một câu hỏi đầy đủ, ví dụ: "
        "`Trung bình salary theo department là bao nhiêu?`, "
        "`So sánh Exam_Score theo Gender`, hoặc "
        "`Cột salary có outlier không?`"
    )
    if "nhập lại thành một câu hỏi đầy đủ" in message:
        return message
    return f"{message} {example}"


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


def _data_quality_answer(data: dict[str, Any]) -> str:
    issue_count = int(data.get("issue_count", 0) or 0)
    duplicate_rows = int(data.get("duplicate_rows", 0) or 0)
    missing_columns = data.get("missing_columns") or []
    constant_columns = data.get("constant_columns") or []
    high_cardinality_columns = data.get("high_cardinality_columns") or []
    possible_id_columns = data.get("possible_id_columns") or []

    if issue_count == 0:
        return "Không phát hiện tín hiệu chất lượng dữ liệu đáng chú ý trong dataset."

    parts = []
    if duplicate_rows:
        parts.append(f"{_format_number(duplicate_rows)} dòng trùng lặp")
    if missing_columns:
        parts.append(
            "thiếu dữ liệu ở "
            + ", ".join(str(column) for column in missing_columns[:5])
        )
    if constant_columns:
        parts.append(
            "cột hằng số: " + ", ".join(str(column) for column in constant_columns[:5])
        )
    if high_cardinality_columns:
        parts.append(
            "cột high-cardinality: "
            + ", ".join(str(column) for column in high_cardinality_columns[:5])
        )
    if possible_id_columns:
        parts.append(
            "cột có vẻ là ID: "
            + ", ".join(str(column) for column in possible_id_columns[:5])
        )
    return "Phát hiện các tín hiệu chất lượng dữ liệu: " + "; ".join(parts) + "."


def _outlier_answer(data: dict[str, Any]) -> str:
    column = data.get("column")
    outlier_count = data.get("outlier_count")
    valid_count = data.get("valid_count")
    outlier_percent = data.get("outlier_percent")
    lower_bound = data.get("lower_bound")
    upper_bound = data.get("upper_bound")
    if int(outlier_count or 0) == 0:
        return (
            f"Không phát hiện outlier trong {column} theo IQR "
            f"(ngưỡng {_format_number(lower_bound)} đến {_format_number(upper_bound)})."
        )
    return (
        f"Phát hiện {_format_number(outlier_count)} / {_format_number(valid_count)} "
        f"giá trị outlier trong {column} theo IQR "
        f"({_format_number(outlier_percent)}%). Ngưỡng hợp lệ xấp xỉ "
        f"{_format_number(lower_bound)} đến {_format_number(upper_bound)}."
    )


def _value_counts_answer(
    question: str, data: dict[str, Any], table: list[dict[str, Any]]
) -> str | None:
    column = data.get("column")
    unique_count = data.get("unique_count")
    non_null_count = data.get("non_null_count")
    normalized = _normalize_answer_text(question)
    if any(
        token in normalized
        for token in (
            "khac nhau",
            "distinct",
            "unique",
            "gia tri rieng",
            "so luong gia tri",
        )
    ):
        return (
            f"Cột {column} có {_format_number(unique_count)} giá trị khác nhau "
            f"trên {_format_number(non_null_count)} giá trị hợp lệ."
        )
    if table:
        first = table[0]
        return (
            f"Giá trị phổ biến nhất trong {column} là {first.get('value')} "
            f"({_format_number(first.get('count'))} dòng, {_format_number(first.get('percent'))}%)."
        )
    return None


def _compare_groups_answer(
    table: list[dict[str, Any]], data: dict[str, Any] | list[Any] | None
) -> str | None:
    if not table or not isinstance(data, dict):
        return None
    metric_column = str(data.get("metric_column"))
    group_by = str(data.get("group_by"))
    mean_column = f"mean_{metric_column}"
    if mean_column not in table[0]:
        return None
    details = ", ".join(
        f"{row.get(group_by)}: mean={_format_number(row.get(mean_column))}, "
        f"n={_format_number(row.get('count'))}"
        for row in table[:5]
    )
    return f"So sánh {metric_column} theo {group_by}: {details}."


def _append_outlier_context(
    question: str, answer: str, data: dict[str, Any] | list[Any] | None
) -> str:
    if not isinstance(data, dict):
        return answer
    normalized = _normalize_answer_text(question)
    if not any(token in normalized for token in ("outlier", "ngoai lai", "bat thuong")):
        return answer
    summary = data.get("outlier_summary")
    if not isinstance(summary, dict):
        return answer
    outlier_count = int(summary.get("outlier_count") or 0)
    valid_count = int(summary.get("valid_count") or 0)
    if outlier_count <= 0:
        return answer + " Không phát hiện outlier theo IQR trong metric này."
    return (
        answer
        + f" Lưu ý: metric này có {_format_number(outlier_count)} / {_format_number(valid_count)} "
        + "giá trị outlier theo IQR, nên trung bình của nhóm có thể bị kéo lệch; hãy xem thêm median/min/max trong bảng."
    )


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
