from dataclasses import dataclass, field
from typing import Any, Literal
import re
import unicodedata

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


ROUTER_CONFIDENCE_THRESHOLD = 0.85
RouteType = Literal["tool", "fallback", "clarify"]


@dataclass(frozen=True)
class RouterDecision:
    route_type: RouteType
    confidence: float
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    @property
    def should_use_tool(self) -> bool:
        return self.route_type == "tool" and self.confidence >= ROUTER_CONFIDENCE_THRESHOLD


def route_question(dataframe: pd.DataFrame, question: str) -> RouterDecision:
    normalized = _normalize(question)
    if not normalized:
        return _fallback("Question is empty.")

    if _has_any(normalized, ("bao nhieu dong", "bao nhiêu dòng", "row count", "number of rows", "so dong", "số dòng")):
        return _tool("profile_dataset", {}, 0.96)

    if _has_any(normalized, ("bao nhieu cot", "bao nhiêu cột", "number of columns", "so cot", "số cột")):
        return _tool("profile_dataset", {}, 0.96)

    if _has_any(normalized, ("nhung cot nao", "những cột nào", "danh sach cot", "danh sách cột", "list columns", "columns")):
        return _tool("list_columns", {}, 0.95)

    if _has_any(normalized, ("thieu du lieu", "thiếu dữ liệu", "missing", "null", "na values", "cot nao thieu", "cột nào thiếu")):
        return _tool("detect_missing_values", {}, 0.95)

    if _has_any(normalized, ("mo ta", "mô tả", "describe", "summary", "thong ke", "thống kê")):
        column = _find_column(dataframe, normalized)
        if column is not None:
            if _is_numeric_column(dataframe, column):
                return _tool("describe_numeric", {"column": column}, 0.93)
            return _clarify(f"Cột '{column}' không phải numeric. Hãy chọn một cột numeric để mô tả.")
        if _has_any(normalized, ("numeric", "so", "số", "number")):
            return _tool("describe_numeric", {}, 0.9)

    if _has_any(normalized, ("top", "value counts", "dem", "đếm", "pho bien", "phổ biến", "tan suat", "tần suất")):
        column = _find_column(dataframe, normalized)
        if column is not None:
            return _tool("value_counts", {"column": column, "top_n": _extract_top_n(normalized)}, 0.91)

    aggregate_operation = _detect_aggregation(normalized)
    if aggregate_operation is not None:
        metric_column = _find_metric_column(dataframe, normalized)
        group_column = _find_group_column(dataframe, normalized, exclude={metric_column} if metric_column else set())
        if metric_column and group_column:
            return _tool(
                "aggregate_metric",
                {"metric_column": metric_column, "group_by": group_column, "operation": aggregate_operation},
                0.9,
            )
        if _has_any(normalized, ("theo nhom", "theo nhóm", "by group", "group by", "theo")):
            return _clarify("Bạn muốn tính metric nào và nhóm theo cột nào?")

    if _has_any(
        normalized,
        (
            "ve bieu do",
            "vẽ biểu đồ",
            "chart",
            "plot",
            "visualize",
            "bieu do",
            "biểu đồ",
            "histogram",
            "scatter",
            "heatmap",
            "boxplot",
            "pie chart",
        ),
    ):
        chart_type = _detect_chart_type(normalized)
        metric_column = _find_metric_column(dataframe, normalized)
        group_column = _find_group_column(dataframe, normalized, exclude={metric_column} if metric_column else set())

        if chart_type == "histogram":
            column = metric_column or _find_column(dataframe, normalized)
            if column and _is_numeric_column(dataframe, column):
                return _tool("generate_chart_spec", {"chart_type": "histogram", "x": column}, 0.9)

        if chart_type == "correlation_heatmap":
            return _tool("generate_chart_spec", {"chart_type": "correlation_heatmap"}, 0.88)

        if metric_column and group_column:
            return _tool(
                "generate_chart_spec",
                {"chart_type": chart_type, "x": group_column, "y": metric_column},
                0.88,
            )
        return _clarify("Bạn muốn vẽ biểu đồ cho metric nào và theo cột nào?")

    return _fallback("Router confidence is low; use LLM fallback.")


def _tool(tool_name: str, arguments: dict[str, Any], confidence: float) -> RouterDecision:
    return RouterDecision(route_type="tool", confidence=confidence, tool_name=tool_name, arguments=arguments)


def _fallback(message: str) -> RouterDecision:
    return RouterDecision(route_type="fallback", confidence=0.0, message=message)


def _clarify(message: str) -> RouterDecision:
    return RouterDecision(route_type="clarify", confidence=0.0, message=message)


def _detect_aggregation(normalized: str) -> str | None:
    if _has_any(normalized, ("trung binh", "trung bình", "average", "mean", "avg")):
        return "mean"
    if _has_any(normalized, ("tong", "tổng", "sum", "total")):
        return "sum"
    if _has_any(normalized, ("median", "trung vi", "trung vị")):
        return "median"
    if _has_any(normalized, ("min", "nho nhat", "nhỏ nhất")):
        return "min"
    if _has_any(normalized, ("max", "lon nhat", "lớn nhất", "cao nhat", "cao nhất")):
        return "max"
    return None


def _detect_chart_type(normalized: str) -> str:
    if _has_any(normalized, ("scatter", "phan tan", "phân tán")):
        return "scatter"
    if _has_any(normalized, ("line", "duong", "đường")):
        return "line"
    if _has_any(normalized, ("histogram", "phan phoi", "phân phối")):
        return "histogram"
    if _has_any(normalized, ("heatmap", "correlation", "tuong quan", "tương quan")):
        return "correlation_heatmap"
    if _has_any(normalized, ("box", "boxplot")):
        return "box"
    if _has_any(normalized, ("pie", "tron", "tròn")):
        return "pie"
    return "bar"


def _find_metric_column(dataframe: pd.DataFrame, normalized: str) -> str | None:
    for column in _matching_columns(dataframe, normalized):
        if _is_numeric_column(dataframe, column):
            return column
    numeric_columns = [str(column) for column in dataframe.columns if _is_numeric_column(dataframe, str(column))]
    return numeric_columns[0] if len(numeric_columns) == 1 else None


def _find_group_column(dataframe: pd.DataFrame, normalized: str, exclude: set[str] | None = None) -> str | None:
    exclude = exclude or set()
    for column in _matching_columns(dataframe, normalized):
        if column in exclude:
            continue
        if not _is_numeric_column(dataframe, column):
            return column
    categorical_columns = [str(column) for column in dataframe.columns if not _is_numeric_column(dataframe, str(column))]
    candidates = [column for column in categorical_columns if column not in exclude]
    return candidates[0] if len(candidates) == 1 else None


def _find_column(dataframe: pd.DataFrame, normalized: str) -> str | None:
    matches = _matching_columns(dataframe, normalized)
    return matches[0] if matches else None


def _matching_columns(dataframe: pd.DataFrame, normalized: str) -> list[str]:
    matches = []
    for column in dataframe.columns:
        column_name = str(column)
        normalized_column = _normalize_identifier(column_name)
        if re.search(rf"(?<!\w){re.escape(normalized_column)}(?!\w)", normalized):
            matches.append(column_name)
    return matches


def _extract_top_n(normalized: str) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", normalized)
    if not match:
        return 10
    return max(1, min(50, int(match.group(1))))


def _is_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    return is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_normalize(phrase) in text for phrase in phrases)


def _normalize(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in stripped if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d")
    ascii_text = ascii_text.replace("_", " ")
    ascii_text = re.sub(r"[^a-z0-9_]+", " ", ascii_text)
    return " ".join(ascii_text.strip().split())


def _normalize_identifier(identifier: str) -> str:
    return _normalize(identifier.replace("_", " "))
