from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal
import unicodedata

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.agent.column_resolver import (
    contains_normalized_column,
    normalize_identifier,
    resolve_column,
)


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


@dataclass(frozen=True)
class RouteCandidate:
    intent: str
    priority: int
    score: float
    decision: RouterDecision


def route_question(dataframe: pd.DataFrame, question: str) -> RouterDecision:
    normalized = _normalize(question)
    if not normalized:
        return _fallback("Question is empty.")
    return _route_with_candidates(dataframe, normalized)


def _route_with_candidates(dataframe: pd.DataFrame, normalized: str) -> RouterDecision:
    candidates = _build_route_candidates(dataframe, normalized)
    if not candidates:
        return _fallback("Router confidence is low; use LLM fallback.")

    candidates = _filter_compatible_candidates(candidates, normalized)
    candidates = sorted(
        candidates,
        key=lambda candidate: (candidate.score, candidate.priority, candidate.decision.confidence),
        reverse=True,
    )
    best = candidates[0]
    if len(candidates) > 1 and _has_candidate_conflict(best, candidates[1]):
        return _fallback("Router detected conflicting intents; use LLM fallback.")
    return best.decision


def _build_route_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    candidates.extend(_profile_candidates(normalized))
    candidates.extend(_percentage_candidates(dataframe, normalized))
    candidates.extend(_correlation_candidates(dataframe, normalized))
    candidates.extend(_missing_candidates(normalized))
    candidates.extend(_list_column_candidates(normalized))
    candidates.extend(_distribution_candidates(dataframe, normalized))
    candidates.extend(_describe_candidates(dataframe, normalized))
    candidates.extend(_value_count_candidates(dataframe, normalized))
    candidates.extend(_aggregate_candidates(dataframe, normalized))
    candidates.extend(_chart_candidates(dataframe, normalized))
    return candidates


def _candidate(intent: str, priority: int, decision: RouterDecision, evidence: float = 1.0) -> RouteCandidate:
    score = priority + (decision.confidence * 10) + evidence
    return RouteCandidate(intent=intent, priority=priority, score=score, decision=decision)


def _profile_candidates(normalized: str) -> list[RouteCandidate]:
    if _has_any(normalized, ("bao nhieu dong", "row count", "number of rows", "how many rows", "so dong")):
        return [_candidate("profile", 100, _tool("profile_dataset", {}, 0.96), evidence=2.0)]
    if _has_any(normalized, ("bao nhieu cot", "number of columns", "how many columns", "so cot")):
        return [_candidate("profile", 100, _tool("profile_dataset", {}, 0.96), evidence=2.0)]
    return []


def _percentage_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_any(normalized, ("ty le", "phan tram", "percent", "percentage")):
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(dataframe, normalized)
    if column is None:
        return []

    condition = _extract_numeric_condition(normalized)
    if condition is not None:
        if _is_numeric_column(dataframe, column):
            operator, value = condition
            return [
                _candidate(
                    "percentage",
                    95,
                    _tool("conditional_percentage", {"column": column, "operator": operator, "value": value}, 0.93),
                    evidence=3.0,
                )
            ]
        return [
            _candidate(
                "percentage",
                95,
                _clarify(f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de tinh ty le."),
            )
        ]

    if not _is_numeric_column(dataframe, column):
        categorical_condition = _extract_categorical_percentage_condition(dataframe, column, normalized)
        if categorical_condition is not None:
            operator, value = categorical_condition
            return [
                _candidate(
                    "percentage",
                    95,
                    _tool("conditional_percentage", {"column": column, "operator": operator, "value": value}, 0.92),
                    evidence=3.0,
                )
            ]
    return []


def _correlation_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_correlation_intent(normalized) or _is_chart_request(normalized):
        return []
    columns = _find_correlation_columns(dataframe, normalized)
    if len(columns) >= 2:
        return [
            _candidate(
                "correlation",
                100,
                _tool("correlation_analysis", {"columns": columns}, 0.93),
                evidence=min(4.0, float(len(columns))),
            )
        ]
    return []


def _missing_candidates(normalized: str) -> list[RouteCandidate]:
    if _has_any(normalized, ("thieu du lieu", "missing", "null", "na values", "cot nao thieu")):
        return [_candidate("missing", 90, _tool("detect_missing_values", {}, 0.95), evidence=2.0)]
    return []


def _list_column_candidates(normalized: str) -> list[RouteCandidate]:
    if _has_any(normalized, ("nhung cot nao", "danh sach cot", "list columns", "columns")):
        return [_candidate("list_columns", 50, _tool("list_columns", {}, 0.95))]
    return []


def _distribution_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_any(normalized, ("phan phoi", "distribution", "histogram")):
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(dataframe, normalized)
    if column is None:
        return [_candidate("distribution", 85, _clarify("Ban muon xem phan phoi cua cot numeric nao?"))]
    if not _is_numeric_column(dataframe, column):
        return [
            _candidate(
                "distribution",
                85,
                _clarify(f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de ve phan phoi."),
            )
        ]
    return [
        _candidate(
            "distribution",
            85,
            _tool(
                "generate_chart_spec",
                {"chart_type": "histogram", "x": column, "bins": _histogram_bins(dataframe, column)},
                0.93,
            ),
            evidence=3.0,
        )
    ]


def _describe_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_any(normalized, ("mo ta", "describe", "summary", "thong ke")):
        return []

    column = _find_column(dataframe, normalized)
    if column is not None:
        if _is_numeric_column(dataframe, column):
            return [_candidate("describe", 60, _tool("describe_numeric", {"column": column}, 0.93), evidence=2.0)]
        return [
            _candidate(
                "describe",
                60,
                _clarify(f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de mo ta."),
            )
        ]
    if _has_any(normalized, ("numeric", "so", "number")):
        return [_candidate("describe", 60, _tool("describe_numeric", {}, 0.9))]
    return []


def _value_count_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_any(normalized, ("top", "value counts", "dem", "pho bien", "tan suat")):
        return []
    column = _find_column(dataframe, normalized)
    if column is None:
        return []
    return [
        _candidate(
            "value_counts",
            70,
            _tool("value_counts", {"column": column, "top_n": _extract_top_n(normalized)}, 0.91),
            evidence=2.0,
        )
    ]


def _aggregate_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    operation = _detect_aggregation(normalized)
    if operation is None:
        return []

    metric_column = _find_metric_column(dataframe, normalized)
    has_group_intent = _has_group_intent(normalized)
    group_column = (
        _find_group_column(dataframe, normalized, exclude={metric_column} if metric_column else set())
        if has_group_intent
        else None
    )

    if metric_column and group_column:
        return [
            _candidate(
                "aggregate",
                75,
                _tool(
                    "aggregate_metric",
                    {"metric_column": metric_column, "group_by": group_column, "operation": operation},
                    0.9,
                ),
                evidence=3.0,
            )
        ]
    if metric_column and not has_group_intent:
        return [
            _candidate(
                "aggregate",
                75,
                _tool("describe_numeric", {"column": metric_column}, 0.91),
                evidence=2.0,
            )
        ]
    if has_group_intent:
        return [_candidate("aggregate", 75, _clarify("Ban muon tinh metric nao va nhom theo cot nao?"))]
    return []


def _chart_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_chart_intent(normalized):
        return []

    chart_type = _detect_chart_type(normalized)
    metric_column = _find_metric_column(dataframe, normalized)
    group_column = _find_group_column(dataframe, normalized, exclude={metric_column} if metric_column else set())

    if chart_type == "histogram":
        column = metric_column or _find_column(dataframe, normalized)
        if column and _is_numeric_column(dataframe, column):
            return [
                _candidate(
                    "chart",
                    80,
                    _tool(
                        "generate_chart_spec",
                        {"chart_type": "histogram", "x": column, "bins": _histogram_bins(dataframe, column)},
                        0.9,
                    ),
                    evidence=2.0,
                )
            ]

    if chart_type == "correlation_heatmap":
        return [
            _candidate(
                "chart",
                80,
                _tool("generate_chart_spec", {"chart_type": "correlation_heatmap"}, 0.88),
                evidence=2.0,
            )
        ]

    if chart_type == "scatter":
        numeric_columns = [column for column in _matching_columns(dataframe, normalized) if _is_numeric_column(dataframe, column)]
        if len(numeric_columns) >= 2:
            return [
                _candidate(
                    "chart",
                    80,
                    _tool("generate_chart_spec", {"chart_type": "scatter", "x": numeric_columns[0], "y": numeric_columns[1]}, 0.9),
                    evidence=3.0,
                )
            ]

    if metric_column and group_column:
        return [
            _candidate(
                "chart",
                80,
                _tool("generate_chart_spec", {"chart_type": chart_type, "x": group_column, "y": metric_column}, 0.88),
                evidence=2.0,
            )
        ]
    return [_candidate("chart", 80, _clarify("Ban muon ve bieu do cho metric nao va theo cot nao?"))]


def _filter_compatible_candidates(candidates: list[RouteCandidate], normalized: str) -> list[RouteCandidate]:
    intents = {candidate.intent for candidate in candidates}
    filtered = list(candidates)

    if "correlation" in intents:
        filtered = [candidate for candidate in filtered if candidate.intent != "list_columns"]
        if _is_chart_request(normalized):
            filtered = [candidate for candidate in filtered if candidate.intent != "correlation"]
    if "missing" in intents:
        filtered = [candidate for candidate in filtered if candidate.intent != "list_columns"]
    if "distribution" in intents:
        filtered = [candidate for candidate in filtered if candidate.intent not in {"chart", "describe"}]
    if "percentage" in intents:
        filtered = [candidate for candidate in filtered if candidate.intent not in {"aggregate", "describe"}]

    return filtered or candidates


def _has_candidate_conflict(best: RouteCandidate, second: RouteCandidate) -> bool:
    if best.intent == second.intent:
        return False
    return best.score - second.score < 12


def _tool(tool_name: str, arguments: dict[str, Any], confidence: float) -> RouterDecision:
    return RouterDecision(route_type="tool", confidence=confidence, tool_name=tool_name, arguments=arguments)


def _fallback(message: str) -> RouterDecision:
    return RouterDecision(route_type="fallback", confidence=0.0, message=message)


def _clarify(message: str) -> RouterDecision:
    return RouterDecision(route_type="clarify", confidence=0.0, message=message)


def _detect_aggregation(normalized: str) -> str | None:
    if _has_any(normalized, ("trung binh", "average", "mean", "avg")):
        return "mean"
    if _has_any(normalized, ("tong", "sum", "total")):
        return "sum"
    if _has_any(normalized, ("median", "trung vi")):
        return "median"
    if _has_any(normalized, ("min", "nho nhat")):
        return "min"
    if _has_any(normalized, ("max", "lon nhat", "cao nhat")):
        return "max"
    return None


def _has_group_intent(normalized: str) -> bool:
    return _has_any(normalized, ("theo nhom", "by group", "group by", "theo", "nhom")) or " by " in f" {normalized} "


def _has_chart_intent(normalized: str) -> bool:
    return _has_any(
        normalized,
        (
            "ve bieu do",
            "chart",
            "plot",
            "visualize",
            "bieu do",
            "histogram",
            "scatter",
            "heatmap",
            "boxplot",
            "pie chart",
        ),
    )


def _detect_chart_type(normalized: str) -> str:
    if _has_any(normalized, ("scatter", "phan tan")):
        return "scatter"
    if _has_any(normalized, ("line", "duong")):
        return "line"
    if _has_any(normalized, ("histogram", "phan phoi")):
        return "histogram"
    if _has_any(normalized, ("heatmap", "correlation", "tuong quan")):
        return "correlation_heatmap"
    if _has_any(normalized, ("box", "boxplot")):
        return "box"
    if _has_any(normalized, ("pie", "tron")):
        return "pie"
    return "bar"


def _has_correlation_intent(normalized: str) -> bool:
    return _has_any(normalized, ("tuong quan", "correlation", "lien quan", "related"))


def _is_chart_request(normalized: str) -> bool:
    return _has_any(
        normalized,
        (
            "ve bieu do",
            "chart",
            "plot",
            "visualize",
            "bieu do",
            "heatmap",
            "scatter",
            "boxplot",
        ),
    )


def _histogram_bins(dataframe: pd.DataFrame, column: str) -> int:
    series = dataframe[column].dropna()
    row_count = int(series.count())
    unique_count = int(series.nunique())
    if row_count <= 0 or unique_count <= 0:
        return 10
    if unique_count <= 20:
        return max(1, unique_count)
    rice_bins = math.ceil(2 * (row_count ** (1 / 3)))
    return max(8, min(50, unique_count, rice_bins))


def _extract_numeric_condition(normalized: str) -> tuple[str, float] | None:
    patterns = (
        (r"(duoi|nho hon|less than|below|under)\s+(-?\d+(?:\.\d+)?)", "lt"),
        (r"(toi da|nho hon hoac bang|at most|less than or equal)\s+(-?\d+(?:\.\d+)?)", "lte"),
        (r"(tren|lon hon|greater than|above|over)\s+(-?\d+(?:\.\d+)?)", "gt"),
        (r"(toi thieu|lon hon hoac bang|at least|greater than or equal)\s+(-?\d+(?:\.\d+)?)", "gte"),
        (r"(bang|equal to)\s+(-?\d+(?:\.\d+)?)", "eq"),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, normalized)
        if match:
            value = float(match.group(2))
            if value.is_integer():
                return operator, int(value)
            return operator, value
    return None


def _extract_categorical_percentage_condition(
    dataframe: pd.DataFrame, column: str, normalized: str
) -> tuple[str, str] | None:
    values = [str(value) for value in dataframe[column].dropna().astype(str).unique()]
    value_lookup = {_normalize(value): value for value in values}

    for normalized_value, original_value in value_lookup.items():
        if re.search(rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized):
            return "eq", original_value

    yes_value = _find_categorical_value(value_lookup, ("yes", "true", "co", "1"))
    no_value = _find_categorical_value(value_lookup, ("no", "false", "khong", "0"))
    if no_value is not None and any(token in normalized for token in ("khong tham gia", "khong co", "khong")):
        return "eq", no_value
    if yes_value is not None and any(token in normalized for token in ("tham gia", "co", "yes")):
        return "eq", yes_value
    return None


def _find_categorical_value(value_lookup: dict[str, str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in value_lookup:
            return value_lookup[candidate]
    return None


def _find_metric_column(dataframe: pd.DataFrame, normalized: str) -> str | None:
    for column in _matching_columns(dataframe, normalized):
        if _is_numeric_column(dataframe, column):
            return column
    resolved = resolve_column(dataframe, normalized, expected_type="numeric")
    if resolved is not None:
        return resolved
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


def _find_correlation_columns(dataframe: pd.DataFrame, normalized: str) -> list[str]:
    columns = [column for column in _matching_columns(dataframe, normalized) if _is_numeric_column(dataframe, column)]
    resolved = resolve_column(dataframe, normalized, expected_type="numeric")
    if resolved is not None and resolved not in columns:
        columns.append(resolved)
    for phrase in _correlation_column_phrases(normalized):
        resolved_phrase = resolve_column(dataframe, phrase, expected_type="numeric")
        if resolved_phrase is not None and resolved_phrase not in columns:
            columns.append(resolved_phrase)

    numeric_columns = [str(column) for column in dataframe.columns if _is_numeric_column(dataframe, str(column))]
    if len(columns) >= 2:
        return columns
    if len(columns) == 1 and _asks_correlation_columns(normalized):
        return columns + [column for column in numeric_columns if column not in columns]
    if not columns and len(numeric_columns) == 2:
        return numeric_columns
    return columns


def _asks_correlation_columns(normalized: str) -> bool:
    return _has_any(
        normalized,
        (
            "cot nao",
            "nhung cot nao",
            "yeu to nao",
            "yeu to numeric nao",
            "cac cot con lai",
            "cac cot numeric con lai",
            "remaining columns",
            "remaining numeric columns",
        ),
    )


def _correlation_column_phrases(normalized: str) -> list[str]:
    phrases = [normalized]
    for marker in (" voi ", " with ", " va ", " and "):
        if marker in f" {normalized} ":
            phrases.extend(part.strip() for part in f" {normalized} ".split(marker) if part.strip())
    return phrases


def _find_column(dataframe: pd.DataFrame, normalized: str) -> str | None:
    matches = _matching_columns(dataframe, normalized)
    return matches[0] if matches else None


def _matching_columns(dataframe: pd.DataFrame, normalized: str) -> list[str]:
    matches = []
    for column in dataframe.columns:
        column_name = str(column)
        normalized_column = _normalize_identifier(column_name)
        if _contains_normalized_column(normalized, normalized_column):
            matches.append(column_name)
    if matches:
        return matches

    resolved = resolve_column(dataframe, normalized)
    return [resolved] if resolved else []


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
    ascii_text = ascii_text.replace("\u0111", "d").replace("_", " ")
    ascii_text = re.sub(r"[^a-z0-9_]+", " ", ascii_text)
    return " ".join(ascii_text.strip().split())


def _normalize_identifier(identifier: str) -> str:
    return normalize_identifier(identifier)


def _contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    return contains_normalized_column(normalized_text, normalized_column)
