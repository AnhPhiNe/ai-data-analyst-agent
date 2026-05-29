from dataclasses import dataclass, field
import re
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.agent.column_resolver import (
    ExpectedType,
    contains_normalized_column,
    normalize_identifier,
    normalize_text,
    resolve_column,
)
from backend.agent.helpers import (
    detect_aggregation,
    detect_chart_type,
    get_histogram_bins,
    has_any_phrase,
    has_group_intent,
)


ROUTER_CONFIDENCE_THRESHOLD = 0.85
RouteType = Literal["tool", "fallback", "clarify"]

CONFIDENCE_PROFILE = 0.96
CONFIDENCE_MISSING = 0.95
CONFIDENCE_DATA_QUALITY = 0.94
CONFIDENCE_LIST_COLUMNS = 0.95
CONFIDENCE_PERCENTAGE_NUMERIC = 0.93
CONFIDENCE_PERCENTAGE_CATEGORICAL = 0.92
CONFIDENCE_CORRELATION = 0.93
CONFIDENCE_DISTRIBUTION = 0.93
CONFIDENCE_DESCRIBE_COL = 0.93
CONFIDENCE_DESCRIBE_ALL = 0.90
CONFIDENCE_VALUE_COUNTS = 0.91
CONFIDENCE_SORT = 0.91
CONFIDENCE_AGGREGATE = 0.90
CONFIDENCE_COMPARE_GROUPS = 0.91
CONFIDENCE_OUTLIER = 0.92
CONFIDENCE_AGGREGATE_DESCRIBE = 0.91
CONFIDENCE_CHART_HISTOGRAM = 0.90
CONFIDENCE_CHART_HEATMAP = 0.88
CONFIDENCE_CHART_SCATTER = 0.90
CONFIDENCE_CHART_GENERIC = 0.88
CONFIDENCE_SQL_FALLBACK = 0.90
EXPLICIT_NUMERIC_REFERENCE_TOKENS = {
    "age",
    "amount",
    "count",
    "income",
    "metric",
    "order",
    "performance",
    "percentage",
    "price",
    "quantity",
    "rate",
    "revenue",
    "salary",
    "score",
}
EXPLICIT_GROUP_REFERENCE_TOKENS = {
    "branch",
    "category",
    "city",
    "class",
    "country",
    "department",
    "gender",
    "group",
    "location",
    "phong",
    "region",
    "segment",
    "team",
}


@dataclass(frozen=True)
class RouterDecision:
    route_type: RouteType
    confidence: float
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    message: str | None = None

    @property
    def should_use_tool(self) -> bool:
        return (
            self.route_type == "tool" and self.confidence >= ROUTER_CONFIDENCE_THRESHOLD
        )


@dataclass(frozen=True)
class RouteCandidate:
    intent: str
    priority: int
    score: float
    decision: RouterDecision


RouteBuilder = Callable[[pd.DataFrame, str], list[RouteCandidate]]


def route_question(dataframe: pd.DataFrame, question: str) -> RouterDecision:
    normalized = normalize_text(_expand_comparison_symbols(question))
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
        key=lambda candidate: (
            candidate.score,
            candidate.priority,
            candidate.decision.confidence,
        ),
        reverse=True,
    )
    best = candidates[0]
    if len(candidates) > 1 and _has_candidate_conflict(best, candidates[1]):
        return _fallback("Router detected conflicting intents; use LLM fallback.")
    return best.decision


def _build_route_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    for builder in ROUTE_BUILDERS:
        candidates.extend(builder(dataframe, normalized))
    return candidates


def _candidate(
    intent: str, priority: int, decision: RouterDecision, evidence: float = 1.0
) -> RouteCandidate:
    score = priority + (decision.confidence * 10) + evidence
    return RouteCandidate(
        intent=intent, priority=priority, score=score, decision=decision
    )


def _profile_candidates(normalized: str) -> list[RouteCandidate]:
    if has_group_intent(normalized):
        return []
    if has_any_phrase(
        normalized,
        ("bao nhieu dong", "row count", "number of rows", "how many rows", "so dong"),
    ):
        return [
            _candidate(
                "profile",
                100,
                _tool("profile_dataset", {}, CONFIDENCE_PROFILE),
                evidence=2.0,
            )
        ]
    if has_any_phrase(
        normalized, ("bao nhieu cot", "number of columns", "how many columns", "so cot")
    ):
        return [
            _candidate(
                "profile",
                100,
                _tool("profile_dataset", {}, CONFIDENCE_PROFILE),
                evidence=2.0,
            )
        ]
    return []


def _percentage_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not has_any_phrase(normalized, ("ty le", "phan tram", "percent", "percentage")):
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(
        dataframe, normalized
    )
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
                    _tool(
                        "conditional_percentage",
                        {"column": column, "operator": operator, "value": value},
                        CONFIDENCE_PERCENTAGE_NUMERIC,
                    ),
                    evidence=3.0,
                )
            ]
        return [
            _candidate(
                "percentage",
                95,
                _clarify(
                    f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de tinh ty le."
                ),
            )
        ]

    if not _is_numeric_column(dataframe, column):
        categorical_condition = _extract_categorical_percentage_condition(
            dataframe, column, normalized
        )
        if categorical_condition is not None:
            operator, value = categorical_condition
            return [
                _candidate(
                    "percentage",
                    95,
                    _tool(
                        "conditional_percentage",
                        {"column": column, "operator": operator, "value": value},
                        CONFIDENCE_PERCENTAGE_CATEGORICAL,
                    ),
                    evidence=3.0,
                )
            ]
    return []


def _correlation_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not _has_correlation_intent(normalized) or _is_chart_request(normalized):
        return []
    columns = _find_correlation_columns(dataframe, normalized)
    if len(columns) >= 2:
        return [
            _candidate(
                "correlation",
                100,
                _tool(
                    "correlation_analysis", {"columns": columns}, CONFIDENCE_CORRELATION
                ),
                evidence=min(4.0, float(len(columns))),
            )
        ]
    return []


def _missing_candidates(normalized: str) -> list[RouteCandidate]:
    if has_any_phrase(
        normalized, ("thieu du lieu", "missing", "null", "na values", "cot nao thieu")
    ):
        return [
            _candidate(
                "missing",
                90,
                _tool("detect_missing_values", {}, CONFIDENCE_MISSING),
                evidence=2.0,
            )
        ]
    return []


def _data_quality_candidates(normalized: str) -> list[RouteCandidate]:
    if has_any_phrase(
        normalized,
        (
            "chat luong du lieu",
            "data quality",
            "van de du lieu",
            "du lieu co van de",
            "duplicate",
            "trung lap",
            "dong trung",
            "cot hang so",
            "constant column",
            "high cardinality",
            "id column",
            "cot id",
            "giong id",
            "khoa chinh",
            "primary key",
            "nen dung de phan tich",
            "dung de phan tich",
        ),
    ):
        return [
            _candidate(
                "data_quality",
                92,
                _tool("data_quality_report", {}, CONFIDENCE_DATA_QUALITY),
                evidence=2.5,
            )
        ]
    return []


def _list_column_candidates(normalized: str) -> list[RouteCandidate]:
    if has_any_phrase(
        normalized, ("nhung cot nao", "danh sach cot", "list columns", "columns")
    ):
        return [
            _candidate(
                "list_columns", 50, _tool("list_columns", {}, CONFIDENCE_LIST_COLUMNS)
            )
        ]
    return []


def _distribution_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not has_any_phrase(normalized, ("phan phoi", "distribution", "histogram")):
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(
        dataframe, normalized
    )
    if column is None:
        return [
            _candidate(
                "distribution",
                85,
                _clarify("Ban muon xem phan phoi cua cot numeric nao?"),
            )
        ]
    if not _is_numeric_column(dataframe, column):
        return [
            _candidate(
                "distribution",
                85,
                _clarify(
                    f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de ve phan phoi."
                ),
            )
        ]
    return [
        _candidate(
            "distribution",
            85,
            _tool(
                "generate_chart_spec",
                {
                    "chart_type": "histogram",
                    "x": column,
                    "bins": get_histogram_bins(dataframe, column),
                },
                CONFIDENCE_DISTRIBUTION,
            ),
            evidence=3.0,
        )
    ]


def _outlier_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not has_any_phrase(
        normalized,
        (
            "outlier",
            "ngoai lai",
            "bat thuong",
            "gia tri bat thuong",
            "diem bat thuong",
            "du lieu bat thuong",
        ),
    ):
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(
        dataframe, normalized
    )
    if column is None:
        return [
            _candidate(
                "outlier",
                88,
                _clarify("Ban muon kiem tra outlier cho cot numeric nao?"),
            )
        ]
    if not _is_numeric_column(dataframe, column):
        return [
            _candidate(
                "outlier",
                88,
                _clarify(
                    f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de kiem tra outlier."
                ),
            )
        ]
    return [
        _candidate(
            "outlier",
            88,
            _tool(
                "outlier_detection",
                {"column": column, "limit": 20},
                CONFIDENCE_OUTLIER,
            ),
            evidence=3.0,
        )
    ]


def _describe_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not has_any_phrase(normalized, ("mo ta", "describe", "summary", "thong ke")):
        return []

    column = _find_column(dataframe, normalized)
    if column is not None:
        if _is_numeric_column(dataframe, column):
            return [
                _candidate(
                    "describe",
                    60,
                    _tool(
                        "describe_numeric", {"column": column}, CONFIDENCE_DESCRIBE_COL
                    ),
                    evidence=2.0,
                )
            ]
        return [
            _candidate(
                "describe",
                60,
                _clarify(
                    f"Cot '{column}' khong phai numeric. Hay chon mot cot numeric de mo ta."
                ),
            )
        ]
    if has_any_phrase(normalized, ("numeric", "so", "number")):
        return [
            _candidate(
                "describe", 60, _tool("describe_numeric", {}, CONFIDENCE_DESCRIBE_ALL)
            )
        ]
    return []


def _value_count_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    explicit_frequency_intent = has_any_phrase(
        normalized,
        (
            "value counts",
            "pho bien",
            "tan suat",
            "khac nhau",
            "distinct",
            "unique",
            "gia tri rieng",
            "so luong gia tri",
            "ty le",
            "phan tram",
            "percent",
            "percentage",
            "ratio",
        ),
    )
    if (
        has_any_phrase(normalized, ("top", "cao nhat", "thap nhat"))
        and _find_metric_column(dataframe, normalized) is not None
        and not explicit_frequency_intent
    ):
        return []

    has_value_count_intent = has_any_phrase(
        normalized,
        (
            "top",
            "value counts",
            "dem",
            "pho bien",
            "tan suat",
            "khac nhau",
            "distinct",
            "unique",
            "gia tri rieng",
            "so luong gia tri",
            "ty le",
            "phan tram",
            "percent",
            "percentage",
            "ratio",
        ),
    )
    if not has_value_count_intent:
        return []
    column = _find_column(dataframe, normalized)
    if column is None:
        if has_any_phrase(normalized, ("dem", "count", "so dong")) and has_group_intent(
            normalized
        ):
            column = _find_group_column(dataframe, normalized)
    if column is None:
        return []
    return [
        _candidate(
            "value_counts",
            70,
            _tool(
                "value_counts",
                {"column": column, "top_n": _extract_top_n(normalized)},
                CONFIDENCE_VALUE_COUNTS,
            ),
            evidence=2.0,
        )
    ]


def _sort_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    has_explicit_sort = has_any_phrase(normalized, ("sap xep", "sort", "rank"))
    has_rank_language = has_any_phrase(
        normalized,
        (
            "cao nhat",
            "thap nhat",
            "highest",
            "lowest",
        ),
    )
    if not has_explicit_sort and not has_rank_language:
        return []
    if not has_explicit_sort and detect_aggregation(normalized) is not None:
        return []

    column = _find_metric_column(dataframe, normalized) or _find_column(
        dataframe, normalized
    )
    if column is None:
        return []

    ascending = has_any_phrase(normalized, ("thap nhat", "lowest", "ascending"))
    return [
        _candidate(
            "sort",
            74,
            _tool(
                "sort_values",
                {
                    "column": column,
                    "ascending": ascending,
                    "limit": _extract_top_n(normalized),
                },
                CONFIDENCE_SORT,
            ),
            evidence=2.5,
        )
    ]


def _sql_fallback_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    top_rows = _sql_top_rows_candidate(dataframe, normalized)
    if top_rows is not None:
        return [top_rows]

    group_count = _sql_group_count_candidate(dataframe, normalized)
    if group_count is not None:
        return [group_count]

    multi_filter = _sql_multi_filter_candidate(dataframe, normalized)
    if multi_filter is not None:
        return [multi_filter]

    return []


def _sql_top_rows_candidate(
    dataframe: pd.DataFrame, normalized: str
) -> RouteCandidate | None:
    if not has_any_phrase(
        normalized, ("top", "cao nhat", "highest", "thap nhat", "lowest")
    ):
        return None
    if not has_any_phrase(normalized, ("liet ke", "list", "top")):
        return None

    metric_column = _find_metric_column(dataframe, normalized)
    if metric_column is None:
        return None

    ascending = has_any_phrase(normalized, ("thap nhat", "lowest"))
    select_columns = _sql_select_columns_for_top_request(
        dataframe, normalized, metric_column
    )
    order_direction = "ASC" if ascending else "DESC"
    sql = (
        f"SELECT {', '.join(_quote_identifier(column) for column in select_columns)} "
        f"FROM dataset ORDER BY {_quote_identifier(metric_column)} {order_direction}"
    )
    limit = _extract_top_n(normalized)
    return _candidate(
        "sql_fallback",
        86,
        _tool(
            "query_table_sql",
            {"sql": sql, "limit": limit},
            CONFIDENCE_SQL_FALLBACK,
        ),
        evidence=3.0,
    )


def _sql_group_count_candidate(
    dataframe: pd.DataFrame, normalized: str
) -> RouteCandidate | None:
    if not has_any_phrase(
        normalized, ("dem", "so dong", "row count", "count rows", "number of rows")
    ):
        return None
    if not has_group_intent(normalized):
        return None

    group_column = _find_group_column(dataframe, normalized)
    if group_column is None:
        return None

    sort_direction = "DESC"
    if has_any_phrase(normalized, ("tang dan", "ascending", "asc")):
        sort_direction = "ASC"
    sql = (
        f"SELECT {_quote_identifier(group_column)}, COUNT(*) AS row_count "
        f"FROM dataset GROUP BY {_quote_identifier(group_column)} "
        f"ORDER BY row_count {sort_direction}"
    )
    return _candidate(
        "sql_fallback",
        84,
        _tool(
            "query_table_sql",
            {"sql": sql, "limit": _extract_top_n(normalized)},
            CONFIDENCE_SQL_FALLBACK,
        ),
        evidence=3.0,
    )


def _sql_multi_filter_candidate(
    dataframe: pd.DataFrame, normalized: str
) -> RouteCandidate | None:
    if not has_any_phrase(normalized, ("loc", "filter", "where", "cac dong")):
        return None

    categorical_condition = _extract_categorical_equality_condition(
        dataframe, normalized
    )
    numeric_condition = _extract_numeric_column_condition(dataframe, normalized)
    if categorical_condition is None or numeric_condition is None:
        return None

    cat_column, cat_value = categorical_condition
    num_column, operator, num_value = numeric_condition
    sql_operator = {
        "eq": "=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
    }[operator]
    sql = (
        "SELECT * FROM dataset WHERE "
        f"{_quote_identifier(cat_column)} = {_sql_literal(cat_value)} AND "
        f"{_quote_identifier(num_column)} {sql_operator} {num_value}"
    )
    return _candidate(
        "sql_fallback",
        86,
        _tool(
            "query_table_sql",
            {"sql": sql, "limit": _extract_top_n(normalized)},
            CONFIDENCE_SQL_FALLBACK,
        ),
        evidence=3.5,
    )


def _compare_group_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    if not has_any_phrase(
        normalized,
        ("so sanh", "compare", "comparison", "khac biet", "chenh lech"),
    ):
        return []

    metric_column = _find_metric_column(dataframe, normalized)
    group_column = _find_group_column(
        dataframe, normalized, exclude={metric_column} if metric_column else set()
    )
    if metric_column is None and _has_unmatched_explicit_column_reference(
        dataframe, normalized
    ):
        return [
            _candidate(
                "compare_groups",
                76,
                RouterDecision(
                    route_type="clarify",
                    confidence=0.0,
                    arguments={
                        "intent": "compare_groups",
                        "option_type": "numeric",
                        "group_by": group_column,
                    },
                    message=(
                        "Mình không tìm thấy metric numeric bạn nêu trong dataset. "
                        "Hãy chọn một cột numeric hiện có để so sánh."
                    ),
                ),
            )
        ]
    if metric_column and group_column:
        return [
            _candidate(
                "compare_groups",
                76,
                _tool(
                    "compare_groups",
                    {
                        "metric_column": metric_column,
                        "group_by": group_column,
                        "operation": detect_aggregation(normalized) or "mean",
                    },
                    CONFIDENCE_COMPARE_GROUPS,
                ),
                evidence=3.0,
            )
        ]
    return [
        _candidate(
            "compare_groups",
            76,
            RouterDecision(
                route_type="clarify",
                confidence=0.0,
                arguments={"intent": "compare_groups", "option_type": "numeric"},
                message="Bạn muốn so sánh metric numeric nào theo cột nhóm nào?",
            ),
        )
    ]


def _aggregate_candidates(
    dataframe: pd.DataFrame, normalized: str
) -> list[RouteCandidate]:
    operation = detect_aggregation(normalized)
    if operation is None:
        return []

    metric_column = _find_metric_column(dataframe, normalized)
    has_group_ = has_group_intent(normalized)
    group_column = (
        _find_group_column(
            dataframe, normalized, exclude={metric_column} if metric_column else set()
        )
        if has_group_
        else None
    )

    if metric_column and group_column:
        return [
            _candidate(
                "aggregate",
                75,
                _tool(
                    "aggregate_metric",
                    {
                        "metric_column": metric_column,
                        "group_by": group_column,
                        "operation": operation,
                    },
                    CONFIDENCE_AGGREGATE,
                ),
                evidence=3.0,
            )
        ]
    if metric_column and has_group_:
        return [
            _candidate(
                "aggregate",
                75,
                RouterDecision(
                    route_type="clarify",
                    confidence=0.0,
                    arguments={
                        "intent": "aggregate_metric",
                        "option_type": "categorical",
                        "metric_column": metric_column,
                        "operation": operation,
                    },
                    message=(
                        f"Mình đã nhận metric `{metric_column}` nhưng chưa xác định được cột nhóm. "
                        "Hãy chọn một cột phân nhóm hiện có."
                    ),
                ),
            )
        ]
    if group_column and has_group_:
        return [
            _candidate(
                "aggregate",
                75,
                RouterDecision(
                    route_type="clarify",
                    confidence=0.0,
                    arguments={
                        "intent": "aggregate_metric",
                        "option_type": "numeric",
                        "group_by": group_column,
                        "operation": operation,
                    },
                    message=(
                        f"Mình đã nhận cột nhóm `{group_column}` nhưng chưa xác định được metric numeric. "
                        "Hãy chọn một cột numeric (cot so) để tính."
                    ),
                ),
            )
        ]
    if metric_column and not has_group_:
        return [
            _candidate(
                "aggregate",
                75,
                _tool(
                    "describe_numeric",
                    {"column": metric_column},
                    CONFIDENCE_AGGREGATE_DESCRIBE,
                ),
                evidence=2.0,
            )
        ]
    if has_group_:
        return [
            _candidate(
                "aggregate",
                75,
                _clarify("Ban muon tinh metric nao va nhom theo cot nao?"),
            )
        ]
    return []


def _chart_candidates(dataframe: pd.DataFrame, normalized: str) -> list[RouteCandidate]:
    if not _has_chart_intent(normalized):
        return []

    chart_type = detect_chart_type(normalized)
    metric_column = _find_metric_column(dataframe, normalized)
    group_column = _find_group_column(
        dataframe, normalized, exclude={metric_column} if metric_column else set()
    )

    if chart_type == "histogram":
        column = metric_column or _find_column(dataframe, normalized)
        if column and _is_numeric_column(dataframe, column):
            return [
                _candidate(
                    "chart",
                    80,
                    _tool(
                        "generate_chart_spec",
                        {
                            "chart_type": "histogram",
                            "x": column,
                            "bins": get_histogram_bins(dataframe, column),
                        },
                        CONFIDENCE_CHART_HISTOGRAM,
                    ),
                    evidence=2.0,
                )
            ]

    if chart_type == "correlation_heatmap":
        return [
            _candidate(
                "chart",
                80,
                _tool(
                    "generate_chart_spec",
                    {"chart_type": "correlation_heatmap"},
                    CONFIDENCE_CHART_HEATMAP,
                ),
                evidence=2.0,
            )
        ]

    if chart_type == "scatter":
        numeric_columns = [
            column
            for column in _matching_columns(dataframe, normalized)
            if _is_numeric_column(dataframe, column)
        ]
        if len(numeric_columns) >= 2:
            return [
                _candidate(
                    "chart",
                    80,
                    _tool(
                        "generate_chart_spec",
                        {
                            "chart_type": "scatter",
                            "x": numeric_columns[0],
                            "y": numeric_columns[1],
                        },
                        CONFIDENCE_CHART_SCATTER,
                    ),
                    evidence=3.0,
                )
            ]

    if chart_type == "pie":
        category_column = group_column or _find_column(dataframe, normalized)
        if category_column and not _is_numeric_column(dataframe, category_column):
            return [
                _candidate(
                    "chart",
                    80,
                    _tool(
                        "generate_chart_spec",
                        {"chart_type": "pie", "names": category_column},
                        CONFIDENCE_CHART_GENERIC,
                    ),
                    evidence=2.0,
                )
            ]

    if metric_column and group_column:
        return [
            _candidate(
                "chart",
                80,
                _tool(
                    "generate_chart_spec",
                    {"chart_type": chart_type, "x": group_column, "y": metric_column},
                    CONFIDENCE_CHART_GENERIC,
                ),
                evidence=2.0,
            )
        ]
    return [
        _candidate(
            "chart", 80, _clarify("Ban muon ve bieu do cho metric nao va theo cot nao?")
        )
    ]


ROUTE_BUILDERS: tuple[RouteBuilder, ...] = (
    lambda dataframe, normalized: _profile_candidates(normalized),
    _percentage_candidates,
    _correlation_candidates,
    lambda dataframe, normalized: _missing_candidates(normalized),
    lambda dataframe, normalized: _data_quality_candidates(normalized),
    lambda dataframe, normalized: _list_column_candidates(normalized),
    _distribution_candidates,
    _outlier_candidates,
    _describe_candidates,
    _sql_fallback_candidates,
    _value_count_candidates,
    _sort_candidates,
    _compare_group_candidates,
    _aggregate_candidates,
    _chart_candidates,
)


def _filter_compatible_candidates(
    candidates: list[RouteCandidate], normalized: str
) -> list[RouteCandidate]:
    intents = {candidate.intent for candidate in candidates}
    filtered = list(candidates)

    if "correlation" in intents:
        filtered = [
            candidate for candidate in filtered if candidate.intent != "list_columns"
        ]
        if _is_chart_request(normalized):
            filtered = [
                candidate for candidate in filtered if candidate.intent != "correlation"
            ]
    if "missing" in intents:
        filtered = [
            candidate for candidate in filtered if candidate.intent != "list_columns"
        ]
    if "data_quality" in intents:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent not in {"list_columns", "missing"}
        ]
    if "distribution" in intents:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent not in {"chart", "describe"}
        ]
    if "outlier" in intents:
        filtered = [
            candidate for candidate in filtered if candidate.intent != "describe"
        ]
        if "aggregate" in intents and has_any_phrase(
            normalized, ("nhom", "theo", "trung binh", "average", "mean")
        ):
            filtered = [
                candidate for candidate in filtered if candidate.intent != "outlier"
            ]
    if "percentage" in intents:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent not in {"aggregate", "describe"}
        ]
    if "compare_groups" in intents:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent not in {"aggregate", "describe"}
        ]
    if "sort" in intents and has_any_phrase(normalized, ("sap xep", "sort", "rank")):
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent not in {"aggregate", "describe"}
        ]
    if "sql_fallback" in intents:
        filtered = [
            candidate
            for candidate in filtered
            if candidate.intent
            not in {"profile", "value_counts", "sort", "aggregate", "describe"}
        ]

    return filtered or candidates


def _has_candidate_conflict(best: RouteCandidate, second: RouteCandidate) -> bool:
    if best.intent == second.intent:
        return False
    return best.score - second.score < 12


def _tool(
    tool_name: str, arguments: dict[str, Any], confidence: float
) -> RouterDecision:
    return RouterDecision(
        route_type="tool",
        confidence=confidence,
        tool_name=tool_name,
        arguments=arguments,
    )


def _fallback(message: str) -> RouterDecision:
    return RouterDecision(route_type="fallback", confidence=0.0, message=message)


def _clarify(message: str) -> RouterDecision:
    return RouterDecision(route_type="clarify", confidence=0.0, message=message)


def _has_chart_intent(normalized: str) -> bool:
    return has_any_phrase(
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


def _has_correlation_intent(normalized: str) -> bool:
    return has_any_phrase(
        normalized, ("tuong quan", "correlation", "lien quan", "related")
    )


def _is_chart_request(normalized: str) -> bool:
    return has_any_phrase(
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


def _extract_numeric_condition(normalized: str) -> tuple[str, float] | None:
    patterns = (
        (r"\bgte\s+(-?\d+(?:\.\d+)?)", "gte"),
        (r"\blte\s+(-?\d+(?:\.\d+)?)", "lte"),
        (r"\bgt\s+(-?\d+(?:\.\d+)?)", "gt"),
        (r"\blt\s+(-?\d+(?:\.\d+)?)", "lt"),
        (r"\beq\s+(-?\d+(?:\.\d+)?)", "eq"),
        (r"(?:>=)\s*(-?\d+(?:\.\d+)?)", "gte"),
        (r"(?:<=)\s*(-?\d+(?:\.\d+)?)", "lte"),
        (r"(?:>)\s*(-?\d+(?:\.\d+)?)", "gt"),
        (r"(?:<)\s*(-?\d+(?:\.\d+)?)", "lt"),
        (r"(?:=|==)\s*(-?\d+(?:\.\d+)?)", "eq"),
        (r"(duoi|nho hon|less than|below|under)\s+(-?\d+(?:\.\d+)?)", "lt"),
        (
            r"(toi da|nho hon hoac bang|at most|less than or equal)\s+(-?\d+(?:\.\d+)?)",
            "lte",
        ),
        (r"(tren|lon hon|greater than|above|over)\s+(-?\d+(?:\.\d+)?)", "gt"),
        (
            r"(toi thieu|lon hon hoac bang|at least|greater than or equal)\s+(-?\d+(?:\.\d+)?)",
            "gte",
        ),
        (r"(bang|equal to)\s+(-?\d+(?:\.\d+)?)", "eq"),
    )
    for pattern, operator in patterns:
        match = re.search(pattern, normalized)
        if match:
            value = float(match.group(match.lastindex or 1))
            if value.is_integer():
                return operator, int(value)
            return operator, value
    return None


def _extract_numeric_column_condition(
    dataframe: pd.DataFrame, normalized: str
) -> tuple[str, str, int | float] | None:
    operator_patterns = (
        (r"gte", "gte"),
        (r"lte", "lte"),
        (r"gt", "gt"),
        (r"lt", "lt"),
        (r"eq", "eq"),
        (r">=", "gte"),
        (r"<=", "lte"),
        (r">", "gt"),
        (r"<", "lt"),
        (r"(?:=|==)", "eq"),
        (r"(?:lon hon|greater than|tren)", "gt"),
        (r"(?:nho hon|less than|duoi)", "lt"),
        (r"(?:bang|equal to)", "eq"),
    )
    for column in dataframe.columns:
        column_name = str(column)
        if not _is_numeric_column(dataframe, column_name):
            continue
        normalized_column = _normalize_identifier(column_name)
        if not _contains_normalized_column(normalized, normalized_column):
            continue
        for operator_pattern, operator in operator_patterns:
            pattern = (
                rf"{re.escape(normalized_column)}\s*(?:la|is)?\s*"
                rf"{operator_pattern}\s*(-?\d+(?:\.\d+)?)"
            )
            match = re.search(pattern, normalized)
            if not match:
                continue
            value = float(match.group(1))
            return column_name, operator, int(value) if value.is_integer() else value
    for operator_pattern, operator in operator_patterns:
        match = re.search(
            rf"([a-z0-9 ]{{1,80}}?)\s+{operator_pattern}\s+(-?\d+(?:\.\d+)?)",
            normalized,
        )
        if not match:
            continue
        column_text = match.group(1).strip()
        resolved_column = resolve_column(
            dataframe, column_text, expected_type="numeric"
        )
        if resolved_column is None:
            continue
        value = float(match.group(2))
        return (
            resolved_column,
            operator,
            int(value) if value.is_integer() else value,
        )
    return None


def _extract_categorical_percentage_condition(
    dataframe: pd.DataFrame, column: str, normalized: str
) -> tuple[str, str] | None:
    values = [str(value) for value in dataframe[column].dropna().astype(str).unique()]
    value_lookup = {normalize_text(value): value for value in values}

    for normalized_value, original_value in value_lookup.items():
        if re.search(rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized):
            return "eq", original_value

    yes_value = _find_categorical_value(value_lookup, ("yes", "true", "co", "1"))
    no_value = _find_categorical_value(value_lookup, ("no", "false", "khong", "0"))
    if no_value is not None and any(
        token in normalized for token in ("khong tham gia", "khong co", "khong")
    ):
        return "eq", no_value
    if yes_value is not None and any(
        token in normalized for token in ("tham gia", "co", "yes")
    ):
        return "eq", yes_value
    return None


def _find_categorical_value(
    value_lookup: dict[str, str], candidates: tuple[str, ...]
) -> str | None:
    for candidate in candidates:
        if candidate in value_lookup:
            return value_lookup[candidate]
    return None


def _extract_categorical_equality_condition(
    dataframe: pd.DataFrame, normalized: str
) -> tuple[str, str] | None:
    for column in dataframe.columns:
        column_name = str(column)
        if _is_numeric_column(dataframe, column_name):
            continue
        normalized_column = _normalize_identifier(column_name)
        if not _column_is_referenced(dataframe, normalized, column_name, "categorical"):
            continue

        values = [str(value) for value in dataframe[column_name].dropna().unique()]
        for value in values:
            normalized_value = normalize_text(value)
            if not normalized_value:
                continue
            pattern = (
                rf"{re.escape(normalized_column)}\s*(?:la|=|==|is)?\s*"
                rf"{re.escape(normalized_value)}"
            )
            if re.search(pattern, normalized) or (
                not _contains_normalized_column(normalized, normalized_column)
                and re.search(
                    rf"(?<!\w){re.escape(normalized_value)}(?!\w)", normalized
                )
            ):
                return column_name, value
    return None


def _column_is_referenced(
    dataframe: pd.DataFrame,
    normalized: str,
    column: str,
    expected_type: ExpectedType | None,
) -> bool:
    normalized_column = _normalize_identifier(column)
    if _contains_normalized_column(normalized, normalized_column):
        return True
    return resolve_column(dataframe, normalized, expected_type=expected_type) == column


def _sql_select_columns_for_top_request(
    dataframe: pd.DataFrame, normalized: str, metric_column: str
) -> list[str]:
    selected = []
    if has_any_phrase(normalized, ("user", "khach hang", "customer")):
        id_column = _find_id_like_column(dataframe)
        if id_column is not None:
            selected.append(id_column)
    selected.append(metric_column)
    return _dedupe_columns(selected)


def _find_id_like_column(dataframe: pd.DataFrame) -> str | None:
    for column in dataframe.columns:
        normalized_column = _normalize_identifier(str(column))
        if normalized_column == "id" or normalized_column.endswith(" id"):
            return str(column)
    return None


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _dedupe_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        result.append(column)
    return result


def _expand_comparison_symbols(text: str) -> str:
    return (
        text.replace(">=", " gte ")
        .replace("<=", " lte ")
        .replace("==", " eq ")
        .replace(">", " gt ")
        .replace("<", " lt ")
        .replace("=", " eq ")
    )


def _find_metric_column(dataframe: pd.DataFrame, normalized: str) -> str | None:
    for column in _matching_columns(dataframe, normalized):
        if _is_numeric_column(dataframe, column):
            return column
    if _has_unmatched_explicit_column_reference(dataframe, normalized):
        return None
    resolved = resolve_column(dataframe, normalized, expected_type="numeric")
    if resolved is not None:
        return resolved
    numeric_columns = [
        str(column)
        for column in dataframe.columns
        if _is_numeric_column(dataframe, str(column))
    ]
    return numeric_columns[0] if len(numeric_columns) == 1 else None


def _find_group_column(
    dataframe: pd.DataFrame, normalized: str, exclude: set[str] | None = None
) -> str | None:
    exclude = exclude or set()
    for column in _matching_columns(dataframe, normalized):
        if column in exclude:
            continue
        if not _is_numeric_column(dataframe, column):
            return column
    if _has_unmatched_explicit_group_reference(dataframe, normalized):
        return None
    categorical_columns = [
        str(column)
        for column in dataframe.columns
        if not _is_numeric_column(dataframe, str(column))
    ]
    candidates = [
        column
        for column in categorical_columns
        if column not in exclude and _is_group_candidate(dataframe, column)
    ]
    return candidates[0] if len(candidates) == 1 else None


def _is_group_candidate(dataframe: pd.DataFrame, column: str) -> bool:
    normalized_column = normalize_identifier(column)
    if normalized_column == "id" or normalized_column.endswith("_id"):
        return False
    if normalized_column in {"note", "notes", "comment", "comments", "description"}:
        return False
    non_null_count = int(dataframe[column].notna().sum())
    if non_null_count <= 0:
        return False
    unique_count = int(dataframe[column].dropna().nunique())
    unique_ratio = unique_count / non_null_count
    return unique_count <= 20 and unique_ratio < 0.95


def _find_correlation_columns(dataframe: pd.DataFrame, normalized: str) -> list[str]:
    columns = [
        column
        for column in _matching_columns(dataframe, normalized)
        if _is_numeric_column(dataframe, column)
    ]
    resolved = resolve_column(dataframe, normalized, expected_type="numeric")
    if resolved is not None and resolved not in columns:
        columns.append(resolved)
    for phrase in _correlation_column_phrases(normalized):
        resolved_phrase = resolve_column(dataframe, phrase, expected_type="numeric")
        if resolved_phrase is not None and resolved_phrase not in columns:
            columns.append(resolved_phrase)

    numeric_columns = [
        str(column)
        for column in dataframe.columns
        if _is_numeric_column(dataframe, str(column))
    ]
    if len(columns) >= 2:
        return columns
    if len(columns) == 1 and _asks_correlation_columns(normalized):
        return columns + [column for column in numeric_columns if column not in columns]
    if not columns and len(numeric_columns) == 2:
        return numeric_columns
    return columns


def _asks_correlation_columns(normalized: str) -> bool:
    return has_any_phrase(
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
            phrases.extend(
                part.strip() for part in f" {normalized} ".split(marker) if part.strip()
            )
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


def _has_unmatched_explicit_column_reference(
    dataframe: pd.DataFrame, normalized: str
) -> bool:
    normalized_columns = {
        _normalize_identifier(str(column)) for column in dataframe.columns
    }
    explicit_tokens = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)+\b", normalized
    )
    if any(token not in normalized_columns for token in explicit_tokens):
        return True

    numeric_column_tokens = set()
    for column in dataframe.columns:
        column_name = str(column)
        if _is_numeric_column(dataframe, column_name):
            numeric_column_tokens.update(_normalize_identifier(column_name).split())

    return any(
        token in EXPLICIT_NUMERIC_REFERENCE_TOKENS
        and token not in numeric_column_tokens
        for token in normalized.split()
    )


def _has_unmatched_explicit_group_reference(
    dataframe: pd.DataFrame, normalized: str
) -> bool:
    group_column_tokens = set()
    for column in dataframe.columns:
        column_name = str(column)
        if not _is_numeric_column(dataframe, column_name):
            group_column_tokens.update(_normalize_identifier(column_name).split())

    return any(
        token in EXPLICIT_GROUP_REFERENCE_TOKENS and token not in group_column_tokens
        for token in normalized.split()
    )


def _extract_top_n(normalized: str) -> int:
    match = re.search(r"\btop\s+(\d{1,2})\b", normalized)
    if not match:
        return 10
    return max(1, min(50, int(match.group(1))))


def _is_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    return is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])


def _normalize_identifier(identifier: str) -> str:
    return normalize_identifier(identifier)


def _contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    return contains_normalized_column(normalized_text, normalized_column)
