import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.gemini_runtime import LLMProvider
from backend.services.profiling import profile_dataset


MAX_QUESTIONS = 6
MAX_INSIGHTS = 5
MIN_INSIGHTS = 1
MIN_CORRELATION_TO_HIGHLIGHT = 0.5
MIN_TOP_CATEGORY_PERCENT = 30.0
MIN_MISSING_PERCENT_TO_HIGHLIGHT = 5.0
MIN_GROUP_ETA_SQUARED = 0.05
MIN_OUTLIER_IQR_SEVERITY = 2.0

SPECULATIVE_WORDING = (
    "có thể do",
    "co the do",
    "nguyên nhân",
    "nguyen nhan",
    "có lẽ",
    "co le",
    "dường như",
    "duong nhu",
    "nhiều khả năng",
    "nhieu kha nang",
)
CAUSAL_WORDING = (
    "gây ra",
    "gay ra",
    "dẫn đến",
    "dan den",
    "khiến",
    "khien",
)
GENERIC_WORDING = (
    "phần lớn",
    "phan lon",
    "hầu hết",
    "hau het",
    "đa số",
    "da so",
    "chiếm đa số",
    "chiem da so",
    "một số cột",
    "mot so cot",
    "nhiều nhóm khác nhau",
    "nhieu nhom khac nhau",
)
AWKWARD_ANALYST_WORDING = (
    "bị thiếu giá trị cho cột",
    "bi thieu gia tri cho cot",
    "có một giá trị ngoại lệ là",
    "co mot gia tri ngoai le la",
    "một giá trị ngoại lệ là",
    "mot gia tri ngoai le la",
)
QUESTION_SCHEMA_WORDING = (
    "dataset",
    "dữ liệu",
    "du lieu",
    "cột nào",
    "cot nao",
    "bao nhiêu dòng",
    "bao nhieu dong",
    "bao nhiêu cột",
    "bao nhieu cot",
    "thiếu",
    "thieu",
    "missing",
)


class SuggestedContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    source: str = "fallback"


def generate_suggested_content(
    dataframe: pd.DataFrame, provider: LLMProvider | None = None
) -> SuggestedContent:
    profile = profile_dataset(dataframe)
    signals = _build_profiling_signals(profile, dataframe)
    fallback = _fallback_suggested_content(profile, signals)

    if provider is None:
        return fallback

    try:
        raw_response = provider.generate(_build_suggestions_prompt(profile, signals))
        suggested = _parse_suggested_content(raw_response)
    except Exception:
        return fallback

    validated_questions = _validate_questions(suggested.questions, profile)

    if not validated_questions:
        return fallback
    questions = _validate_questions(
        _dedupe(validated_questions + fallback.questions), profile
    )

    return SuggestedContent(
        questions=questions[:MAX_QUESTIONS],
        insights=fallback.insights,
        source="llm",
    )


def _build_suggestions_prompt(
    profile: dict[str, Any], signals: dict[str, Any] | None = None
) -> str:
    profiling_signals = signals or _build_profiling_signals(profile)
    safe_profile = {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "missing_values": profile["missing_values"],
        "numeric_summary": profile["numeric_summary"],
        "top_categories": profile["top_categories"],
    }
    contract = {
        "questions": [
            "Vietnamese question grounded in existing columns only",
        ],
        "insights": [
            "Vietnamese insight telling a business story based on the data (min 3, max 5)",
        ],
    }
    return (
        "Bạn là data analyst assistant cho dữ liệu dạng bảng.\n"
        "Nhiệm vụ: viết suggested questions bằng tiếng Việt, dựa hoàn toàn trên PROFILE "
        "và PROFILING_SIGNALS bên dưới.\n\n"
        "Luật cho questions:\n"
        "- Trả tối đa 6 câu hỏi, đa dạng theo missing value, aggregate, comparison, distribution, top category, correlation.\n"
        "- Chỉ hỏi correlation nếu PROFILING_SIGNALS.correlation_candidates có item với |r| >= 0.50.\n"
        "- Chỉ nhắc tới cột thật trong PROFILE.column_names.\n"
        "- Không tạo câu hỏi vô nghĩa hoặc cần business context không có trong dữ liệu.\n\n"
        "Luật cho insights:\n"
        "- Không tự tạo insights ngoài các tín hiệu trong PROFILE và PROFILING_SIGNALS; hệ thống sẽ ưu tiên insight deterministic nếu phản hồi không đạt chuẩn.\n"
        "- Viết 3 đến 5 câu insights cực kỳ sắc bén bằng tiếng Việt dựa trên PROFILE và PROFILING_SIGNALS.\n"
        "- Khác với thống kê máy móc, mỗi câu insight phải kết hợp nhiều con số để kể một câu chuyện phân tích (ví dụ: thay vì nói 'Missing 687 Cabin', hãy phân tích 'Hơn 77% dữ liệu Cabin bị thiếu, cho thấy việc ghi nhận vị trí buồng phòng rất kém').\n"
        "- KHÔNG liệt kê các con số khô khan, hãy phân tích ý nghĩa thực tiễn của chúng.\n\n"
        "Chỉ trả về JSON object hợp lệ theo CONTRACT, không markdown.\n\n"
        f"PROFILE={json.dumps(safe_profile, ensure_ascii=False)}\n"
        f"PROFILING_SIGNALS={json.dumps(profiling_signals, ensure_ascii=False)}\n"
        f"CONTRACT={json.dumps(contract, ensure_ascii=False)}"
    )


def _parse_suggested_content(raw_response: str) -> SuggestedContent:
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Response does not contain JSON.")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Response is not valid JSON.") from exc

    try:
        return SuggestedContent.model_validate(payload)
    except ValidationError as exc:
        raise ValueError("Response does not match suggested content schema.") from exc


def _build_profiling_signals(
    profile: dict[str, Any], dataframe: pd.DataFrame | None = None
) -> dict[str, Any]:
    missing_values = sorted(
        profile.get("missing_values", []),
        key=lambda item: int(item.get("missing_count", 0)),
        reverse=True,
    )
    numeric_summary = profile.get("numeric_summary", [])
    top_categories = profile.get("top_categories", [])

    return {
        "dataset_shape": {
            "rows": profile.get("rows", 0),
            "columns": profile.get("columns", 0),
        },
        "high_missing_columns": [
            {
                "column": item["name"],
                "missing_count": item["missing_count"],
                "missing_percent": item["missing_percent"],
            }
            for item in missing_values[:5]
        ],
        "numeric_mean_median": _numeric_mean_median_signals(numeric_summary),
        "numeric_ranges": _numeric_range_signals(numeric_summary),
        "top_categories_with_percentage": _top_category_signals(top_categories),
        "correlation_candidates": _correlation_candidates(dataframe)
        if dataframe is not None
        else [],
        "possible_outliers": _possible_outliers(numeric_summary),
        "group_difference_candidates": _group_difference_candidates(dataframe)
        if dataframe is not None
        else [],
    }


def _fallback_suggested_content(
    profile: dict[str, Any], signals: dict[str, Any] | None = None
) -> SuggestedContent:
    profiling_signals = signals or _build_profiling_signals(profile)
    questions = _fallback_questions(profile, profiling_signals)
    insights = _fallback_insights(profile, profiling_signals)
    return SuggestedContent(
        questions=questions[:MAX_QUESTIONS],
        insights=insights[:MAX_INSIGHTS],
        source="fallback",
    )


def _fallback_questions(profile: dict[str, Any], signals: dict[str, Any]) -> list[str]:
    numeric_columns = [item["column"] for item in profile.get("numeric_summary", [])]
    categorical_columns = [item["column"] for item in profile.get("top_categories", [])]
    missing_columns = [
        item["column"] for item in signals.get("high_missing_columns", [])
    ]
    correlation_candidates = signals.get("correlation_candidates", [])

    questions: list[str] = []

    if missing_columns:
        questions.append(f"Cột {missing_columns[0]} đang thiếu bao nhiêu giá trị?")
    else:
        questions.append("Có cột nào thiếu dữ liệu không?")

    if numeric_columns and categorical_columns:
        metric = numeric_columns[0]
        group = categorical_columns[0]
        questions.append(f"Trung bình {metric} theo {group} là bao nhiêu?")
        questions.append(f"Nhóm {group} nào có {metric} trung bình cao nhất?")
    elif numeric_columns:
        questions.append(
            f"Giá trị trung bình, min và max của {numeric_columns[0]} là bao nhiêu?"
        )

    if numeric_columns:
        questions.append(f"Phân phối của {numeric_columns[0]} trông như thế nào?")

    if categorical_columns:
        questions.append(
            f"Giá trị nào xuất hiện nhiều nhất trong cột {categorical_columns[0]}?"
        )

    if correlation_candidates:
        candidate = correlation_candidates[0]
        questions.append(
            f"{candidate['column_a']} có tương quan với {candidate['column_b']} không?"
        )

    questions.append("Dataset có bao nhiêu dòng và bao nhiêu cột?")

    return _validate_questions(_dedupe(questions), profile)


def _fallback_insights(
    profile: dict[str, Any], signals: dict[str, Any] | None = None
) -> list[str]:
    profiling_signals = signals or _build_profiling_signals(profile)
    insights: list[str] = []

    for item in profiling_signals.get("group_difference_candidates", [])[:1]:
        insights.append(
            f"{item['metric_column']} khác biệt rõ theo {item['group_by']}: "
            f"{item['top_group']} mean={_format_metric(item['top_mean'])} cao hơn "
            f"{item['bottom_group']} mean={_format_metric(item['bottom_mean'])} "
            f"khoảng {_format_metric(item['mean_gap'])} (eta²={_format_metric(item['eta_squared'])}). "
            "Đây là chart so sánh nhóm đáng ưu tiên."
        )

    for item in profiling_signals.get("correlation_candidates", [])[:1]:
        insights.append(
            f"{item['column_a']} vs {item['column_b']} có tương quan {item['direction']} "
            f"mức {item['strength']} (r={_format_metric(item['correlation'])}); "
            "nên xem scatter để kiểm tra xu hướng giữa hai biến."
        )

    for item in profiling_signals.get("possible_outliers", [])[:1]:
        side_label = "max" if item["side"] == "max" else "min"
        direction_label = "cao" if item["side"] == "max" else "thấp"
        insights.append(
            f"{item['column']} có {side_label}={_format_metric(item['value'])} "
            f"{direction_label} bất thường so với ngưỡng IQR {_format_metric(item['threshold'])}; "
            "nên kiểm tra điểm cực trị trước khi diễn giải mean."
        )

    missing_columns = profiling_signals.get("high_missing_columns", [])
    significant_missing = [
        item
        for item in missing_columns
        if float(item.get("missing_percent", 0.0)) >= MIN_MISSING_PERCENT_TO_HIGHLIGHT
    ]
    if significant_missing:
        item = significant_missing[0]
        insights.append(
            f"{item['column']} thiếu {_format_count(item['missing_count'])} dòng "
            f"(~{_format_metric(item['missing_percent'])}%); cần xử lý missing trước khi dùng cột này để so sánh nhóm."
        )

    validated = _validate_insights(_dedupe(insights), profile)
    if len(validated) >= MIN_INSIGHTS:
        return validated[:MAX_INSIGHTS]
    return validated


def _validate_questions(questions: list[str], profile: dict[str, Any]) -> list[str]:
    column_names = {str(column) for column in profile["column_names"]}
    validated = []
    for question in questions:
        if not isinstance(question, str) or not question.strip():
            continue
        cleaned = question.strip()
        if len(cleaned) < 12:
            continue
        if _mentions_unknown_structured_column(cleaned, column_names):
            continue
        if not _is_grounded_question(cleaned, column_names):
            continue
        validated.append(cleaned)
    return _dedupe(validated)


def _validate_insights(
    insights: list[str], profile: dict[str, Any] | None = None
) -> list[str]:
    column_names = (
        {str(column) for column in profile.get("column_names", [])}
        if profile
        else set()
    )
    validated: list[str] = []
    for insight in insights:
        if not isinstance(insight, str):
            continue
        cleaned = insight.strip()
        if len(cleaned) < 24:
            continue
        if not re.search(r"\d", cleaned):
            continue
        lowered = cleaned.lower()
        if any(phrase in lowered for phrase in SPECULATIVE_WORDING + CAUSAL_WORDING):
            continue
        if any(phrase in lowered for phrase in AWKWARD_ANALYST_WORDING):
            continue
        if _starts_with_percent_without_count(lowered):
            continue
        if _contains_generic_without_metric(lowered):
            continue
        if column_names and _mentions_unknown_structured_column(cleaned, column_names):
            continue
        validated.append(cleaned)
    return _dedupe(validated)


def _contains_generic_without_metric(lowered_text: str) -> bool:
    if not any(phrase in lowered_text for phrase in GENERIC_WORDING):
        return False
    has_metric = bool(re.search(r"\d", lowered_text))
    has_percent = (
        "%" in lowered_text
        or "phần trăm" in lowered_text
        or "phan tram" in lowered_text
    )
    has_count = any(
        word in lowered_text for word in ("dòng", "dong", "giá trị", "gia tri", "count")
    )
    return not (has_metric and (has_percent or has_count))


def _starts_with_percent_without_count(lowered_text: str) -> bool:
    if not re.match(r"^\s*\d+([.,]\d+)?\s*%", lowered_text):
        return False
    count_words = ("dòng", "dong", "bản ghi", "ban ghi", "giá trị", "gia tri", "count")
    return not any(word in lowered_text for word in count_words)


def _is_grounded_question(question: str, column_names: set[str]) -> bool:
    lowered = question.lower()
    if any(_contains_column(lowered, column) for column in column_names):
        return True
    return any(phrase in lowered for phrase in QUESTION_SCHEMA_WORDING)


def _mentions_unknown_structured_column(text: str, column_names: set[str]) -> bool:
    known_lower = {column.lower() for column in column_names}
    tokens = set(text.replace("`", " ").replace(",", " ").split())
    for token in tokens:
        cleaned = token.strip(" .?!:;()[]{}").lower()
        structured_name = cleaned.split("=", 1)[0].strip("'\"")
        if "_" in structured_name and structured_name not in known_lower:
            return True
    return False


def _numeric_mean_median_signals(
    numeric_summary: list[dict[str, Any]],
) -> list[dict[str, object]]:
    items = [
        _numeric_signal(item) for item in numeric_summary if _has_numeric_range(item)
    ]
    return [
        {
            "column": item["column"],
            "count": item["count"],
            "mean": item["mean"],
            "median": item["median"],
        }
        for item in sorted(items, key=_numeric_salience, reverse=True)[:8]
    ]


def _numeric_range_signals(
    numeric_summary: list[dict[str, Any]],
) -> list[dict[str, object]]:
    items = [
        _numeric_signal(item) for item in numeric_summary if _has_numeric_range(item)
    ]
    return [
        {
            "column": item["column"],
            "min": item["min"],
            "max": item["max"],
            "std": item["std"],
            "range_width": item["range_width"],
        }
        for item in sorted(items, key=_numeric_salience, reverse=True)[:8]
    ]


def _numeric_signal(item: dict[str, Any]) -> dict[str, object]:
    min_value = float(item["min"])
    max_value = float(item["max"])
    return {
        "column": item["column"],
        "count": item["count"],
        "mean": item["mean"],
        "median": item["median"],
        "min": item["min"],
        "max": item["max"],
        "std": item["std"],
        "range_width": round(max_value - min_value, 4),
    }


def _has_numeric_range(item: dict[str, Any]) -> bool:
    if item.get("count", 0) <= 0:
        return False
    if item.get("min") is None or item.get("max") is None:
        return False
    return float(item["max"]) != float(item["min"])


def _numeric_salience(item: dict[str, Any]) -> float:
    mean = item.get("mean")
    std = item.get("std")
    range_width = float(item.get("range_width", 0.0))
    if std is not None and mean not in (None, 0):
        return abs(float(std) / float(mean))
    return range_width


def _top_category_signals(
    top_categories: list[dict[str, Any]],
) -> list[dict[str, object]]:
    signals = []
    for item in top_categories:
        if not item.get("values"):
            continue
        top_value = item["values"][0]
        percentage = float(top_value["percent"])
        if percentage < MIN_TOP_CATEGORY_PERCENT:
            continue
        signals.append(
            {
                "column": item["column"],
                "top_value": top_value["value"],
                "count": top_value["count"],
                "percentage": top_value["percent"],
                "rank": 1,
            }
        )
    return sorted(signals, key=lambda item: float(item["percentage"]), reverse=True)[:8]


def _correlation_candidates(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    numeric_frame = dataframe.select_dtypes(include="number")
    if numeric_frame.shape[1] < 2:
        return []

    matrix = numeric_frame.corr(numeric_only=True)
    candidates: list[dict[str, object]] = []
    columns = list(matrix.columns)
    for index, column_a in enumerate(columns):
        for column_b in columns[index + 1 :]:
            coefficient = matrix.loc[column_a, column_b]
            if pd.isna(coefficient):
                continue
            coefficient = round(float(coefficient), 4)
            if abs(coefficient) < MIN_CORRELATION_TO_HIGHLIGHT:
                continue
            candidates.append(
                {
                    "column_a": str(column_a),
                    "column_b": str(column_b),
                    "correlation": coefficient,
                    "abs_correlation": round(abs(coefficient), 4),
                    "direction": "dương" if coefficient >= 0 else "âm",
                    "strength": _correlation_strength(coefficient),
                }
            )

    return sorted(
        candidates, key=lambda item: float(item["abs_correlation"]), reverse=True
    )[:5]


def _group_difference_candidates(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    numeric_columns = [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not _is_id_like_column(str(column))
    ]
    categorical_columns = [
        str(column)
        for column in dataframe.columns
        if str(column) not in numeric_columns and not _is_id_like_column(str(column))
    ]

    candidates: list[dict[str, object]] = []
    for group_by in categorical_columns[:8]:
        unique_count = int(dataframe[group_by].nunique(dropna=True))
        if unique_count < 2 or unique_count > 15:
            continue
        for metric_column in numeric_columns[:8]:
            df_clean = dataframe[[group_by, metric_column]].dropna()
            if df_clean.empty:
                continue
            group_stats = df_clean.groupby(group_by)[metric_column].agg(
                ["count", "mean"]
            )
            if group_stats.shape[0] < 2:
                continue
            min_group_count = max(2, int(len(df_clean) * 0.02))
            if int(group_stats["count"].min()) < min_group_count:
                continue
            grand_mean = float(df_clean[metric_column].mean())
            ss_total = float(((df_clean[metric_column] - grand_mean) ** 2).sum())
            if ss_total <= 0:
                continue
            ss_between = float(
                (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
            )
            eta_squared = round(ss_between / ss_total, 4)
            if eta_squared < MIN_GROUP_ETA_SQUARED:
                continue

            ordered = group_stats.sort_values("mean", ascending=False)
            top_group = ordered.index[0]
            bottom_group = ordered.index[-1]
            top_mean = float(ordered.iloc[0]["mean"])
            bottom_mean = float(ordered.iloc[-1]["mean"])
            candidates.append(
                {
                    "metric_column": metric_column,
                    "group_by": group_by,
                    "top_group": str(top_group),
                    "bottom_group": str(bottom_group),
                    "top_mean": round(top_mean, 4),
                    "bottom_mean": round(bottom_mean, 4),
                    "mean_gap": round(abs(top_mean - bottom_mean), 4),
                    "eta_squared": eta_squared,
                }
            )

    return sorted(
        candidates, key=lambda item: float(item["eta_squared"]), reverse=True
    )[:5]


def _possible_outliers(
    numeric_summary: list[dict[str, Any]],
) -> list[dict[str, object]]:
    outliers: list[dict[str, object]] = []
    for item in numeric_summary:
        p25 = item.get("p25")
        p75 = item.get("p75")
        min_value = item.get("min")
        max_value = item.get("max")
        if p25 is None or p75 is None or min_value is None or max_value is None:
            continue
        iqr = float(p75) - float(p25)
        if iqr <= 0:
            continue
        lower_threshold = float(p25) - 1.5 * iqr
        upper_threshold = float(p75) + 1.5 * iqr
        if float(min_value) < lower_threshold:
            severity = (lower_threshold - float(min_value)) / iqr
            if severity < MIN_OUTLIER_IQR_SEVERITY:
                continue
            outliers.append(
                {
                    "column": item["column"],
                    "side": "min",
                    "value": min_value,
                    "threshold": round(lower_threshold, 4),
                    "severity": round(severity, 4),
                }
            )
        if float(max_value) > upper_threshold:
            severity = (float(max_value) - upper_threshold) / iqr
            if severity < MIN_OUTLIER_IQR_SEVERITY:
                continue
            outliers.append(
                {
                    "column": item["column"],
                    "side": "max",
                    "value": max_value,
                    "threshold": round(upper_threshold, 4),
                    "severity": round(severity, 4),
                }
            )
    return sorted(outliers, key=lambda item: float(item["severity"]), reverse=True)[:5]


def _is_id_like_column(column: str) -> bool:
    lowered = column.lower()
    return (
        lowered == "id"
        or lowered.endswith("_id")
        or lowered.endswith("id")
        or lowered.endswith("_key")
        or lowered.endswith("_code")
    )


def _correlation_strength(coefficient: float) -> str:
    absolute = abs(coefficient)
    if absolute >= 0.7:
        return "mạnh"
    if absolute >= 0.5:
        return "vừa"
    return "không đáng kể"


def _find_signal_by_column(
    items: list[dict[str, Any]], column: str
) -> dict[str, Any] | None:
    for item in items:
        if str(item.get("column")) == column:
            return item
    return None


def _contains_column(lowered_text: str, column: str) -> bool:
    lowered_column = column.lower()
    return lowered_column in lowered_text


def _format_count(value: Any) -> str:
    try:
        return f"{int(value):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "0"


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return _format_count(numeric)
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
