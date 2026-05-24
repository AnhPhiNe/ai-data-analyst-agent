import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.agent.gemini_runtime import LLMProvider
from backend.services.profiling import profile_dataset


class SuggestedContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    source: str = "fallback"


def generate_suggested_content(dataframe: pd.DataFrame, provider: LLMProvider | None = None) -> SuggestedContent:
    profile = profile_dataset(dataframe)
    fallback = _fallback_suggested_content(profile)

    if provider is None:
        return fallback

    try:
        raw_response = provider.generate(_build_suggestions_prompt(profile))
        suggested = _parse_suggested_content(raw_response)
    except Exception:
        return fallback

    validated_questions = _validate_questions(suggested.questions, profile)
    validated_insights = _validate_insights(suggested.insights)

    if not validated_questions:
        validated_questions = fallback.questions
    if not validated_insights:
        validated_insights = fallback.insights

    return SuggestedContent(
        questions=validated_questions[:6],
        insights=validated_insights[:5],
        source="gemini",
    )


def _build_suggestions_prompt(profile: dict[str, Any]) -> str:
    safe_profile = {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "column_names": profile["column_names"],
        "dtypes": profile["dtypes"],
        "missing_values": profile["missing_values"],
        "numeric_summary": profile["numeric_summary"],
        "top_categories": profile["top_categories"],
    }
    contract = {
        "questions": [
            "Vietnamese question grounded in existing columns only",
        ],
        "insights": [
            "Short Vietnamese insight grounded only in the provided profile",
        ],
    }
    return (
        "Bạn là data analyst assistant. Dựa trên profile dataset, hãy tạo câu hỏi gợi ý và insight nhẹ.\n"
        "Không bịa business context. Không nhắc tới cột không tồn tại. Không dùng số liệu ngoài profile.\n"
        "Chỉ trả về JSON object hợp lệ theo contract, không markdown.\n\n"
        f"PROFILE={json.dumps(safe_profile, ensure_ascii=False)}\n"
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


def _fallback_suggested_content(profile: dict[str, Any]) -> SuggestedContent:
    numeric_columns = [item["column"] for item in profile.get("numeric_summary", [])]
    categorical_columns = [item["column"] for item in profile.get("top_categories", [])]
    questions: list[str] = [
        "Dataset có bao nhiêu dòng và bao nhiêu cột?",
        "Cột nào đang có giá trị thiếu?",
        "Dataset có những cột nào?",
    ]

    if numeric_columns:
        questions.append(f"Mô tả thống kê cột {numeric_columns[0]}.")
    if categorical_columns:
        questions.append(f"Top giá trị phổ biến nhất của {categorical_columns[0]} là gì?")
    if numeric_columns and categorical_columns:
        questions.append(f"Tính trung bình {numeric_columns[0]} theo {categorical_columns[0]}.")
        questions.append(f"Vẽ biểu đồ {numeric_columns[0]} theo {categorical_columns[0]}.")
    if len(numeric_columns) >= 2:
        questions.append(f"Cột nào tương quan mạnh nhất với {numeric_columns[0]}?")

    insights = _fallback_insights(profile)
    return SuggestedContent(
        questions=_dedupe(questions)[:6],
        insights=insights[:5],
        source="fallback",
    )


def _fallback_insights(profile: dict[str, Any]) -> list[str]:
    insights = [
        f"Dữ liệu có {profile['rows']} dòng và {profile['columns']} cột.",
    ]

    missing_values = profile.get("missing_values", [])
    if missing_values:
        columns = ", ".join(str(item["name"]) for item in missing_values[:3])
        insights.append(f"Dữ liệu cho thấy có giá trị thiếu ở một số cột như {columns}.")
    else:
        insights.append("Dữ liệu không ghi nhận giá trị thiếu trong profile hiện tại.")

    numeric_summary = profile.get("numeric_summary", [])
    if numeric_summary:
        first_numeric = numeric_summary[0]
        insights.append(
            f"Cột {first_numeric['column']} có giá trị trung bình khoảng {first_numeric['mean']}."
        )

    top_categories = profile.get("top_categories", [])
    if top_categories and top_categories[0]["values"]:
        first_category = top_categories[0]
        top_value = first_category["values"][0]
        insights.append(
            f"Ở cột {first_category['column']}, giá trị phổ biến nhất là {top_value['value']} "
            f"với {top_value['count']} dòng."
        )

    return insights


def _validate_questions(questions: list[str], profile: dict[str, Any]) -> list[str]:
    column_names = {str(column) for column in profile["column_names"]}
    validated = []
    for question in questions:
        if not isinstance(question, str) or not question.strip():
            continue
        if _mentions_unknown_structured_column(question, column_names):
            continue
        validated.append(question.strip())
    return _dedupe(validated)


def _validate_insights(insights: list[str]) -> list[str]:
    return _dedupe([insight.strip() for insight in insights if isinstance(insight, str) and insight.strip()])


def _mentions_unknown_structured_column(text: str, column_names: set[str]) -> bool:
    known_lower = {column.lower() for column in column_names}
    tokens = set(text.replace("`", " ").replace(",", " ").split())
    for token in tokens:
        cleaned = token.strip(" .?!:;()[]{}").lower()
        if "_" in cleaned and cleaned not in known_lower:
            return True
    return False


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
