from __future__ import annotations

import json
from typing import Literal, TYPE_CHECKING

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:
    from backend.agent.gemini_runtime import LLMProvider

ExpectedType = Literal["numeric", "categorical"]


def resolve_column(
    dataframe: pd.DataFrame,
    text: str,
    provider: LLMProvider | None = None,
    expected_type: ExpectedType | None = None,
) -> str | None:
    if provider is None:
        from backend.main import get_llm_provider
        provider = get_llm_provider()
        
    if not provider:
        return None

    candidates = _candidate_columns(dataframe, expected_type)
    if not candidates:
        return None

    prompt = (
        "Bạn là một trợ lý phân tích dữ liệu chuyên nghiệp.\n"
        "Nhiệm vụ của bạn là đọc câu hỏi của người dùng và xác định xem người dùng đang nhắc đến cột nào trong danh sách cột cho sẵn.\n"
        "Luật:\n"
        "1. CHỈ ĐƯỢC PHÉP chọn 1 cột DUY NHẤT có trong danh sách Cột Khả Dụng.\n"
        "2. Nếu câu hỏi không nhắc đến cột nào cụ thể, hoặc bạn không chắc chắn, hoặc có nhiều hơn 1 cột phù hợp (mập mờ), hãy trả về null.\n"
        "3. Kết quả bắt buộc phải là JSON hợp lệ theo định dạng: {\"matched_column\": \"tên_cột\"} hoặc {\"matched_column\": null}\n"
        "4. KHÔNG giải thích gì thêm.\n\n"
        f"Danh sách Cột Khả Dụng: {json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"Câu hỏi của người dùng: '{text}'"
    )

    try:
        response_json = provider.generate_structured(prompt)
        parsed = json.loads(response_json)
        matched_column = parsed.get("matched_column")
        if matched_column in candidates:
            return matched_column
        return None
    except Exception:
        return None


def _candidate_columns(
    dataframe: pd.DataFrame, expected_type: ExpectedType | None
) -> list[str]:
    columns = [str(column) for column in dataframe.columns]
    if expected_type == "numeric":
        return [column for column in columns if _is_numeric_column(dataframe, column)]
    if expected_type == "categorical":
        return [
            column for column in columns if not _is_numeric_column(dataframe, column)
        ]
    return columns


def normalize_text(text: str) -> str:
    import unicodedata
    import re
    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in stripped if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d").replace("_", " ")
    ascii_text = ascii_text.replace("đ", "d")
    ascii_text = ascii_text.replace("\u0111", "d")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.strip().split())


def contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    import re
    if re.search(rf"(?<!\w){re.escape(normalized_column)}(?!\w)", normalized_text):
        return True
    return normalized_column.replace(" ", "") in normalized_text.replace(" ", "")


def normalize_identifier(identifier: str) -> str:
    return normalize_text(identifier.replace("_", " "))


def _is_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    return is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])

