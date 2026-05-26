from typing import Any
import difflib
import math
import re

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.agent.column_resolver import (
    contains_normalized_column,
    normalize_text,
    resolve_column,
)
from backend.services.session_store import DatasetSession


def find_mentioned_numeric_column(session: DatasetSession, question: str) -> str | None:
    normalized_question = _normalize_text(question)
    for column in session.dataframe.columns:
        column_name = str(column)
        if not is_numeric_dtype(session.dataframe[column_name]) or is_bool_dtype(
            session.dataframe[column_name]
        ):
            continue
        normalized_column = _normalize_text(column_name.replace("_", " "))
        if _contains_normalized_column(normalized_question, normalized_column):
            return column_name

    target_phrase = _extract_correlation_target_phrase(question)
    if target_phrase is not None:
        return _find_close_numeric_column(session, target_phrase)
    return None


def correlation_target_issue(
    session: DatasetSession, question: str, tool_name: str
) -> str | None:
    if tool_name != "correlation_analysis":
        return None

    if find_mentioned_numeric_column(session, question) is not None:
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


def correlation_answer(question: str, table: list[dict[str, Any]]) -> str | None:
    target_column = _find_target_in_correlation_table(question, table)
    if target_column is None:
        return "Đã tính xong ma trận tương quan cho các cột numeric đã chọn."

    target_row = next(
        (row for row in table if row.get("column") == target_column), None
    )
    if target_row is None:
        return None

    candidates: list[tuple[str, float]] = []
    for column, value in target_row.items():
        if column == "column" or column == target_column:
            continue
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and not math.isnan(float(value))
        ):
            candidates.append((column, float(value)))

    if not candidates:
        return None

    normalized_question = _normalize_text(question)
    if _asks_for_negative_correlation(normalized_question):
        negative_candidates = sorted(
            [
                (column, coefficient)
                for column, coefficient in candidates
                if coefficient < 0
            ],
            key=lambda item: item[1],
        )
        if not negative_candidates:
            return f"Không có cột numeric nào có tương quan âm với '{target_column}' trong các cột đã kiểm tra."
        details = ", ".join(
            f"{column} (r={coefficient:.3f})"
            for column, coefficient in negative_candidates
        )
        return (
            f"Các cột có tương quan âm với '{target_column}' là: {details}. "
            "Lưu ý: tương quan không khẳng định quan hệ nhân quả."
        )

    if _asks_for_positive_correlation(normalized_question):
        positive_candidates = sorted(
            [
                (column, coefficient)
                for column, coefficient in candidates
                if coefficient > 0
            ],
            key=lambda item: item[1],
            reverse=True,
        )
        if not positive_candidates:
            return f"Không có cột numeric nào có tương quan dương với '{target_column}' trong các cột đã kiểm tra."
        details = ", ".join(
            f"{column} (r={coefficient:.3f})"
            for column, coefficient in positive_candidates
        )
        return (
            f"Các cột có tương quan dương với '{target_column}' là: {details}. "
            "Lưu ý: tương quan không khẳng định quan hệ nhân quả."
        )

    strongest_column, coefficient = max(candidates, key=lambda item: abs(item[1]))
    direction = "dương" if coefficient >= 0 else "âm"
    return (
        f"Dữ liệu cho thấy '{strongest_column}' có tương quan {direction} mạnh nhất với '{target_column}' "
        f"trong các cột numeric đã kiểm tra, với hệ số tương quan khoảng {coefficient:.3f}. "
        "Lưu ý: tương quan không khẳng định quan hệ nhân quả."
    )


def _extract_correlation_target_phrase(question: str) -> str | None:
    normalized = _normalize_text(question)
    if not any(
        phrase in normalized
        for phrase in ("tuong quan", "lien quan", "correlation", "related")
    ):
        return None

    explicit_target = _extract_explicit_target_phrase(normalized)
    if explicit_target is not None:
        return explicit_target

    for marker in (" voi ", " with ", " to "):
        if marker in f" {normalized} ":
            target = f" {normalized} ".split(marker, 1)[1].strip()
            target = re.sub(
                r"\b(nhat|manh nhat|cao nhat|khong|khong)\b", "", target
            ).strip()
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
    return resolve_column(session.dataframe, phrase)


def _find_close_numeric_column(session: DatasetSession, phrase: str) -> str | None:
    numeric_columns = [
        str(column)
        for column in session.dataframe.columns
        if is_numeric_dtype(session.dataframe[str(column)])
        and not is_bool_dtype(session.dataframe[str(column)])
    ]
    resolved = resolve_column(session.dataframe, phrase, expected_type="numeric")
    if resolved is not None:
        return resolved
    return _find_close_column_name(phrase, numeric_columns)


def _find_target_in_correlation_table(
    question: str, table: list[dict[str, Any]]
) -> str | None:
    normalized_question = _normalize_text(question)
    columns = [str(row.get("column")) for row in table if row.get("column") is not None]
    for column in columns:
        normalized_column = _normalize_text(column.replace("_", " "))
        if _contains_normalized_column(normalized_question, normalized_column):
            return column

    target_phrase = _extract_correlation_target_phrase(question)
    if target_phrase is not None:
        resolved = _resolve_column_from_candidates(target_phrase, columns)
        if resolved is not None:
            return resolved
        return _find_close_column_name(target_phrase, columns)
    return None


def _asks_for_negative_correlation(normalized_question: str) -> bool:
    return any(
        token in normalized_question
        for token in ("tuong quan am", "correlation am", "negative correlation")
    )


def _asks_for_positive_correlation(normalized_question: str) -> bool:
    return any(
        token in normalized_question
        for token in ("tuong quan duong", "correlation duong", "positive correlation")
    )


def _contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    return contains_normalized_column(normalized_text, normalized_column)


def _find_close_column_name(phrase: str, columns: list[str]) -> str | None:
    normalized_phrase = _normalize_text(phrase)
    lookup = {_normalize_text(column.replace("_", " ")): column for column in columns}
    matches = difflib.get_close_matches(
        normalized_phrase, list(lookup), n=1, cutoff=0.86
    )
    if not matches:
        return None
    return lookup[matches[0]]


def _resolve_column_from_candidates(phrase: str, columns: list[str]) -> str | None:
    if not columns:
        return None
    candidate_frame = pd.DataFrame({column: [0.0] for column in columns})
    return resolve_column(candidate_frame, phrase, expected_type="numeric")


def _normalize_text(text: str) -> str:
    return normalize_text(text)
