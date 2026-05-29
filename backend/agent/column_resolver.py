from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


ExpectedType = Literal["numeric", "categorical"]
MIN_COLUMN_SCORE = 0.72
AMBIGUITY_MARGIN = 0.08
HIGH_SIGNAL_TOKENS = {
    "age",
    "amount",
    "attendance",
    "duration",
    "exam",
    "grade",
    "hours",
    "income",
    "mark",
    "price",
    "revenue",
    "salary",
    "score",
    "sleep",
}

TOKEN_TRANSLATIONS: dict[str, tuple[str, ...]] = {
    "age": ("tuoi", "do tuoi"),
    "amount": ("so tien", "gia tri"),
    "attendance": ("chuyen can", "di hoc", "tham du", "ty le di hoc"),
    "activity": ("hoat dong",),
    "activities": ("hoat dong",),
    "category": ("danh muc", "nhom", "loai"),
    "class": ("lop",),
    "customer": ("khach hang",),
    "date": ("ngay", "thoi gian"),
    "department": ("phong ban", "bo phan"),
    "duration": ("thoi luong", "thoi gian"),
    "education": ("giao duc", "hoc van", "trinh do hoc van"),
    "exam": ("thi", "bai thi", "ky thi"),
    "final": ("cuoi ky", "cuoi cung"),
    "grade": ("diem", "diem so", "xep hang"),
    "hours": ("gio", "so gio", "thoi gian"),
    "income": ("thu nhap",),
    "level": ("muc do", "cap do", "trinh do"),
    "mark": ("diem", "diem so"),
    "monthly": ("hang thang", "theo thang"),
    "parent": ("phu huynh", "cha me"),
    "parental": ("phu huynh", "cha me"),
    "physical": ("the chat", "van dong"),
    "previous": ("truoc do", "truoc"),
    "price": ("gia", "gia ban"),
    "product": ("san pham",),
    "quantity": ("so luong",),
    "region": ("vung", "khu vuc"),
    "revenue": ("doanh thu",),
    "salary": ("luong", "tien luong", "muc luong"),
    "score": ("diem", "diem so", "ket qua"),
    "sleep": ("ngu", "giac ngu"),
    "school": ("truong", "nha truong"),
    "studied": ("hoc", "da hoc"),
    "study": ("hoc", "hoc tap"),
    "sessions": ("buoi", "so buoi", "phien"),
    "teacher": ("giao vien", "giang vien"),
    "type": ("loai", "kieu"),
    "quality": ("chat luong",),
    "tutoring": ("phu dao", "day kem", "hoc them"),
}


@dataclass(frozen=True)
class ColumnMatch:
    column: str
    score: float
    ambiguous: bool = False


def resolve_column(
    dataframe: pd.DataFrame,
    text: str,
    expected_type: ExpectedType | None = None,
) -> str | None:
    match = resolve_column_match(dataframe, text, expected_type=expected_type)
    if match is None or match.ambiguous:
        return None
    return match.column


def resolve_column_match(
    dataframe: pd.DataFrame,
    text: str,
    expected_type: ExpectedType | None = None,
) -> ColumnMatch | None:
    normalized_text = normalize_text(text)
    candidates = _candidate_columns(dataframe, expected_type)
    scored = [
        ColumnMatch(column=column, score=_score_column(column, normalized_text))
        for column in candidates
    ]
    scored = [item for item in scored if item.score >= MIN_COLUMN_SCORE]
    if not scored:
        return None
    scored.sort(key=lambda item: item.score, reverse=True)
    best = scored[0]
    if len(scored) > 1 and best.score - scored[1].score < AMBIGUITY_MARGIN:
        return ColumnMatch(column=best.column, score=best.score, ambiguous=True)
    return best


def normalize_text(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(char for char in stripped if not unicodedata.combining(char))
    ascii_text = ascii_text.replace("đ", "d").replace("_", " ")
    ascii_text = ascii_text.replace("đ", "d")
    ascii_text = ascii_text.replace("\u0111", "d")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return " ".join(ascii_text.strip().split())


def contains_normalized_column(normalized_text: str, normalized_column: str) -> bool:
    if re.search(rf"(?<!\w){re.escape(normalized_column)}(?!\w)", normalized_text):
        return True
    return normalized_column.replace(" ", "") in normalized_text.replace(" ", "")


def normalize_identifier(identifier: str) -> str:
    return normalize_text(identifier.replace("_", " "))


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


def _score_column(column: str, normalized_text: str) -> float:
    normalized_column = normalize_identifier(column)
    if contains_normalized_column(normalized_text, normalized_column):
        return 1.0

    compact_text = normalized_text.replace(" ", "")
    compact_column = normalized_column.replace(" ", "")
    best_score = difflib.SequenceMatcher(None, compact_text, compact_column).ratio()

    reverse_terms = _reverse_translation_terms(normalized_column)
    if reverse_terms and any(
        _term_in_text(term, normalized_text) for term in reverse_terms
    ):
        best_score = max(best_score, 0.92)

    column_terms = _column_terms(normalized_column)
    if column_terms:
        matched_terms = sum(
            1 for term in column_terms if _term_in_text(term, normalized_text)
        )
        token_score = matched_terms / len(column_terms)
        best_score = max(best_score, token_score)
    best_score = max(
        best_score, _column_token_score(normalized_column, normalized_text)
    )

    for phrase in _text_phrases(normalized_text):
        phrase_score = difflib.SequenceMatcher(
            None, phrase.replace(" ", ""), compact_column
        ).ratio()
        best_score = max(best_score, phrase_score)

    return best_score


def _column_terms(normalized_column: str) -> set[str]:
    terms: set[str] = set()
    terms.update(_reverse_translation_terms(normalized_column))
    for token in normalized_column.split():
        terms.add(token)
        terms.update(_translation_terms_for_token(token))
    return {term for term in terms if term}


def _column_token_score(normalized_column: str, normalized_text: str) -> float:
    tokens = normalized_column.split()
    if not tokens:
        return 0.0

    matched_tokens = []
    high_signal_matched = False
    for token in tokens:
        terms = {token, *_translation_terms_for_token(token)}
        if any(_term_in_text(term, normalized_text) for term in terms):
            matched_tokens.append(token)
            if any(
                term in HIGH_SIGNAL_TOKENS and _term_in_text(term, normalized_text)
                for term in terms
            ):
                high_signal_matched = True

    if not matched_tokens:
        return 0.0

    score = len(matched_tokens) / len(tokens)
    if len(matched_tokens) == 1 and (
        matched_tokens[0] in HIGH_SIGNAL_TOKENS or high_signal_matched
    ):
        score = max(score, 0.82)
    return score


def _translation_terms_for_token(token: str) -> set[str]:
    terms = set(TOKEN_TRANSLATIONS.get(token, ()))
    terms.update(_reverse_translation_terms(token))
    return terms


def _reverse_translation_terms(normalized_term: str) -> set[str]:
    matches: set[str] = set()
    compact_term = normalized_term.replace(" ", "")
    for english_token, aliases in TOKEN_TRANSLATIONS.items():
        for alias in aliases:
            normalized_alias = normalize_text(alias)
            compact_alias = normalized_alias.replace(" ", "")
            if normalized_term == normalized_alias or compact_term == compact_alias:
                matches.add(english_token)
    return matches


def _term_in_text(term: str, normalized_text: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False
    if re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized_text):
        return True
    return normalized_term.replace(" ", "") in normalized_text.replace(" ", "")


def _text_phrases(normalized_text: str) -> set[str]:
    tokens = normalized_text.split()
    phrases = {normalized_text}
    for size in range(1, min(4, len(tokens)) + 1):
        for start in range(0, len(tokens) - size + 1):
            phrases.add(" ".join(tokens[start : start + size]))
    return phrases


def _is_numeric_column(dataframe: pd.DataFrame, column: str) -> bool:
    return is_numeric_dtype(dataframe[column]) and not is_bool_dtype(dataframe[column])
