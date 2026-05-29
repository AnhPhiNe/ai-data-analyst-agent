import pandas as pd
from backend.agent.column_resolver import (
    normalize_text,
    contains_normalized_column,
    resolve_column,
)


def test_normalize_text() -> None:
    # 1. Strips Vietnamese accents
    assert normalize_text("Tính trung bình lương") == "tinh trung binh luong"
    assert normalize_text("chạy_code") == "chay code"
    # 2. Handles special char 'đ' and '\u0111'
    assert normalize_text("điểm thi") == "diem thi"
    # 3. Strips multiple spaces and non-alphanumeric chars
    assert normalize_text("  salary  $$  department ") == "salary department"


def test_contains_normalized_column() -> None:
    assert contains_normalized_column("tinh trung binh salary", "salary") is True
    # The compact check fallback 'eval' in 'evaluation' returns True
    assert contains_normalized_column("tinh trung binh evaluation", "eval") is True


def test_resolve_column() -> None:
    df = pd.DataFrame(
        {
            "department": ["HR", "Sales"],
            "Monthly_Revenue": [1000.0, 1500.0],
            "is_active": [True, False],
        }
    )

    # 1. Exact match
    assert resolve_column(df, "Monthly_Revenue") == "Monthly_Revenue"

    # 2. Resolves with fuzzy matches & accents
    assert resolve_column(df, "doanh thu hang thang") == "Monthly_Revenue"
    assert resolve_column(df, "phong ban") == "department"

    # 3. Categorical vs Numeric expected type checks
    assert (
        resolve_column(df, "Monthly_Revenue", expected_type="numeric")
        == "Monthly_Revenue"
    )
    # department is not numeric
    assert resolve_column(df, "phong ban", expected_type="numeric") is None

    assert resolve_column(df, "phong ban", expected_type="categorical") == "department"
    # Monthly_Revenue is numeric, not categorical
    assert (
        resolve_column(df, "doanh thu hang thang", expected_type="categorical") is None
    )


def test_resolve_column_supports_bidirectional_vietnamese_english_aliases() -> None:
    english_df = pd.DataFrame(
        {
            "department": ["Engineering", "Sales"],
            "salary": [1200.0, 900.0],
            "Monthly_Revenue": [1000.0, 1500.0],
            "Exam_Score": [8.5, 7.0],
        }
    )
    vietnamese_df = pd.DataFrame(
        {
            "phong_ban": ["Engineering", "Sales"],
            "luong": [1200.0, 900.0],
            "doanh_thu": [1000.0, 1500.0],
            "diem_thi": [8.5, 7.0],
        }
    )

    assert resolve_column(english_df, "luong", expected_type="numeric") == "salary"
    assert (
        resolve_column(english_df, "doanh thu", expected_type="numeric")
        == "Monthly_Revenue"
    )
    assert (
        resolve_column(english_df, "diem thi", expected_type="numeric") == "Exam_Score"
    )
    assert resolve_column(vietnamese_df, "salary", expected_type="numeric") == "luong"
    assert (
        resolve_column(vietnamese_df, "revenue", expected_type="numeric") == "doanh_thu"
    )
    assert resolve_column(vietnamese_df, "score", expected_type="numeric") == "diem_thi"
    assert (
        resolve_column(vietnamese_df, "department", expected_type="categorical")
        == "phong_ban"
    )
