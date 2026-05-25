import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.services.profiling import profile_dataset


MAX_ITEMS = 5


def generate_auto_analysis(dataframe: pd.DataFrame, profile: dict[str, object] | None = None) -> dict[str, object]:
    profile = profile or profile_dataset(dataframe)
    numeric_columns = _numeric_columns(dataframe)
    categorical_columns = _categorical_columns(dataframe)
    correlations = _correlation_highlights(dataframe, numeric_columns)

    return {
        "workflow_steps": [
            "profile_dataset",
            "detect_missing_values",
            "describe_numeric",
            "value_counts",
            "correlation_analysis" if len(numeric_columns) >= 2 else None,
            "generate_chart_spec",
        ],
        "overview": _overview(profile),
        "data_quality": _data_quality(profile),
        "numeric_highlights": _numeric_highlights(profile),
        "categorical_highlights": _categorical_highlights(profile),
        "correlation_highlights": correlations,
        "recommended_charts": _recommended_charts(dataframe, numeric_columns, categorical_columns, correlations),
        "next_questions": _next_questions(numeric_columns, categorical_columns, correlations),
    }


def _overview(profile: dict[str, object]) -> dict[str, object]:
    return {
        "rows": int(profile.get("rows", 0)),
        "columns": int(profile.get("columns", 0)),
        "column_names": profile.get("column_names", []),
    }


def _data_quality(profile: dict[str, object]) -> dict[str, object]:
    missing_values = list(profile.get("missing_values", []))
    missing_values = sorted(
        missing_values,
        key=lambda item: int(item.get("missing_count", 0)),
        reverse=True,
    )
    total_missing = sum(int(item.get("missing_count", 0)) for item in missing_values)
    return {
        "total_missing_cells": total_missing,
        "columns_with_missing": len(missing_values),
        "top_missing_columns": missing_values[:MAX_ITEMS],
    }


def _numeric_highlights(profile: dict[str, object]) -> list[dict[str, object]]:
    items = []
    for item in profile.get("numeric_summary", []):
        if not isinstance(item, dict):
            continue
        mean = item.get("mean")
        median = item.get("median")
        min_value = item.get("min")
        max_value = item.get("max")
        if mean is None or median is None or min_value is None or max_value is None:
            continue
        items.append(
            {
                "column": item.get("column"),
                "count": item.get("count"),
                "mean": mean,
                "median": median,
                "min": min_value,
                "max": max_value,
                "range_width": round(float(max_value) - float(min_value), 4),
                "mean_median_gap": round(abs(float(mean) - float(median)), 4),
            }
        )
    return sorted(items, key=lambda row: float(row["range_width"]), reverse=True)[:MAX_ITEMS]


def _categorical_highlights(profile: dict[str, object]) -> list[dict[str, object]]:
    highlights = []
    for item in profile.get("top_categories", []):
        if not isinstance(item, dict):
            continue
        values = item.get("values", [])
        if not values:
            continue
        top_value = values[0]
        highlights.append(
            {
                "column": item.get("column"),
                "top_value": top_value.get("value"),
                "count": top_value.get("count"),
                "percent": top_value.get("percent"),
                "unique_values_shown": len(values),
            }
        )
    return sorted(highlights, key=lambda row: float(row.get("percent") or 0), reverse=True)[:MAX_ITEMS]


def _correlation_highlights(dataframe: pd.DataFrame, numeric_columns: list[str]) -> list[dict[str, object]]:
    if len(numeric_columns) < 2:
        return []

    matrix = dataframe[numeric_columns].corr(numeric_only=True)
    highlights = []
    for index, column_a in enumerate(numeric_columns):
        for column_b in numeric_columns[index + 1 :]:
            coefficient = matrix.loc[column_a, column_b]
            if pd.isna(coefficient):
                continue
            coefficient = round(float(coefficient), 4)
            highlights.append(
                {
                    "column_a": column_a,
                    "column_b": column_b,
                    "correlation": coefficient,
                    "abs_correlation": round(abs(coefficient), 4),
                    "direction": "positive" if coefficient >= 0 else "negative",
                }
            )
    return sorted(highlights, key=lambda row: float(row["abs_correlation"]), reverse=True)[:MAX_ITEMS]


def _recommended_charts(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    correlations: list[dict[str, object]],
) -> list[dict[str, object]]:
    charts: list[dict[str, object]] = []
    if numeric_columns:
        charts.append(
            {
                "title": f"Distribution of {numeric_columns[0]}",
                "chart_spec": {"chart_type": "histogram", "x": numeric_columns[0], "bins": _histogram_bins(dataframe, numeric_columns[0])},
                "reason": "Start with the distribution of a numeric column to inspect spread and potential outliers.",
            }
        )
    if correlations:
        top = correlations[0]
        charts.append(
            {
                "title": f"{top['column_a']} vs {top['column_b']}",
                "chart_spec": {"chart_type": "scatter", "x": top["column_a"], "y": top["column_b"]},
                "reason": "This pair has the strongest absolute correlation among numeric columns.",
            }
        )
    if len(numeric_columns) >= 2:
        charts.append(
            {
                "title": "Correlation heatmap",
                "chart_spec": {"chart_type": "correlation_heatmap", "columns": numeric_columns[:12]},
                "reason": "A heatmap gives a compact view of relationships across numeric columns.",
            }
        )
    if numeric_columns and categorical_columns:
        charts.append(
            {
                "title": f"{numeric_columns[0]} by {categorical_columns[0]}",
                "chart_spec": {"chart_type": "box", "x": categorical_columns[0], "y": numeric_columns[0]},
                "reason": "A box plot compares numeric spread across categories.",
            }
        )
    if categorical_columns:
        low_cardinality = next(
            (
                column
                for column in categorical_columns
                if int(dataframe[column].nunique(dropna=True)) <= 10
            ),
            None,
        )
        if low_cardinality is not None:
            charts.append(
                {
                    "title": f"Share of {low_cardinality}",
                    "chart_spec": {"chart_type": "pie", "names": low_cardinality},
                    "reason": "A low-cardinality categorical column can be inspected as a share chart.",
                }
            )
    return charts[:MAX_ITEMS]


def _next_questions(
    numeric_columns: list[str],
    categorical_columns: list[str],
    correlations: list[dict[str, object]],
) -> list[str]:
    questions = ["Có cột nào thiếu dữ liệu nhiều không?"]
    if numeric_columns:
        questions.append(f"Phân phối của {numeric_columns[0]} trông như thế nào?")
    if numeric_columns and categorical_columns:
        questions.append(f"Trung bình {numeric_columns[0]} theo {categorical_columns[0]} là bao nhiêu?")
    if categorical_columns:
        questions.append(f"Giá trị nào xuất hiện nhiều nhất trong {categorical_columns[0]}?")
    if correlations:
        top = correlations[0]
        questions.append(f"{top['column_a']} có tương quan với {top['column_b']} không?")
    return questions[:MAX_ITEMS]


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not is_bool_dtype(dataframe[column])
    ]


def _categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.columns
        if not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column])
    ]


def _histogram_bins(dataframe: pd.DataFrame, column: str) -> int:
    series = dataframe[column].dropna()
    row_count = int(series.count())
    unique_count = int(series.nunique())
    if row_count <= 0 or unique_count <= 0:
        return 10
    if unique_count <= 20:
        return max(1, unique_count)
    return max(8, min(50, unique_count, round(2 * (row_count ** (1 / 3)))))
