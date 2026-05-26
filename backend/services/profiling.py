import math
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def dataframe_preview(
    dataframe: pd.DataFrame, limit: int = 10
) -> list[dict[str, object]]:
    preview_frame = dataframe.head(limit).astype(object)
    preview_frame = preview_frame.where(pd.notna(preview_frame), None)
    return preview_frame.to_dict(orient="records")


def list_columns(dataframe: pd.DataFrame) -> list[str]:
    return [str(column) for column in dataframe.columns]


def profile_dataset(dataframe: pd.DataFrame) -> dict[str, object]:
    rows, columns = dataframe.shape
    column_profiles = _column_profiles(dataframe)

    return {
        "rows": int(rows),
        "columns": int(columns),
        "column_names": list_columns(dataframe),
        "preview": dataframe_preview(dataframe),
        "dtypes": column_profiles,
        "missing_values": [
            profile for profile in column_profiles if profile["missing_count"] > 0
        ],
        "numeric_summary": _numeric_summary(dataframe),
        "top_categories": _top_categories(dataframe),
        "distributions": _distribution_specs(dataframe),
    }


def _column_profiles(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    row_count = len(dataframe)
    profiles: list[dict[str, object]] = []

    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        missing_percent = round(
            (missing_count / row_count * 100) if row_count else 0.0, 2
        )
        profiles.append(
            {
                "name": str(column),
                "dtype": str(dataframe[column].dtype),
                "non_null_count": int(dataframe[column].notna().sum()),
                "missing_count": missing_count,
                "missing_percent": missing_percent,
            }
        )

    return profiles


def _numeric_summary(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    numeric_columns = dataframe.select_dtypes(include="number").columns

    for column in numeric_columns:
        series = dataframe[column].dropna()
        quantiles = (
            series.quantile([0.25, 0.5, 0.75])
            if not series.empty
            else pd.Series(dtype=float)
        )
        summaries.append(
            {
                "column": str(column),
                "count": int(series.count()),
                "mean": _round_or_none(series.mean() if not series.empty else None),
                "std": _round_or_none(series.std() if len(series) > 1 else None),
                "min": _round_or_none(series.min() if not series.empty else None),
                "p25": _round_or_none(quantiles.get(0.25)),
                "median": _round_or_none(quantiles.get(0.5)),
                "p75": _round_or_none(quantiles.get(0.75)),
                "max": _round_or_none(series.max() if not series.empty else None),
            }
        )

    return summaries


def _top_categories(
    dataframe: pd.DataFrame, max_values: int = 5
) -> list[dict[str, object]]:
    categories: list[dict[str, object]] = []
    row_count = len(dataframe)

    for column in dataframe.columns:
        series = dataframe[column]
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            continue

        value_counts = series.dropna().astype(str).value_counts().head(max_values)
        if value_counts.empty:
            continue

        categories.append(
            {
                "column": str(column),
                "values": [
                    {
                        "value": str(value),
                        "count": int(count),
                        "percent": round(
                            (int(count) / row_count * 100) if row_count else 0.0, 2
                        ),
                    }
                    for value, count in value_counts.items()
                ],
            }
        )

    return categories


def _distribution_specs(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []

    for column in dataframe.columns:
        series = dataframe[column].dropna()
        if series.empty:
            continue

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            spec = _numeric_distribution_spec(str(column), series)
        else:
            spec = _category_distribution_spec(str(column), series)

        if spec:
            specs.append(spec)

    return specs


def _numeric_distribution_spec(
    column: str, series: pd.Series
) -> dict[str, object] | None:
    unique_count = int(series.nunique())
    if unique_count == 0:
        return None

    if unique_count <= 10:
        counts = series.value_counts().sort_index()
        data = [
            {"bin": str(_json_safe(value)), "count": int(count)}
            for value, count in counts.items()
        ]
    else:
        bins = min(10, unique_count)
        bucketed = pd.cut(series, bins=bins, duplicates="drop")
        counts = bucketed.value_counts().sort_index()
        data = [
            {"bin": str(interval), "count": int(count)}
            for interval, count in counts.items()
        ]

    return {
        "chart_type": "histogram",
        "column": column,
        "x_label": column,
        "y_label": "Count",
        "data": data,
    }


def _category_distribution_spec(
    column: str, series: pd.Series
) -> dict[str, object] | None:
    counts = series.astype(str).value_counts().head(10)
    if counts.empty:
        return None

    return {
        "chart_type": "bar",
        "column": column,
        "x_label": column,
        "y_label": "Count",
        "data": [
            {"category": str(value), "count": int(count)}
            for value, count in counts.items()
        ],
    }


def _round_or_none(value: Any) -> float | None:
    value = _json_safe(value)
    if value is None:
        return None
    return round(float(value), 4)
