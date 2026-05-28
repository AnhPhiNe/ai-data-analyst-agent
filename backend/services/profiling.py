import math
import re
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


SEMANTIC_ALIAS_MAP: dict[str, tuple[str, ...]] = {
    "age": ("tuổi", "độ tuổi"),
    "amount": ("số tiền", "giá trị"),
    "attendance": ("chuyên cần", "tỷ lệ đi học"),
    "category": ("danh mục", "nhóm", "loại"),
    "customer": ("khách hàng",),
    "date": ("ngày", "thời gian"),
    "department": ("phòng ban", "bộ phận"),
    "duration": ("thời lượng", "thời gian"),
    "exam": ("điểm thi", "bài thi"),
    "gender": ("giới tính",),
    "grade": ("điểm", "điểm số", "xếp hạng"),
    "hours": ("giờ", "số giờ"),
    "income": ("thu nhập",),
    "level": ("mức độ", "cấp độ"),
    "mark": ("điểm", "điểm số"),
    "monthly": ("hàng tháng", "theo tháng"),
    "parent": ("phụ huynh", "cha mẹ"),
    "parental": ("phụ huynh", "cha mẹ"),
    "price": ("giá", "giá bán"),
    "product": ("sản phẩm",),
    "quality": ("chất lượng",),
    "quantity": ("số lượng",),
    "region": ("vùng", "khu vực"),
    "revenue": ("doanh thu",),
    "salary": ("lương", "thu nhập"),
    "score": ("điểm", "điểm số", "kết quả"),
    "teacher": ("giáo viên", "giảng viên"),
    "type": ("loại", "kiểu"),
}


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
        "column_metadata": _column_metadata(dataframe, column_profiles),
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


def _column_metadata(
    dataframe: pd.DataFrame, column_profiles: list[dict[str, object]]
) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    profile_by_name = {str(profile["name"]): profile for profile in column_profiles}

    for column in dataframe.columns:
        column_name = str(column)
        series = dataframe[column]
        non_null = series.dropna()
        profile = profile_by_name[column_name]
        unique_count = int(non_null.nunique())
        non_null_count = int(profile["non_null_count"])
        unique_ratio = unique_count / non_null_count if non_null_count else 0.0
        inferred_kind = _infer_column_kind(column_name, series, unique_count)
        metadata.append(
            {
                "name": column_name,
                "dtype": str(series.dtype),
                "missing_percent": float(profile["missing_percent"]),
                "unique_count": unique_count,
                "unique_ratio": round(unique_ratio, 4),
                "sample_values": _sample_values(non_null),
                "inferred_kind": inferred_kind,
                "analysis_role": _analysis_role(inferred_kind, series, unique_ratio),
                "semantic_aliases": _semantic_aliases(column_name),
            }
        )

    return metadata


def _sample_values(series: pd.Series, limit: int = 3) -> list[object]:
    samples: list[object] = []
    for value in series.head(limit):
        samples.append(_json_safe(value))
    return samples


def _infer_column_kind(column_name: str, series: pd.Series, unique_count: int) -> str:
    normalized_name = column_name.lower().replace("-", "_").replace(" ", "_")
    non_null_count = int(series.notna().sum())
    unique_ratio = unique_count / non_null_count if non_null_count else 0.0

    if normalized_name == "id" or normalized_name.endswith("_id"):
        return "id_like"
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numeric"
    if is_datetime64_any_dtype(series) or _looks_datetime_like(series):
        return "datetime_like"
    if non_null_count >= 10 and unique_ratio >= 0.95:
        return "id_like"
    return "categorical"


def _analysis_role(inferred_kind: str, series: pd.Series, unique_ratio: float) -> str:
    if inferred_kind == "id_like":
        return "identifier"
    if inferred_kind == "boolean":
        return "boolean_dimension"
    if inferred_kind == "datetime_like":
        return "time_dimension"
    if inferred_kind == "numeric":
        return "numeric_metric"
    if unique_ratio >= 0.8 and int(series.notna().sum()) >= 20:
        return "high_cardinality_dimension"
    return "categorical_dimension"


def _semantic_aliases(column_name: str) -> list[str]:
    tokens = _column_tokens(column_name)
    aliases: list[str] = []
    for token in tokens:
        aliases.extend(SEMANTIC_ALIAS_MAP.get(token, ()))
    return list(dict.fromkeys(aliases))[:6]


def _column_tokens(column_name: str) -> list[str]:
    normalized = column_name.lower().replace("-", "_").replace(" ", "_")
    tokens = re.split(r"[_\W]+", normalized)
    return [token for token in tokens if token]


def _looks_datetime_like(series: pd.Series) -> bool:
    non_null = series.dropna().head(25)
    if non_null.empty:
        return False
    string_values = non_null.astype(str)
    date_like = string_values.str.contains(
        r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})|(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        regex=True,
    )
    if float(date_like.mean()) < 0.8:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)


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
