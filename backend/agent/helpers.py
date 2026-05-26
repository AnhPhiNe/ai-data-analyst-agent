import math
import pandas as pd
from backend.agent.column_resolver import normalize_text


def has_any_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    norm_text = normalize_text(text)
    return any(normalize_text(phrase) in norm_text for phrase in phrases)


def get_histogram_bins(dataframe: pd.DataFrame, column: str) -> int:
    series = dataframe[column].dropna()
    row_count = int(series.count())
    unique_count = int(series.nunique())
    if row_count <= 0 or unique_count <= 0:
        return 10
    if unique_count <= 20:
        return max(1, unique_count)
    rice_bins = math.ceil(2 * (row_count ** (1 / 3)))
    return max(8, min(50, unique_count, rice_bins))


def detect_aggregation(text: str) -> str | None:
    norm_text = normalize_text(text)
    if has_any_phrase(norm_text, ("trung binh", "average", "mean", "avg")):
        return "mean"
    if has_any_phrase(norm_text, ("tong", "sum", "total")):
        return "sum"
    if has_any_phrase(norm_text, ("median", "trung vi")):
        return "median"
    if has_any_phrase(norm_text, ("min", "nho nhat")):
        return "min"
    if has_any_phrase(norm_text, ("max", "lon nhat", "cao nhat")):
        return "max"
    return None


def has_group_intent(text: str) -> bool:
    norm_text = normalize_text(text)
    return (
        has_any_phrase(norm_text, ("theo nhom", "by group", "group by", "theo", "nhom"))
        or " by " in f" {norm_text} "
    )


def detect_chart_type(text: str) -> str:
    norm_text = normalize_text(text)
    if has_any_phrase(norm_text, ("scatter", "phan tan")):
        return "scatter"
    if has_any_phrase(norm_text, ("line", "duong")):
        return "line"
    if has_any_phrase(norm_text, ("histogram", "phan phoi")):
        return "histogram"
    if "correlation_heatmap" in norm_text or has_any_phrase(
        norm_text, ("heatmap", "correlation", "tuong quan")
    ):
        return "correlation_heatmap"
    if has_any_phrase(norm_text, ("box", "boxplot")):
        return "box"
    if has_any_phrase(norm_text, ("pie", "tron")):
        return "pie"
    return "bar"
