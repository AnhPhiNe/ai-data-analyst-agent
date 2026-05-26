import pandas as pd
from backend.agent.helpers import (
    has_any_phrase,
    get_histogram_bins,
    detect_aggregation,
    has_group_intent,
    detect_chart_type,
)


def test_has_any_phrase() -> None:
    assert has_any_phrase("Hãy tính trung bình", ("trung binh", "mean")) is True
    assert has_any_phrase("Hãy tính tổng", ("trung binh", "mean")) is False


def test_get_histogram_bins() -> None:
    # 1. Row count <= 0 or unique count <= 0
    df_empty = pd.DataFrame({"col": []})
    assert get_histogram_bins(df_empty, "col") == 10

    # 2. Unique count <= 20
    df_small = pd.DataFrame({"col": [1] * 10 + [2] * 5})  # 2 unique values
    assert get_histogram_bins(df_small, "col") == 2

    # 3. Rice bins calculation
    df_large = pd.DataFrame({"col": list(range(100))})  # 100 unique values
    # row_count = 100
    # rice_bins = ceil(2 * (100 ** (1/3))) = ceil(2 * 4.64) = 10
    # min(50, 100, 10) = 10
    assert get_histogram_bins(df_large, "col") == 10


def test_detect_aggregation() -> None:
    assert detect_aggregation("trung bình lương") == "mean"
    assert detect_aggregation("tổng thu nhập") == "sum"
    assert detect_aggregation("trung vị lương") == "median"
    assert detect_aggregation("nhỏ nhất") == "min"
    assert detect_aggregation("lớn nhất") == "max"
    assert detect_aggregation("điểm số") is None


def test_has_group_intent() -> None:
    assert has_group_intent("lương theo phòng ban") is True
    assert has_group_intent("lương group by phòng ban") is True
    assert has_group_intent("lương") is False


def test_detect_chart_type() -> None:
    assert detect_chart_type("vẽ phân tán") == "scatter"
    assert detect_chart_type("vẽ đường") == "line"
    assert detect_chart_type("vẽ phân phối") == "histogram"
    assert detect_chart_type("vẽ heatmap tương quan") == "correlation_heatmap"
    assert detect_chart_type("vẽ boxplot") == "box"
    assert detect_chart_type("vẽ biểu đồ tròn") == "pie"
    assert detect_chart_type("vẽ biểu đồ cột") == "bar"
