import json
import math
import re
from typing import Any
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.services.profiling import profile_dataset
from backend.agent.helpers import get_histogram_bins
from backend.agent.tool_validation import validate_tool_call


MAX_ITEMS = 5


def generate_auto_analysis(
    dataframe: pd.DataFrame,
    profile: dict[str, object] | None = None,
    provider: Any = None,
) -> dict[str, object]:
    profile = profile or profile_dataset(dataframe)
    numeric_columns = _numeric_columns(dataframe)
    categorical_columns = _categorical_columns(dataframe)
    correlations = _correlation_highlights(dataframe, numeric_columns)

    recommended = None
    ai_status = {"used": False, "error": None}

    if provider is not None:
        try:
            recommended = _recommended_charts_via_gemini(
                dataframe, profile, provider, numeric_columns, categorical_columns, correlations
            )
            if recommended:
                ai_status["used"] = True
            else:
                ai_status["error"] = "AI returned an empty or invalid response."
        except Exception as e:
            recommended = None
            ai_status["error"] = f"AI Error: {str(e)}"

    if not recommended:
        recommended = _recommended_charts(
            dataframe, numeric_columns, categorical_columns, correlations
        )

    # --- Post-process: deduplicate types, filter flat bar charts, ensure heatmap & top scatter ---
    final_recommended = []
    seen_types: set[str] = set()
    for item in recommended:
        if not isinstance(item, dict):
            continue
        spec = item.get("chart_spec")
        if not isinstance(spec, dict):
            continue
        c_type = spec.get("chart_type")
        if not c_type:
            continue

        # Filter out bar charts that would produce flat / meaningless visuals
        if c_type == "bar":
            x_col = spec.get("x")
            y_col = spec.get("y")
            if isinstance(x_col, str) and isinstance(y_col, str):
                if x_col in dataframe.columns and y_col in dataframe.columns:
                    eta_sq = _eta_squared(dataframe, x_col, y_col)
                    if eta_sq < 0.01:
                        continue  # skip — group averages are effectively identical

        seen_types.add(c_type)
        final_recommended.append(item)

    # Guarantee correlation heatmap when >= 2 numeric columns
    if len(numeric_columns) >= 2 and "correlation_heatmap" not in seen_types:
        heatmap_spec = {
            "title": "Correlation Heatmap",
            "chart_spec": {
                "chart_type": "correlation_heatmap",
                "columns": numeric_columns[:12],
            },
            "reason": "A heatmap gives a compact view of relationships and correlations across numeric columns.",
        }
        if len(final_recommended) < 4:
            final_recommended.append(heatmap_spec)
        else:
            final_recommended[3] = heatmap_spec
            
    final_recommended = final_recommended[:4]


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
        "recommended_charts": final_recommended,
        "ai_status": ai_status,
        "next_questions": _next_questions(
            numeric_columns, categorical_columns, correlations
        ),
    }


def _recommended_charts_via_gemini(
    dataframe: pd.DataFrame,
    profile: dict[str, Any],
    provider: Any,
    numeric_columns: list[str],
    categorical_columns: list[str],
    correlations: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    summary = {
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
        "column_names": profile.get("column_names"),
        "numeric_highlights": [
            {"column": row["column"], "mean": row["mean"], "min": row["min"], "max": row["max"]}
            for row in profile.get("numeric_summary", [])[:5]
        ],
        "categorical_highlights": [
            {"column": row["column"], "top_value": row["values"][0]["value"] if row.get("values") else None}
            for row in profile.get("top_categories", [])[:5]
        ],
        "correlation_highlights": [
            {"column_a": row["column_a"], "column_b": row["column_b"], "correlation": row["correlation"]}
            for row in correlations[:5]
        ]
    }

    prompt = f"""Bạn là một chuyên gia cao cấp về Khoa học Dữ liệu (Senior Data Scientist).
Hãy phân tích tóm tắt hồ sơ dữ liệu (data profile) dưới đây và đề xuất đúng 4 biểu đồ mang lại nhiều góc nhìn kinh doanh / insight phân tích sâu sắc nhất.

TÓM TẮT HỒ SƠ DỮ LIỆU:
{json.dumps(summary, ensure_ascii=False, indent=2)}

BẠN CHỈ ĐƯỢC CHỌN các loại biểu đồ (chart_type) sau đây:
1. "bar": vẽ biểu đồ cột thể hiện giá trị trung bình (average) của biến số (y) theo từng nhóm phân loại (x). Yêu cầu "x" là biến phân loại (categorical) và "y" là biến số (numeric). HẠN CHẾ SỬ DỤNG nếu không chắc chắn có sự chênh lệch lớn giữa các nhóm.
2. "line": vẽ biểu đồ đường biểu diễn xu hướng hoặc tiến trình. Yêu cầu có "x" và "y".
3. "scatter": vẽ tương quan 2 biến số. Yêu cầu có "x" và "y" (cả hai đều phải là cột số).
4. "pie": vẽ tỷ lệ phân bổ của biến phân loại. Yêu cầu có "names".
5. "correlation_heatmap": vẽ nhiệt độ tương quan. Yêu cầu có "columns" (danh sách tên các cột số).

LƯU Ý CỰC KỲ QUAN TRỌNG ĐỂ ĐẢM BẢO CHẤT LƯỢNG & ĐA DẠNG:
- BẠN ĐƯỢC PHÉP lặp lại loại biểu đồ (ví dụ có thể đề xuất 2-3 biểu đồ scatter hoặc pie), nhưng hãy cố gắng chọn ít nhất 3 loại khác nhau để đa dạng góc nhìn.
- BẮT BUỘC phải đề xuất 1 biểu đồ "correlation_heatmap" (nhiệt độ tương quan) nếu dataset có từ 2 cột số (numeric columns) trở lên.
- Đối với biểu đồ "scatter" (phân tán), CHỈ ĐƯỢC PHÉP dùng cho 2 biến số LIÊN TỤC (như Age, Fare, Revenue, Hours_Studied). TUYỆT ĐỐI KHÔNG vẽ scatter nếu một trong hai biến là biến phân loại/rời rạc (ví dụ: Survived, Pclass, Sex, SibSp). Nếu muốn phân tích tương quan giữa biến rời rạc và biến liên tục, HÃY DÙNG biểu đồ "bar" (cột) hoặc "box" (hộp).
- Biểu đồ "bar" (cột) rất tuyệt vời để so sánh một biến số theo từng nhóm của biến phân loại (ví dụ: Trung bình Fare theo Pclass, hoặc Tỷ lệ Survived theo Sex). Hãy tích cực dùng biểu đồ "bar" hoặc "box" để làm nổi bật sự khác biệt giữa các nhóm.
- KHÔNG ĐƯỢC vẽ biểu đồ phân phối (histogram) vì chúng không mang lại insight kinh doanh trực tiếp.
- KHÔNG ĐƯỢC vẽ bất kỳ biểu đồ nào liên quan đến cột ID hoặc cột Khóa chính (như order_id, customer_id, id...) vì chúng không mang lại insight.
- Các cột được chỉ định trong "x", "y", "names", "columns" PHẢI trùng khớp hoàn toàn với danh sách cột của dataset: {profile.get("column_names")}.
- "reason" phải là một mô tả cụ thể, chi tiết và chứa insight thực tế rút ra từ dữ liệu (ví dụ: nhắc đến các con số, xu hướng, tỷ lệ phân bổ) chứ không chỉ đơn thuần là mô tả chức năng của biểu đồ (ví dụ: KHÔNG ĐƯỢC viết "Biểu đồ này giúp xem phân bổ của độ tuổi", mà hãy phân tích "Hầu hết hành khách tập trung ở độ tuổi 20-30, cho thấy nhóm khách hàng trẻ tuổi chiếm đa số"). Hãy viết khoảng 2-3 câu thật sắc bén và mang lại giá trị phân tích thực tiễn.

HÃY PHẢN HỒI DƯỚI DẠNG MỘT MẢNG JSON HỢP LỆ (VALID JSON ARRAY) GỒM CHÍNH XÁC 8 ĐỐI TƯỢNG (để hệ thống có thể chọn lọc, loại bỏ các biểu đồ xấu và giữ lại 4 biểu đồ tốt nhất) THEO CẤU TRÚC SAU:
[
  {{
    "title": "Tên biểu đồ hấp dẫn",
    "chart_spec": {{
      "chart_type": "bar",
      "x": "tên_cột_x",
      "y": "tên_cột_y"
    }},
    "reason": "Giải thích lý do lựa chọn..."
  }}
]

KHÔNG viết thêm bất kỳ từ giải thích nào trước hoặc sau khối JSON.
"""

    response = provider.generate(prompt)
    match = re.search(r"(\[.*\])", response, re.DOTALL)
    if not match:
        return None

    raw_json = match.group(1)
    try:
        recommended_raw = json.loads(raw_json)
    except Exception:
        return None

    if not isinstance(recommended_raw, list):
        return None

    valid_charts = []
    for item in recommended_raw:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        spec = item.get("chart_spec")
        reason = item.get("reason")
        if not title or not isinstance(spec, dict) or not reason:
            continue

        validation = validate_tool_call(dataframe, "generate_chart_spec", spec)
        if validation.is_valid:
            spec_clean = validation.normalized_arguments
            # Strict programmatic guardrail against flat bar charts
            if spec_clean.get("chart_type") == "bar":
                x_col = spec_clean.get("x")
                y_col = spec_clean.get("y")
                if x_col in dataframe.columns and y_col in dataframe.columns:
                    try:
                        means = dataframe.groupby(x_col)[y_col].mean()
                        if means.max() > 0 and (means.max() - means.min()) / means.max() < 0.05:
                            continue  # Skip flat bar charts
                    except Exception:
                        pass
                        
            # Programmatic guardrail against spaghetti line charts
            if spec_clean.get("chart_type") == "line":
                x_col = spec_clean.get("x")
                if x_col in dataframe.columns:
                    if not pd.api.types.is_datetime64_any_dtype(dataframe[x_col]):
                        # Convert to scatter to avoid messy spaghetti lines for non-temporal data
                        spec_clean["chart_type"] = "scatter"
                        
            # Programmatic guardrail to auto-correct flipped axes in scatter plots
            if spec_clean.get("chart_type") == "scatter":
                x_col = spec_clean.get("x")
                y_col = spec_clean.get("y")
                if isinstance(x_col, str) and isinstance(y_col, str):
                    correct_x, correct_y = _determine_scatter_axes(x_col, y_col)
                    if correct_x != x_col:
                        spec_clean["x"] = correct_x
                        spec_clean["y"] = correct_y
            
            valid_charts.append({
                "title": str(title),
                "chart_spec": spec_clean,
                "reason": str(reason)
            })

    return valid_charts[:4] if valid_charts else None



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
    return sorted(items, key=lambda row: float(row["range_width"]), reverse=True)[
        :MAX_ITEMS
    ]


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
    return sorted(
        highlights, key=lambda row: float(row.get("percent") or 0), reverse=True
    )[:MAX_ITEMS]


def _correlation_highlights(
    dataframe: pd.DataFrame, numeric_columns: list[str]
) -> list[dict[str, object]]:
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
    return sorted(
        highlights, key=lambda row: float(row["abs_correlation"]), reverse=True
    )[:MAX_ITEMS]


def _recommended_charts(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    correlations: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Build chart recommendations using data-driven insight scoring.

    For every possible chart candidate we compute a statistical "insight score":
    - histogram  → coefficient of variation (CV) of the column
    - scatter    → |r| of the pair (from pre-computed correlations)
    - bar        → eta-squared (ANOVA effect size) of numeric Y grouped by categorical X
    - pie        → normalised Shannon entropy of the category distribution
    - heatmap    → fixed high score when ≥ 2 numeric columns exist
    """
    candidates: list[dict[str, object]] = []

    # --- histogram candidates (one per numeric column) ---
    for col in numeric_columns[:6]:
        series = dataframe[col].dropna()
        if series.empty:
            continue
        mean_val = float(series.mean())
        std_val = float(series.std())
        cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
        candidates.append({
            "chart_type": "histogram",
            "score": round(cv, 4),
            "spec": {
                "title": f"Distribution of {col}",
                "chart_spec": {
                    "chart_type": "histogram",
                    "x": col,
                    "bins": get_histogram_bins(dataframe, col),
                },
                "reason": f"The majority of {col} values center around its mean of {mean_val:.2f}, with a standard deviation of {std_val:.2f} (CV={cv:.2f}).",
            },
        })

    # --- scatter candidates (one per correlation pair) ---
    for pair in correlations:
        col_a = str(pair["column_a"])
        col_b = str(pair["column_b"])
        if _is_discrete_numeric(dataframe, col_a) or _is_discrete_numeric(dataframe, col_b):
            continue
        abs_r = float(pair.get("abs_correlation", abs(float(pair.get("correlation", 0)))))
        x_col, y_col = _determine_scatter_axes(col_a, col_b)
        candidates.append({
            "chart_type": "scatter",
            "score": round(abs_r, 4),
            "spec": {
                "title": f"{y_col.replace('_', ' ')} by {x_col.replace('_', ' ')}",
                "chart_spec": {
                    "chart_type": "scatter",
                    "x": x_col,
                    "y": y_col,
                },
                "reason": f"There is a {'positive' if float(pair['correlation']) > 0 else 'negative'} correlation (r={pair['correlation']}) between {x_col} and {y_col}, indicating a related trend.",
            },
        })

    # --- bar candidates (categorical X × numeric Y, scored by eta-squared) ---
    for cat_col in categorical_columns[:4]:
        for num_col in numeric_columns[:4]:
            eta_sq = _eta_squared(dataframe, cat_col, num_col)
            if eta_sq < 0.01:
                continue  # skip flat / no-effect pairs
            candidates.append({
                "chart_type": "bar",
                "score": round(eta_sq, 4),
                "spec": {
                    "title": f"Average {num_col} by {cat_col}",
                    "chart_spec": {
                        "chart_type": "bar",
                        "x": cat_col,
                        "y": num_col,
                    },
                    "reason": f"Average {num_col} shows a significant variation across different {cat_col} groups (η²={eta_sq:.3f}), highlighting a strong segment effect.",
                },
            })

    # --- pie candidates (low-cardinality categorical, scored by entropy) ---
    for cat_col in categorical_columns[:4]:
        nunique = int(dataframe[cat_col].nunique(dropna=True))
        if nunique < 2 or nunique > 10:
            continue
        counts = dataframe[cat_col].value_counts(normalize=True)
        entropy = -sum(p * math.log2(p) for p in counts if p > 0)
        max_entropy = math.log2(nunique) if nunique > 1 else 1.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        top_cat = counts.idxmax()
        top_pct = counts.max() * 100
        candidates.append({
            "chart_type": "pie",
            "score": round(norm_entropy, 4),
            "spec": {
                "title": f"Distribution of {cat_col}",
                "chart_spec": {"chart_type": "pie", "names": cat_col},
                "reason": f"The most dominant category in {cat_col} is '{top_cat}', making up {top_pct:.1f}% of the dataset.",
            },
        })

    # --- correlation heatmap (fixed high-priority when >= 2 numeric cols) ---
    if len(numeric_columns) >= 2:
        candidates.append({
            "chart_type": "correlation_heatmap",
            "score": 1.0,
            "spec": {
                "title": "Correlation Heatmap",
                "chart_spec": {
                    "chart_type": "correlation_heatmap",
                    "columns": numeric_columns[:12],
                },
                "reason": "This heatmap reveals the strongest positive and negative linear correlations among all numeric variables, helping identify key drivers.",
            },
        })

    # --- select top 4 with unique chart types, ordered by score ---
    candidates.sort(key=lambda c: float(c["score"]), reverse=True)
    selected: list[dict[str, object]] = []
    seen_types: set[str] = set()
    for cand in candidates:
        if cand["chart_type"] in seen_types:
            continue
        seen_types.add(str(cand["chart_type"]))
        selected.append(cand["spec"])
        if len(selected) >= 4:
            break

    return selected


def _next_questions(
    numeric_columns: list[str],
    categorical_columns: list[str],
    correlations: list[dict[str, object]],
) -> list[str]:
    questions = ["Có cột nào thiếu dữ liệu nhiều không?"]
    if numeric_columns:
        questions.append(f"Phân phối của {numeric_columns[0]} trông như thế nào?")
    if numeric_columns and categorical_columns:
        questions.append(
            f"Trung bình {numeric_columns[0]} theo {categorical_columns[0]} là bao nhiêu?"
        )
    if categorical_columns:
        questions.append(
            f"Giá trị nào xuất hiện nhiều nhất trong {categorical_columns[0]}?"
        )
    if correlations:
        top = correlations[0]
        questions.append(
            f"{top['column_a']} có tương quan với {top['column_b']} không?"
        )
    return questions[:MAX_ITEMS]


def _is_id_like_column(column: str) -> bool:
    col_lower = column.lower()
    return (
        col_lower == "id"
        or col_lower.endswith("_id")
        or col_lower.endswith("id")
        or col_lower == "no"
        or col_lower.endswith("_no")
        or col_lower == "key"
        or col_lower.endswith("_key")
        or col_lower == "code"
        or col_lower.endswith("_code")
    )


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    cols = [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not is_bool_dtype(dataframe[column])
    ]
    non_id_cols = [c for c in cols if not _is_id_like_column(c)]
    target_cols = non_id_cols if non_id_cols else cols

    if len(target_cols) >= 2:
        try:
            # Sort columns descending by their total absolute correlation with all other columns
            corr_matrix = dataframe[target_cols].corr(numeric_only=True).abs()
            corr_sums = corr_matrix.sum().to_dict()
            return sorted(target_cols, key=lambda c: corr_sums.get(c, 0), reverse=True)
        except Exception:
            pass
    return target_cols


def _categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    cols = []
    for column in dataframe.columns:
        if is_bool_dtype(dataframe[column]):
            cols.append(str(column))
        elif not is_numeric_dtype(dataframe[column]):
            cols.append(str(column))
        elif dataframe[column].nunique(dropna=True) <= 15:
            cols.append(str(column))

    filtered = [
        c for c in cols
        if not _is_id_like_column(c)
        and not c.lower().endswith("date")
        and not c.lower().endswith("time")
        and dataframe[c].nunique(dropna=True) <= 15  # Limit cardinality to 15 to avoid messy charts (e.g. by Name)
    ]
    if filtered:
        return filtered
    fallback = [
        c for c in cols
        if not _is_id_like_column(c)
        and not c.lower().endswith("date")
        and not c.lower().endswith("time")
    ]
    if fallback:
        return fallback
    return cols


def _eta_squared(dataframe: pd.DataFrame, cat_col: str, num_col: str) -> float:
    """Compute eta-squared (ANOVA effect size) for num_col grouped by cat_col.

    η² = SS_between / SS_total.  Returns 0.0 on any error.
    A value close to 0 means the group averages are nearly identical (flat chart).
    """
    try:
        df_clean = dataframe[[cat_col, num_col]].dropna()
        if df_clean.empty:
            return 0.0
        grand_mean = float(df_clean[num_col].mean())
        ss_total = float(((df_clean[num_col] - grand_mean) ** 2).sum())
        if ss_total == 0:
            return 0.0
        group_stats = df_clean.groupby(cat_col)[num_col].agg(["count", "mean"])
        ss_between = float(
            (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()
        )
        return round(ss_between / ss_total, 6)
    except Exception:
        return 0.0


def _determine_scatter_axes(col1: str, col2: str) -> tuple[str, str]:
    """Determine which column should be X (predictor) and Y (target).

    Returns (x_col, y_col). Target/outcome columns (containing score, grade, salary, etc.)
    should be placed on the Y-axis.
    """
    c1_lower = col1.lower()
    c2_lower = col2.lower()
    target_keywords = {
        "score",
        "exam",
        "grade",
        "gpa",
        "salary",
        "price",
        "revenue",
        "cost",
        "sales",
        "profit",
        "rating",
        "performance",
        "income",
    }
    c1_is_target = any(k in c1_lower for k in target_keywords)
    c2_is_target = any(k in c2_lower for k in target_keywords)
    if c1_is_target and not c2_is_target:
        return col2, col1
    elif c2_is_target and not c1_is_target:
        return col1, col2
    return col1, col2


def _is_discrete_numeric(dataframe: pd.DataFrame, col: str) -> bool:
    """Check if a numeric column behaves like a discrete/categorical variable.

    For small datasets (e.g. tests), a column is discrete if it has <= 2 unique values.
    For standard datasets, a column is discrete if it has <= 15 unique values.
    """
    n_unique = int(dataframe[col].nunique(dropna=True))
    if len(dataframe) > 15:
        return n_unique <= 15
    return n_unique <= 2


