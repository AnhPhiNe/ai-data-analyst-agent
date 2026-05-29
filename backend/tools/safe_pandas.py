from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict

from backend.services.profiling import profile_dataset
from backend.tools.sql_safety import validate_read_only_sql
from backend.visualization.chart_specs import (
    ChartSpecValidationError,
    validate_chart_spec,
)


ToolStatus = Literal["success", "error"]

DANGEROUS_ARG_KEYS = {
    "__class__",
    "__dict__",
    "__globals__",
    "__subclasses__",
    "code",
    "command",
    "eval",
    "exec",
    "expr",
    "expression",
    "file",
    "filepath",
    "path",
    "python",
    "script",
    "shell",
}


class ToolValidationError(ValueError):
    """Raised when a tool call is unsafe or invalid."""


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: ToolStatus
    message: str
    data: dict[str, Any] | list[Any] | None = None
    table: list[dict[str, Any]] | None = None
    chart_spec: dict[str, Any] | None = None


ToolFunction = Callable[[pd.DataFrame, dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    function: ToolFunction


def execute_tool(
    dataframe: pd.DataFrame, tool_name: str, arguments: dict[str, Any] | None = None
) -> ToolResult:
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(
            tool_name=tool_name,
            status="error",
            message=f"Tool '{tool_name}' is not allowed.",
        )

    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ToolResult(
            tool_name=tool_name,
            status="error",
            message="Tool arguments must be a JSON object.",
        )

    try:
        _validate_safe_arguments(arguments)
        return TOOL_REGISTRY[tool_name].function(dataframe, arguments)
    except ToolValidationError as exc:
        return ToolResult(tool_name=tool_name, status="error", message=str(exc))


def list_columns_tool(dataframe: pd.DataFrame, arguments: dict[str, Any]) -> ToolResult:
    table = [
        {"column": str(column), "dtype": str(dataframe[column].dtype)}
        for column in dataframe.columns
    ]
    return ToolResult(
        tool_name="list_columns",
        status="success",
        message=f"Dataset has {len(table)} columns.",
        data={"columns": [item["column"] for item in table]},
        table=table,
    )


def profile_dataset_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    profile = profile_dataset(dataframe)
    return ToolResult(
        tool_name="profile_dataset",
        status="success",
        message=f"Dataset has {profile['rows']} rows and {profile['columns']} columns.",
        data=profile,
    )


def describe_numeric_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    column = arguments.get("column")
    numeric_columns = _numeric_columns(dataframe)

    if column is not None:
        if not isinstance(column, str):
            raise ToolValidationError("'column' must be a string.")
        _require_column(dataframe, column)
        _require_numeric(dataframe, column)
        numeric_columns = [column]

    if not numeric_columns:
        raise ToolValidationError("No numeric columns are available.")

    table = []
    for numeric_column in numeric_columns:
        series = dataframe[numeric_column].dropna()
        table.append(
            {
                "column": numeric_column,
                "count": int(series.count()),
                "mean": _round(series.mean()) if not series.empty else None,
                "std": _round(series.std()) if len(series) > 1 else None,
                "min": _round(series.min()) if not series.empty else None,
                "median": _round(series.median()) if not series.empty else None,
                "max": _round(series.max()) if not series.empty else None,
            }
        )

    return ToolResult(
        tool_name="describe_numeric",
        status="success",
        message=f"Described {len(table)} numeric column(s).",
        table=table,
    )


def detect_missing_values_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    row_count = len(dataframe)
    table = []
    for column in dataframe.columns:
        missing_count = int(dataframe[column].isna().sum())
        table.append(
            {
                "column": str(column),
                "missing_count": missing_count,
                "missing_percent": round(
                    (missing_count / row_count * 100) if row_count else 0.0, 2
                ),
            }
        )

    return ToolResult(
        tool_name="detect_missing_values",
        status="success",
        message="Missing values were calculated for all columns.",
        table=table,
    )


def data_quality_report_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    row_count = int(len(dataframe))
    column_count = int(len(dataframe.columns))
    duplicate_rows = int(dataframe.duplicated().sum())
    table: list[dict[str, Any]] = []

    missing_columns = []
    constant_columns = []
    high_cardinality_columns = []
    possible_id_columns = []

    for column in dataframe.columns:
        column_name = str(column)
        series = dataframe[column]
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        missing_percent = round(
            (missing_count / row_count * 100) if row_count else 0.0, 2
        )
        unique_count = int(series.dropna().nunique())
        unique_ratio = round(
            (unique_count / non_null_count) if non_null_count else 0.0, 4
        )

        if missing_count > 0:
            missing_columns.append(column_name)
            table.append(
                {
                    "check": "missing_values",
                    "column": column_name,
                    "count": missing_count,
                    "percent": missing_percent,
                    "detail": f"{missing_count} missing value(s)",
                }
            )

        if non_null_count > 0 and unique_count <= 1:
            constant_columns.append(column_name)
            table.append(
                {
                    "check": "constant_column",
                    "column": column_name,
                    "count": unique_count,
                    "percent": 100.0,
                    "detail": "Only one distinct non-null value",
                }
            )

        if _is_high_cardinality(series, unique_count, non_null_count):
            high_cardinality_columns.append(column_name)
            table.append(
                {
                    "check": "high_cardinality",
                    "column": column_name,
                    "count": unique_count,
                    "percent": round(unique_ratio * 100, 2),
                    "detail": "Many distinct values for a categorical-like column",
                }
            )

        if _is_possible_id_column(column_name, unique_count, non_null_count):
            possible_id_columns.append(column_name)
            table.append(
                {
                    "check": "possible_id_column",
                    "column": column_name,
                    "count": unique_count,
                    "percent": round(unique_ratio * 100, 2),
                    "detail": "Column name or uniqueness ratio looks ID-like",
                }
            )

    if duplicate_rows > 0:
        table.insert(
            0,
            {
                "check": "duplicate_rows",
                "column": None,
                "count": duplicate_rows,
                "percent": round(
                    (duplicate_rows / row_count * 100) if row_count else 0.0, 2
                ),
                "detail": "Fully duplicated row(s)",
            },
        )

    summary = {
        "rows": row_count,
        "columns": column_count,
        "duplicate_rows": duplicate_rows,
        "missing_columns": missing_columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality_columns,
        "possible_id_columns": possible_id_columns,
        "analysis_candidate_columns": [
            str(column)
            for column in dataframe.columns
            if str(column)
            not in set(
                possible_id_columns + constant_columns + high_cardinality_columns
            )
        ],
        "issue_count": len(table),
    }

    return ToolResult(
        tool_name="data_quality_report",
        status="success",
        message=f"Found {len(table)} data quality signal(s).",
        data=summary,
        table=table,
    )


def value_counts_tool(dataframe: pd.DataFrame, arguments: dict[str, Any]) -> ToolResult:
    column = _required_string(arguments, "column")
    _require_column(dataframe, column)
    top_n = _bounded_int(arguments.get("top_n", 10), "top_n", min_value=1, max_value=50)

    counts = dataframe[column].dropna().astype(str).value_counts().head(top_n)
    total = int(dataframe[column].notna().sum())
    table = [
        {
            "value": str(value),
            "count": int(count),
            "percent": round((int(count) / total * 100) if total else 0.0, 2),
        }
        for value, count in counts.items()
    ]

    return ToolResult(
        tool_name="value_counts",
        status="success",
        message=f"Computed top {len(table)} values for '{column}'.",
        data={
            "column": column,
            "unique_count": int(dataframe[column].dropna().astype(str).nunique()),
            "non_null_count": total,
            "top_n": top_n,
        },
        table=table,
    )


def aggregate_metric_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    metric_column = _required_string(arguments, "metric_column")
    group_by = _required_string(arguments, "group_by")
    operation = str(arguments.get("operation", "mean")).lower()
    limit = _bounded_int(
        arguments.get("limit", 20), "limit", min_value=1, max_value=100
    )

    _require_column(dataframe, metric_column)
    _require_column(dataframe, group_by)
    _require_numeric(dataframe, metric_column)

    allowed_operations = {"mean", "sum", "min", "max", "median", "count"}
    if operation not in allowed_operations:
        raise ToolValidationError(f"Unsupported aggregation operation '{operation}'.")

    grouped = (
        dataframe.groupby(group_by, dropna=False)[metric_column]
        .agg(operation)
        .reset_index()
    )
    result_column = f"{operation}_{metric_column}"
    grouped = grouped.rename(columns={metric_column: result_column})
    grouped = grouped.sort_values(result_column, ascending=False).head(limit)

    return ToolResult(
        tool_name="aggregate_metric",
        status="success",
        message=f"Computed {operation} of '{metric_column}' by '{group_by}'.",
        data={
            "metric_column": metric_column,
            "group_by": group_by,
            "operation": operation,
            "outlier_summary": _numeric_outlier_summary(dataframe, metric_column),
        },
        table=_records(grouped),
    )


def compare_groups_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    metric_column = _required_string(arguments, "metric_column")
    group_by = _required_string(arguments, "group_by")
    operation = str(arguments.get("operation", "mean")).lower()
    limit = _bounded_int(
        arguments.get("limit", 20), "limit", min_value=1, max_value=100
    )

    _require_column(dataframe, metric_column)
    _require_column(dataframe, group_by)
    _require_numeric(dataframe, metric_column)

    if operation not in {"mean", "median", "min", "max", "count"}:
        raise ToolValidationError(f"Unsupported comparison operation '{operation}'.")

    grouped = (
        dataframe.groupby(group_by, dropna=False)[metric_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    overall_mean = dataframe[metric_column].mean()
    grouped["diff_from_overall_mean"] = grouped["mean"] - overall_mean

    rename_map = {
        "count": "count",
        "mean": f"mean_{metric_column}",
        "median": f"median_{metric_column}",
        "min": f"min_{metric_column}",
        "max": f"max_{metric_column}",
    }
    grouped = grouped.rename(columns=rename_map)
    sort_column = rename_map.get(operation, f"mean_{metric_column}")
    grouped = grouped.sort_values(sort_column, ascending=False).head(limit)

    rounded = grouped.copy()
    for column in rounded.columns:
        if column != group_by and is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(_round)

    return ToolResult(
        tool_name="compare_groups",
        status="success",
        message=f"Compared '{metric_column}' across groups in '{group_by}'.",
        data={
            "metric_column": metric_column,
            "group_by": group_by,
            "operation": operation,
            "overall_mean": _round(overall_mean),
            "groups_returned": int(len(rounded)),
            "outlier_summary": _numeric_outlier_summary(dataframe, metric_column),
        },
        table=_records(rounded),
    )


def sort_values_tool(dataframe: pd.DataFrame, arguments: dict[str, Any]) -> ToolResult:
    column = _required_string(arguments, "column")
    _require_column(dataframe, column)
    ascending = bool(arguments.get("ascending", False))
    limit = _bounded_int(
        arguments.get("limit", 10), "limit", min_value=1, max_value=100
    )

    sorted_frame = dataframe.sort_values(
        column, ascending=ascending, na_position="last"
    ).head(limit)
    return ToolResult(
        tool_name="sort_values",
        status="success",
        message=f"Sorted rows by '{column}'.",
        table=_records(sorted_frame),
    )


def outlier_detection_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    column = _required_string(arguments, "column")
    limit = _bounded_int(
        arguments.get("limit", 20), "limit", min_value=1, max_value=100
    )

    _require_column(dataframe, column)
    _require_numeric(dataframe, column)

    summary = _numeric_outlier_summary(dataframe, column)
    if int(summary["valid_count"]) == 0:
        raise ToolValidationError(f"Column '{column}' has no valid numeric values.")

    lower_bound = float(summary["lower_bound"])
    upper_bound = float(summary["upper_bound"])
    mask = dataframe[column].notna() & (
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    )
    outliers = dataframe.loc[mask].head(limit).copy()
    outliers.insert(0, "row_index", outliers.index)

    return ToolResult(
        tool_name="outlier_detection",
        status="success",
        message=f"Detected {summary['outlier_count']} outlier row(s) in '{column}' using IQR.",
        data={**summary, "returned_rows": int(len(outliers))},
        table=_records(outliers),
    )


def filter_rows_tool(dataframe: pd.DataFrame, arguments: dict[str, Any]) -> ToolResult:
    column = _required_string(arguments, "column")
    operator = _required_string(arguments, "operator").lower()
    value = arguments.get("value")
    limit = _bounded_int(
        arguments.get("limit", 20), "limit", min_value=1, max_value=100
    )

    _require_column(dataframe, column)
    mask = _build_filter_mask(dataframe[column], operator, value)
    filtered = dataframe.loc[mask].head(limit)

    return ToolResult(
        tool_name="filter_rows",
        status="success",
        message=f"Filtered {int(mask.sum())} matching row(s) for '{column}' {operator}.",
        data={
            "matched_rows": int(mask.sum()),
            "returned_rows": int(len(filtered)),
            "total_rows": int(len(dataframe)),
        },
        table=_records(filtered),
    )


def conditional_percentage_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    column = _required_string(arguments, "column")
    operator = _required_string(arguments, "operator").lower()
    value = arguments.get("value")

    _require_column(dataframe, column)
    mask = _build_filter_mask(dataframe[column], operator, value)
    valid_mask = dataframe[column].notna()
    valid_rows = int(valid_mask.sum())
    matched_rows = int((mask & valid_mask).sum())
    total_rows = int(len(dataframe))
    percent_of_valid = round(
        (matched_rows / valid_rows * 100) if valid_rows else 0.0, 2
    )
    percent_of_rows = round((matched_rows / total_rows * 100) if total_rows else 0.0, 2)

    return ToolResult(
        tool_name="conditional_percentage",
        status="success",
        message=f"Computed percentage for '{column}' {operator}.",
        data={
            "column": column,
            "operator": operator,
            "value": value,
            "matched_rows": matched_rows,
            "valid_rows": valid_rows,
            "total_rows": total_rows,
            "percent_of_valid": percent_of_valid,
            "percent_of_rows": percent_of_rows,
        },
    )


def correlation_analysis_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    columns_arg = arguments.get("columns")
    if columns_arg is None:
        columns = _numeric_columns(dataframe)
    else:
        if not isinstance(columns_arg, list) or not all(
            isinstance(column, str) for column in columns_arg
        ):
            raise ToolValidationError("'columns' must be a list of column names.")
        columns = columns_arg

    for column in columns:
        _require_column(dataframe, column)
        _require_numeric(dataframe, column)

    if len(columns) < 2:
        raise ToolValidationError(
            "Correlation analysis requires at least two numeric columns."
        )

    matrix = dataframe[columns].corr(numeric_only=True)
    return ToolResult(
        tool_name="correlation_analysis",
        status="success",
        message=f"Computed correlation matrix for {len(columns)} numeric columns.",
        table=_records(matrix.reset_index().rename(columns={"index": "column"})),
    )


def generate_chart_spec_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    chart_spec = _validated_chart_spec(dataframe, arguments)
    return ToolResult(
        tool_name="generate_chart_spec",
        status="success",
        message=f"Generated a {chart_spec['chart_type']} chart spec.",
        chart_spec=chart_spec,
    )


def query_table_sql_tool(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> ToolResult:
    try:
        validated_sql = validate_read_only_sql(
            arguments.get("sql"), arguments.get("limit", 100)
        )
    except ValueError as exc:
        raise ToolValidationError(str(exc)) from exc
    try:
        import duckdb
    except ImportError as exc:
        raise ToolValidationError(
            "DuckDB is not installed. Install the 'duckdb' package to use SQL fallback."
        ) from exc

    try:
        with duckdb.connect(database=":memory:") as connection:
            connection.register("dataset", dataframe)
            result = connection.execute(validated_sql.executable_sql).fetchdf()
    except Exception as exc:
        raise ToolValidationError(f"SQL execution failed: {exc}") from exc

    return ToolResult(
        tool_name="query_table_sql",
        status="success",
        message=f"Executed read-only SQL query and returned {len(result)} row(s).",
        data={"sql": validated_sql.sql, "limit": validated_sql.limit},
        table=_records(result),
    )


def _validated_chart_spec(
    dataframe: pd.DataFrame, arguments: dict[str, Any]
) -> dict[str, Any]:
    try:
        return validate_chart_spec(arguments, dataframe)
    except ChartSpecValidationError as exc:
        raise ToolValidationError(str(exc)) from exc


def _build_filter_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    if operator == "eq":
        return series == value
    if operator == "ne":
        return series != value
    if operator == "contains":
        if value is None:
            raise ToolValidationError("'contains' filter requires a value.")
        return series.astype(str).str.contains(
            str(value), case=False, na=False, regex=False
        )
    if operator == "in":
        if not isinstance(value, list):
            raise ToolValidationError("'in' filter requires a list value.")
        return series.isin(value)
    if operator == "is_missing":
        return series.isna()
    if operator == "is_not_missing":
        return series.notna()

    numeric_operators = {
        "gt": lambda current: current > value,
        "gte": lambda current: current >= value,
        "lt": lambda current: current < value,
        "lte": lambda current: current <= value,
    }
    if operator in numeric_operators:
        if not is_numeric_dtype(series):
            raise ToolValidationError(
                f"Operator '{operator}' requires a numeric column."
            )
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ToolValidationError(
                f"Operator '{operator}' requires a numeric value."
            )
        return numeric_operators[operator](series)

    raise ToolValidationError(f"Unsupported filter operator '{operator}'.")


def _condition_label(column: str, operator: str, value: Any) -> str:
    labels = {
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "eq": "=",
        "ne": "!=",
    }
    if operator == "is_missing":
        return f"{column} is missing"
    if operator == "is_not_missing":
        return f"{column} is not missing"
    symbol = labels.get(operator, operator)
    return f"{column} {symbol} {value}"


def _is_high_cardinality(
    series: pd.Series, unique_count: int, non_null_count: int
) -> bool:
    if non_null_count < 20:
        return False
    if is_numeric_dtype(series) and not is_bool_dtype(series):
        return False
    unique_ratio = unique_count / non_null_count if non_null_count else 0.0
    return unique_count >= 20 and unique_ratio >= 0.5


def _is_possible_id_column(
    column_name: str, unique_count: int, non_null_count: int
) -> bool:
    if non_null_count == 0:
        return False
    normalized_name = column_name.lower().replace("-", "_").replace(" ", "_")
    unique_ratio = unique_count / non_null_count
    return (
        normalized_name == "id"
        or normalized_name.endswith("_id")
        or (non_null_count >= 20 and unique_ratio >= 0.95)
    )


def _validate_safe_arguments(arguments: dict[str, Any]) -> None:
    for key, value in arguments.items():
        if key.lower() in DANGEROUS_ARG_KEYS or key.startswith("__"):
            raise ToolValidationError(f"Argument key '{key}' is not allowed.")
        if isinstance(value, dict):
            _validate_safe_arguments(value)


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"'{key}' is required.")
    return value.strip()


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise ToolValidationError(f"Column '{column}' does not exist.")


def _require_numeric(dataframe: pd.DataFrame, column: str) -> None:
    if not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column]):
        raise ToolValidationError(f"Column '{column}' must be numeric.")


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not is_bool_dtype(dataframe[column])
    ]


def _bounded_int(value: Any, name: str, min_value: int, max_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolValidationError(f"'{name}' must be an integer.")
    if value < min_value or value > max_value:
        raise ToolValidationError(
            f"'{name}' must be between {min_value} and {max_value}."
        )
    return value


def _records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    clean_frame = dataframe.astype(object).where(pd.notna(dataframe), None)
    return clean_frame.to_dict(orient="records")


def _numeric_outlier_summary(dataframe: pd.DataFrame, column: str) -> dict[str, Any]:
    series = dataframe[column].dropna()
    if series.empty:
        return {
            "column": column,
            "method": "iqr",
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "valid_count": 0,
            "outlier_percent": 0.0,
        }
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    mask = dataframe[column].notna() & (
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    )
    outlier_count = int(mask.sum())
    valid_count = int(series.count())
    return {
        "column": column,
        "method": "iqr",
        "q1": _round(q1),
        "q3": _round(q3),
        "iqr": _round(iqr),
        "lower_bound": _round(lower_bound),
        "upper_bound": _round(upper_bound),
        "outlier_count": outlier_count,
        "valid_count": valid_count,
        "outlier_percent": round(
            (outlier_count / valid_count * 100) if valid_count else 0.0, 2
        ),
    }


def _round(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), 4)


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "list_columns": ToolDefinition(
        "list_columns", "List dataset columns and dtypes.", list_columns_tool
    ),
    "profile_dataset": ToolDefinition(
        "profile_dataset", "Return dataset profile summary.", profile_dataset_tool
    ),
    "describe_numeric": ToolDefinition(
        "describe_numeric",
        "Describe one or all numeric columns.",
        describe_numeric_tool,
    ),
    "detect_missing_values": ToolDefinition(
        "detect_missing_values",
        "Calculate missing values for every column.",
        detect_missing_values_tool,
    ),
    "data_quality_report": ToolDefinition(
        "data_quality_report",
        "Report missing values, duplicates, constant columns, high-cardinality columns, and possible ID columns.",
        data_quality_report_tool,
    ),
    "value_counts": ToolDefinition(
        "value_counts", "Return top values for a column.", value_counts_tool
    ),
    "aggregate_metric": ToolDefinition(
        "aggregate_metric",
        "Aggregate a numeric metric by a group column.",
        aggregate_metric_tool,
    ),
    "compare_groups": ToolDefinition(
        "compare_groups",
        "Compare numeric metric statistics across groups.",
        compare_groups_tool,
    ),
    "sort_values": ToolDefinition(
        "sort_values", "Sort rows by a column.", sort_values_tool
    ),
    "outlier_detection": ToolDefinition(
        "outlier_detection",
        "Detect numeric outliers using the IQR method.",
        outlier_detection_tool,
    ),
    "filter_rows": ToolDefinition(
        "filter_rows", "Filter rows using a safe operator.", filter_rows_tool
    ),
    "conditional_percentage": ToolDefinition(
        "conditional_percentage",
        "Calculate the percentage of rows matching a safe condition.",
        conditional_percentage_tool,
    ),
    "correlation_analysis": ToolDefinition(
        "correlation_analysis",
        "Compute a correlation matrix for numeric columns.",
        correlation_analysis_tool,
    ),
    "generate_chart_spec": ToolDefinition(
        "generate_chart_spec",
        "Generate a safe chart specification without executable code.",
        generate_chart_spec_tool,
    ),
    "query_table_sql": ToolDefinition(
        "query_table_sql",
        "Execute a read-only DuckDB SELECT query against the current dataset table.",
        query_table_sql_tool,
    ),
}
