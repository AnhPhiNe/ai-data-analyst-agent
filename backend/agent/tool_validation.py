from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from backend.tools.safe_pandas import DANGEROUS_ARG_KEYS, TOOL_REGISTRY
from backend.tools.sql_safety import validate_read_only_sql
from backend.visualization.chart_specs import (
    ChartSpecValidationError,
    validate_chart_spec,
)


@dataclass(frozen=True)
class ToolCallValidationResult:
    is_valid: bool
    message: str
    normalized_arguments: dict[str, Any] = field(default_factory=dict)


def validate_tool_call(
    dataframe: pd.DataFrame, tool_name: str, arguments: Any
) -> ToolCallValidationResult:
    if tool_name not in TOOL_REGISTRY:
        return _invalid(f"Tool '{tool_name}' is not allowed.")

    if not isinstance(arguments, dict):
        return _invalid("Tool arguments must be a JSON object.")

    dangerous_key = _find_dangerous_key(arguments)
    if dangerous_key is not None:
        return _invalid(f"Argument key '{dangerous_key}' is not allowed.")

    try:
        normalized_arguments = _validate_tool_specific_arguments(
            dataframe, tool_name, arguments
        )
    except ValueError as exc:
        return _invalid(str(exc))

    return ToolCallValidationResult(
        is_valid=True,
        message="Tool call is valid.",
        normalized_arguments=normalized_arguments,
    )


def _validate_tool_specific_arguments(
    dataframe: pd.DataFrame,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name in {
        "list_columns",
        "profile_dataset",
        "detect_missing_values",
        "data_quality_report",
    }:
        return {}

    if tool_name == "describe_numeric":
        column = arguments.get("column")
        if column is None:
            if not _numeric_columns(dataframe):
                raise ValueError("No numeric columns are available.")
            return {}
        column = _required_string(arguments, "column")
        _require_column(dataframe, column)
        _require_numeric(dataframe, column)
        return {"column": column}

    if tool_name == "value_counts":
        column = _required_string(arguments, "column")
        _require_column(dataframe, column)
        return {
            "column": column,
            "top_n": _bounded_int(arguments.get("top_n", 10), "top_n", 1, 50),
        }

    if tool_name == "aggregate_metric":
        metric_column = _required_string(arguments, "metric_column")
        group_by = _required_string(arguments, "group_by")
        operation = str(arguments.get("operation", "mean")).lower()
        _require_column(dataframe, metric_column)
        _require_column(dataframe, group_by)
        _require_numeric(dataframe, metric_column)
        if operation not in {"mean", "sum", "min", "max", "median", "count"}:
            raise ValueError(f"Unsupported aggregation operation '{operation}'.")
        return {
            "metric_column": metric_column,
            "group_by": group_by,
            "operation": operation,
            "limit": _bounded_int(arguments.get("limit", 20), "limit", 1, 100),
        }

    if tool_name == "compare_groups":
        metric_column = _required_string(arguments, "metric_column")
        group_by = _required_string(arguments, "group_by")
        operation = str(arguments.get("operation", "mean")).lower()
        _require_column(dataframe, metric_column)
        _require_column(dataframe, group_by)
        _require_numeric(dataframe, metric_column)
        if operation not in {"mean", "median", "min", "max", "count"}:
            raise ValueError(f"Unsupported comparison operation '{operation}'.")
        return {
            "metric_column": metric_column,
            "group_by": group_by,
            "operation": operation,
            "limit": _bounded_int(arguments.get("limit", 20), "limit", 1, 100),
        }

    if tool_name == "sort_values":
        column = _required_string(arguments, "column")
        _require_column(dataframe, column)
        return {
            "column": column,
            "ascending": bool(arguments.get("ascending", False)),
            "limit": _bounded_int(arguments.get("limit", 10), "limit", 1, 100),
        }

    if tool_name == "outlier_detection":
        column = _required_string(arguments, "column")
        _require_column(dataframe, column)
        _require_numeric(dataframe, column)
        return {
            "column": column,
            "limit": _bounded_int(arguments.get("limit", 20), "limit", 1, 100),
        }

    if tool_name == "filter_rows":
        column = _required_string(arguments, "column")
        operator = _required_string(arguments, "operator").lower()
        value = arguments.get("value")
        _require_column(dataframe, column)
        _validate_filter_operator(dataframe[column], operator, value)
        return {
            "column": column,
            "operator": operator,
            "value": value,
            "limit": _bounded_int(arguments.get("limit", 20), "limit", 1, 100),
        }

    if tool_name == "conditional_percentage":
        column = _required_string(arguments, "column")
        operator = _required_string(arguments, "operator").lower()
        value = arguments.get("value")
        _require_column(dataframe, column)
        _validate_filter_operator(dataframe[column], operator, value)
        return {
            "column": column,
            "operator": operator,
            "value": value,
        }

    if tool_name == "correlation_analysis":
        columns = arguments.get("columns")
        if columns is None:
            columns = _numeric_columns(dataframe)
        if not isinstance(columns, list) or not all(
            isinstance(column, str) for column in columns
        ):
            raise ValueError("'columns' must be a list of column names.")
        for column in columns:
            _require_column(dataframe, column)
            _require_numeric(dataframe, column)
        if len(columns) < 2:
            raise ValueError(
                "Correlation analysis requires at least two numeric columns."
            )
        return {"columns": columns}

    if tool_name == "generate_chart_spec":
        try:
            return validate_chart_spec(arguments, dataframe)
        except ChartSpecValidationError as exc:
            raise ValueError(str(exc)) from exc

    if tool_name == "query_table_sql":
        validated_sql = validate_read_only_sql(
            arguments.get("sql"), arguments.get("limit", 100)
        )
        return {"sql": validated_sql.sql, "limit": validated_sql.limit}

    raise ValueError(f"Tool '{tool_name}' is not supported by validation.")


def _validate_filter_operator(series: pd.Series, operator: str, value: Any) -> None:
    if operator in {"eq", "ne", "is_missing", "is_not_missing"}:
        return
    if operator == "contains":
        if value is None:
            raise ValueError("'contains' filter requires a value.")
        return
    if operator == "in":
        if not isinstance(value, list):
            raise ValueError("'in' filter requires a list value.")
        return
    if operator in {"gt", "gte", "lt", "lte"}:
        if not is_numeric_dtype(series):
            raise ValueError(f"Operator '{operator}' requires a numeric column.")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"Operator '{operator}' requires a numeric value.")
        return
    raise ValueError(f"Unsupported filter operator '{operator}'.")


def _find_dangerous_key(arguments: dict[str, Any]) -> str | None:
    for key, value in arguments.items():
        if key.lower() in DANGEROUS_ARG_KEYS or key.startswith("__"):
            return key
        if isinstance(value, dict):
            nested = _find_dangerous_key(value)
            if nested is not None:
                return nested
    return None


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required.")
    return value.strip()


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' does not exist.")


def _require_numeric(dataframe: pd.DataFrame, column: str) -> None:
    if not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column]):
        raise ValueError(f"Column '{column}' must be numeric.")


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not is_bool_dtype(dataframe[column])
    ]


def _bounded_int(value: Any, name: str, min_value: int, max_value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{name}' must be an integer.")
    if value < min_value or value > max_value:
        raise ValueError(f"'{name}' must be between {min_value} and {max_value}.")
    return value


def _invalid(message: str) -> ToolCallValidationResult:
    return ToolCallValidationResult(is_valid=False, message=message)
