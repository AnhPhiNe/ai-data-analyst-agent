from __future__ import annotations

from typing import Any

import pandas as pd

from backend.agent.column_resolver import resolve_column


def repair_tool_column_arguments(
    dataframe: pd.DataFrame,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return arguments

    repaired = dict(arguments)
    if tool_name == "generate_chart_spec":
        _normalize_chart_argument_aliases(repaired)

    if tool_name == "describe_numeric":
        _repair_column_key(dataframe, repaired, "column", expected_type="numeric")
    elif tool_name == "value_counts":
        _repair_column_key(dataframe, repaired, "column")
    elif tool_name == "aggregate_metric":
        _repair_column_key(dataframe, repaired, "metric_column", expected_type="numeric")
        _repair_column_key(dataframe, repaired, "group_by", expected_type="categorical")
    elif tool_name == "sort_values":
        _repair_column_key(dataframe, repaired, "column")
    elif tool_name in {"filter_rows", "conditional_percentage"}:
        operator = str(repaired.get("operator", "")).lower()
        expected_type = "numeric" if operator in {"gt", "gte", "lt", "lte"} else None
        _repair_column_key(dataframe, repaired, "column", expected_type=expected_type)
    elif tool_name == "correlation_analysis":
        _repair_column_list(dataframe, repaired, "columns", expected_type="numeric")
    elif tool_name == "generate_chart_spec":
        _repair_chart_column_arguments(dataframe, repaired)
    return repaired


def _normalize_chart_argument_aliases(arguments: dict[str, Any]) -> None:
    aliases = {
        "x_axis": "x",
        "y_axis": "y",
        "xAxis": "x",
        "yAxis": "y",
        "name": "names",
        "name_column": "names",
        "names_column": "names",
        "value": "values",
        "value_column": "values",
        "values_column": "values",
        "metric_column": "y",
        "group_by": "x",
    }
    for alias, canonical in aliases.items():
        if alias in arguments:
            if canonical not in arguments:
                arguments[canonical] = arguments[alias]
            arguments.pop(alias, None)


def _repair_chart_column_arguments(dataframe: pd.DataFrame, arguments: dict[str, Any]) -> None:
    chart_type = str(arguments.get("chart_type", "")).lower()
    if chart_type == "histogram":
        _repair_column_key(dataframe, arguments, "x", expected_type="numeric")
    elif chart_type == "scatter":
        _repair_column_key(dataframe, arguments, "x", expected_type="numeric")
        _repair_column_key(dataframe, arguments, "y", expected_type="numeric")
    elif chart_type in {"bar", "line"}:
        _repair_column_key(dataframe, arguments, "x")
        _repair_column_key(dataframe, arguments, "y", expected_type="numeric")
    elif chart_type == "box":
        _repair_column_key(dataframe, arguments, "x")
        _repair_column_key(dataframe, arguments, "y", expected_type="numeric")
    elif chart_type == "pie":
        _repair_column_key(dataframe, arguments, "x")
        _repair_column_key(dataframe, arguments, "names")
        _repair_column_key(dataframe, arguments, "y", expected_type="numeric")
        _repair_column_key(dataframe, arguments, "values", expected_type="numeric")
    elif chart_type == "correlation_heatmap":
        _repair_column_list(dataframe, arguments, "columns", expected_type="numeric")

    _repair_column_key(dataframe, arguments, "color")


def _repair_column_key(
    dataframe: pd.DataFrame,
    arguments: dict[str, Any],
    key: str,
    expected_type: str | None = None,
) -> None:
    value = arguments.get(key)
    resolved = _resolve_argument_column(dataframe, value, expected_type=expected_type)
    if resolved is not None:
        arguments[key] = resolved


def _repair_column_list(
    dataframe: pd.DataFrame,
    arguments: dict[str, Any],
    key: str,
    expected_type: str | None = None,
) -> None:
    values = arguments.get(key)
    if values is None or not isinstance(values, list):
        return

    repaired_values = []
    for value in values:
        resolved = _resolve_argument_column(dataframe, value, expected_type=expected_type)
        repaired_values.append(resolved if resolved is not None else value)
    arguments[key] = repaired_values


def _resolve_argument_column(
    dataframe: pd.DataFrame,
    value: Any,
    expected_type: str | None = None,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    if value in dataframe.columns:
        return value

    resolved = resolve_column(dataframe, value, expected_type=expected_type)
    if resolved is not None:
        return resolved
    if expected_type is not None:
        return resolve_column(dataframe, value)
    return None
