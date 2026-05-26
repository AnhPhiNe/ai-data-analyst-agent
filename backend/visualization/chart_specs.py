from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import BaseModel, ConfigDict, ValidationError


ChartType = Literal[
    "bar", "line", "histogram", "scatter", "correlation_heatmap", "box", "pie"
]


class ChartSpecValidationError(ValueError):
    """Raised when a chart specification is invalid or unsafe."""


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chart_type: ChartType
    title: str | None = None
    x: str | None = None
    y: str | None = None
    names: str | None = None
    values: str | None = None
    color: str | None = None
    columns: list[str] | None = None
    bins: int | None = None


def validate_chart_spec(
    spec: dict[str, Any], dataframe: pd.DataFrame
) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ChartSpecValidationError("Chart spec must be a JSON object.")

    try:
        chart_spec = ChartSpec.model_validate(spec)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        field = ".".join(str(item) for item in first_error["loc"])
        raise ChartSpecValidationError(
            f"Invalid chart spec field '{field}': {first_error['msg']}"
        ) from exc

    normalized = chart_spec.model_dump(exclude_none=True)
    chart_type = chart_spec.chart_type

    if chart_spec.color is not None:
        _require_column(dataframe, chart_spec.color)

    if chart_type == "bar":
        _require_xy(dataframe, chart_spec)
        _require_numeric(dataframe, chart_spec.y)
        # Prevent bar chart if x is a continuous numeric column with > 10 unique values
        x_col = chart_spec.x
        if (
            x_col is not None
            and is_numeric_dtype(dataframe[x_col])
            and not is_bool_dtype(dataframe[x_col])
        ):
            unique_count = int(dataframe[x_col].nunique(dropna=True))
            if unique_count > 10:
                raise ChartSpecValidationError(
                    f"Bar chart x-axis '{x_col}' is a continuous numeric column with {unique_count} unique values. "
                    "For continuous numeric vs numeric variables, please use 'scatter' or 'line' instead."
                )
    elif chart_type == "line":
        _require_xy(dataframe, chart_spec)
        _require_numeric(dataframe, chart_spec.y)
    elif chart_type == "histogram":
        _require_named_column(dataframe, chart_spec.x, "x")
        _require_numeric(dataframe, chart_spec.x)
        if chart_spec.bins is not None and (
            chart_spec.bins < 1 or chart_spec.bins > 100
        ):
            raise ChartSpecValidationError("'bins' must be between 1 and 100.")
    elif chart_type == "scatter":
        _require_xy(dataframe, chart_spec)
        _require_numeric(dataframe, chart_spec.x)
        _require_numeric(dataframe, chart_spec.y)
    elif chart_type == "correlation_heatmap":
        columns = chart_spec.columns or _numeric_columns(dataframe)
        for column in columns:
            _require_column(dataframe, column)
            _require_numeric(dataframe, column)
        if len(columns) < 2:
            raise ChartSpecValidationError(
                "Correlation heatmap requires at least two numeric columns."
            )
        normalized["columns"] = columns
    elif chart_type == "box":
        _require_named_column(dataframe, chart_spec.y, "y")
        _require_numeric(dataframe, chart_spec.y)
        if chart_spec.x is not None:
            _require_column(dataframe, chart_spec.x)
    elif chart_type == "pie":
        names = chart_spec.names or chart_spec.x
        if names is None:
            raise ChartSpecValidationError("'names' is required for pie charts.")
        _require_column(dataframe, names)
        if int(dataframe[names].nunique(dropna=True)) > 10:
            raise ChartSpecValidationError(
                "Pie chart is allowed only for categories with 10 or fewer values."
            )
        values = chart_spec.values or chart_spec.y
        if values is not None:
            _require_column(dataframe, values)
            _require_numeric(dataframe, values)
        normalized = {**normalized, "names": names, "values": values}
        normalized.pop("x", None)
        normalized.pop("y", None)

    return normalized


def _require_xy(dataframe: pd.DataFrame, chart_spec: ChartSpec) -> None:
    _require_named_column(dataframe, chart_spec.x, "x")
    _require_named_column(dataframe, chart_spec.y, "y")


def _require_named_column(
    dataframe: pd.DataFrame, column: str | None, field_name: str
) -> None:
    if not isinstance(column, str) or not column.strip():
        raise ChartSpecValidationError(f"'{field_name}' is required.")
    _require_column(dataframe, column)


def _require_column(dataframe: pd.DataFrame, column: str) -> None:
    if column not in dataframe.columns:
        raise ChartSpecValidationError(f"Column '{column}' does not exist.")


def _require_numeric(dataframe: pd.DataFrame, column: str | None) -> None:
    if column is None:
        raise ChartSpecValidationError("Numeric column is required.")
    if not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column]):
        raise ChartSpecValidationError(f"Column '{column}' must be numeric.")


def _numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in dataframe.select_dtypes(include="number").columns
        if not is_bool_dtype(dataframe[column])
    ]
