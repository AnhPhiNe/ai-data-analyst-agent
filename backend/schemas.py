from pydantic import BaseModel


class DatasetUploadResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict[str, object]]


class ErrorResponse(BaseModel):
    detail: str


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_percent: float


class NumericSummary(BaseModel):
    column: str
    count: int
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None


class ValueCountItem(BaseModel):
    value: str
    count: int
    percent: float


class TopCategory(BaseModel):
    column: str
    values: list[ValueCountItem]


class DistributionSpec(BaseModel):
    chart_type: str
    column: str
    x_label: str
    y_label: str
    data: list[dict[str, object]]


class DatasetProfileResponse(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: list[str]
    preview: list[dict[str, object]]
    dtypes: list[ColumnProfile]
    missing_values: list[ColumnProfile]
    numeric_summary: list[NumericSummary]
    top_categories: list[TopCategory]
    distributions: list[DistributionSpec]
