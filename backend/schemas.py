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
