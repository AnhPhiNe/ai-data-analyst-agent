from io import BytesIO
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


class DatasetLoadError(ValueError):
    """Raised when an uploaded dataset cannot be loaded safely."""


def validate_upload(filename: str, content: bytes, max_upload_mb: int) -> None:
    suffix = Path(filename or "").suffix.lower()
    max_bytes = max_upload_mb * 1024 * 1024

    if not filename:
        raise DatasetLoadError("Missing filename.")
    if suffix not in ALLOWED_EXTENSIONS:
        raise DatasetLoadError("Unsupported file type. Please upload a CSV or XLSX file.")
    if not content:
        raise DatasetLoadError("Uploaded file is empty.")
    if len(content) > max_bytes:
        raise DatasetLoadError(f"File is too large. Maximum upload size is {max_upload_mb} MB.")


def load_dataframe(filename: str, content: bytes, max_upload_mb: int) -> pd.DataFrame:
    validate_upload(filename=filename, content=content, max_upload_mb=max_upload_mb)

    suffix = Path(filename).suffix.lower()
    buffer = BytesIO(content)

    try:
        if suffix == ".csv":
            dataframe = pd.read_csv(buffer)
        else:
            dataframe = pd.read_excel(buffer, engine="openpyxl")
    except EmptyDataError as exc:
        raise DatasetLoadError("Uploaded file has no readable rows.") from exc
    except (ParserError, UnicodeDecodeError, ValueError) as exc:
        raise DatasetLoadError("Could not read the uploaded dataset. Please check the file format.") from exc

    if dataframe.empty:
        raise DatasetLoadError("Uploaded dataset is empty.")
    if len(dataframe.columns) == 0:
        raise DatasetLoadError("Uploaded dataset has no columns.")

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if any(not column for column in dataframe.columns):
        raise DatasetLoadError("Uploaded dataset contains an empty column name.")

    return dataframe
