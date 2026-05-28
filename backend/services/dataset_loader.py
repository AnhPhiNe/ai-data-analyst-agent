from io import BytesIO, StringIO
from pathlib import Path
import csv

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
        raise DatasetLoadError(
            "Unsupported file type. Please upload a CSV or XLSX file."
        )
    if not content:
        raise DatasetLoadError("Uploaded file is empty.")
    if len(content) > max_bytes:
        raise DatasetLoadError(
            f"File is too large. Maximum upload size is {max_upload_mb} MB."
        )


def load_dataframe(
    filename: str,
    content: bytes,
    max_upload_mb: int,
    max_rows: int | None = None,
    max_columns: int | None = None,
) -> pd.DataFrame:
    validate_upload(filename=filename, content=content, max_upload_mb=max_upload_mb)

    suffix = Path(filename).suffix.lower()
    buffer = BytesIO(content)

    try:
        if suffix == ".csv":
            dataframe = read_csv_content(content)
        else:
            dataframe = pd.read_excel(buffer, engine="openpyxl")
    except EmptyDataError as exc:
        raise DatasetLoadError("Uploaded file has no readable rows.") from exc
    except (ParserError, UnicodeDecodeError, ValueError) as exc:
        raise DatasetLoadError(
            "Could not read the uploaded dataset. Please check the file format."
        ) from exc

    if dataframe.empty:
        raise DatasetLoadError("Uploaded dataset is empty.")
    if len(dataframe.columns) == 0:
        raise DatasetLoadError("Uploaded dataset has no columns.")
    if max_rows is not None and len(dataframe) > max_rows:
        raise DatasetLoadError(
            f"Uploaded dataset has too many rows. Maximum is {max_rows}."
        )
    if max_columns is not None and len(dataframe.columns) > max_columns:
        raise DatasetLoadError(
            f"Uploaded dataset has too many columns. Maximum is {max_columns}."
        )

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    if any(not column for column in dataframe.columns):
        raise DatasetLoadError("Uploaded dataset contains an empty column name.")

    return dataframe


def read_csv_content(content: bytes) -> pd.DataFrame:
    dataframe = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    fallback = _reparse_single_column_delimited_csv(dataframe)
    return fallback if fallback is not None else dataframe


def _reparse_single_column_delimited_csv(
    dataframe: pd.DataFrame,
) -> pd.DataFrame | None:
    if len(dataframe.columns) != 1:
        return None

    header = str(dataframe.columns[0])
    delimiter = _detect_delimiter(header)
    if delimiter is None:
        return None

    first_column = dataframe.iloc[:, 0].dropna().astype(str)
    expected_fields = header.count(delimiter) + 1
    if expected_fields <= 1:
        return None
    if not first_column.empty and not all(
        value.count(delimiter) + 1 == expected_fields for value in first_column
    ):
        return None

    rows = [header, *first_column.tolist()]
    parsed_rows = list(csv.reader(rows, delimiter=delimiter, quotechar="\0"))
    if not parsed_rows or any(len(row) != expected_fields for row in parsed_rows):
        return None

    columns = [column.strip() for column in parsed_rows[0]]
    if any(not column for column in columns):
        return None
    repaired_content = "\n".join(rows)
    return pd.read_csv(StringIO(repaired_content), sep=delimiter)


def _detect_delimiter(header: str) -> str | None:
    candidates = [",", ";", "\t", "|"]
    counts = {delimiter: header.count(delimiter) for delimiter in candidates}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    return delimiter if count > 0 else None
