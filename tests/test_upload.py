from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.session_store import session_store


def test_upload_csv_returns_session_and_preview() -> None:
    client = TestClient(app)
    csv_content = b"department,salary\nEngineering,1200\nSales,900\n"

    response = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "hr.csv"
    assert payload["rows"] == 2
    assert payload["columns"] == 2
    assert payload["column_names"] == ["department", "salary"]
    assert payload["preview"][0]["department"] == "Engineering"
    assert session_store.get(payload["session_id"]) is not None


def test_upload_keeps_quoted_single_column_csv_as_one_column() -> None:
    client = TestClient(app)
    csv_content = (
        '"user_id,department,salary,note"\n'
        '"u1,Engineering,1000,note_1"\n'
        '"u2,Engineering,1100,note_2"\n'
    ).encode("utf-8")

    response = client.post(
        "/datasets/upload",
        files={"file": ("quoted_rows.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["rows"] == 2
    assert payload["columns"] == 1
    assert payload["column_names"] == ["user_id,department,salary,note"]


def test_upload_uses_default_csv_parser_without_delimiter_guessing() -> None:
    client = TestClient(app)
    csv_content = b"user_id;department;salary\nu1;Engineering;1000\n"

    response = client.post(
        "/datasets/upload",
        files={"file": ("semicolon.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["columns"] == 1
    assert payload["column_names"] == ["user_id;department;salary"]


def test_upload_xlsx_returns_session() -> None:
    client = TestClient(app)
    buffer = BytesIO()
    dataframe = pd.DataFrame(
        {
            "class_name": ["A1", "A2"],
            "student_score": [8.5, 7.8],
        }
    )
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False)

    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "students.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"] == "students.xlsx"
    assert payload["rows"] == 2
    assert payload["columns"] == 2


def test_upload_keeps_single_column_xlsx_as_one_column() -> None:
    client = TestClient(app)
    buffer = BytesIO()
    dataframe = pd.DataFrame(
        {
            "user_id,department,salary,note,coef": [
                "u1,Engineering,1000,note_1,2",
                "u2,Engineering,1100,note_2,3.5",
                "u3,Engineering,1200,note_3,4",
            ]
        }
    )
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False)

    response = client.post(
        "/datasets/upload",
        files={
            "file": (
                "single_column_rows.xlsx",
                buffer.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["rows"] == 3
    assert payload["columns"] == 1
    assert payload["column_names"] == ["user_id,department,salary,note,coef"]


def test_upload_rejects_unsupported_file_type() -> None:
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_empty_file() -> None:
    client = TestClient(app)

    response = client.post(
        "/datasets/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
