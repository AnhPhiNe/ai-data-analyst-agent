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
