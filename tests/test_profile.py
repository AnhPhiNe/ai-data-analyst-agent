from fastapi.testclient import TestClient

from backend.main import app


def _upload_dataset(client: TestClient) -> str:
    csv_content = (
        "department,salary,tenure_years,performance_score\n"
        "Engineering,1200,2,4.5\n"
        "Sales,900,1,3.8\n"
        "Engineering,1500,5,4.9\n"
        "HR,,3,4.1\n"
    ).encode("utf-8")

    response = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_profile_endpoint_returns_dataset_profile() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.get(f"/datasets/{session_id}/profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["rows"] == 4
    assert payload["columns"] == 4
    assert payload["column_names"] == [
        "department",
        "salary",
        "tenure_years",
        "performance_score",
    ]
    assert payload["preview"][0]["department"] == "Engineering"


def test_profile_includes_dtypes_and_missing_values() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    payload = client.get(f"/datasets/{session_id}/profile").json()

    salary_profile = next(
        column for column in payload["dtypes"] if column["name"] == "salary"
    )
    assert salary_profile["missing_count"] == 1
    assert salary_profile["missing_percent"] == 25.0
    assert payload["missing_values"] == [salary_profile]


def test_profile_includes_numeric_summary_and_top_categories() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    payload = client.get(f"/datasets/{session_id}/profile").json()

    salary_summary = next(
        summary
        for summary in payload["numeric_summary"]
        if summary["column"] == "salary"
    )
    assert salary_summary["count"] == 3
    assert salary_summary["mean"] == 1200.0

    department_categories = next(
        item for item in payload["top_categories"] if item["column"] == "department"
    )
    assert department_categories["values"][0] == {
        "value": "Engineering",
        "count": 2,
        "percent": 50.0,
    }


def test_profile_includes_distribution_specs() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    payload = client.get(f"/datasets/{session_id}/profile").json()

    salary_distribution = next(
        spec for spec in payload["distributions"] if spec["column"] == "salary"
    )
    assert salary_distribution["chart_type"] == "histogram"
    assert salary_distribution["y_label"] == "Count"
    assert salary_distribution["data"]

    department_distribution = next(
        spec for spec in payload["distributions"] if spec["column"] == "department"
    )
    assert department_distribution["chart_type"] == "bar"
    assert department_distribution["data"][0] == {"category": "Engineering", "count": 2}


def test_profile_includes_column_metadata() -> None:
    client = TestClient(app)
    csv_content = (
        "user_id,joined_at,is_active,score,segment\n"
        "u1,2026-01-01,true,80,A\n"
        "u2,2026-01-02,false,90,B\n"
        "u3,2026-01-03,true,85,A\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("metadata.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]

    payload = client.get(f"/datasets/{session_id}/profile").json()
    metadata = {column["name"]: column for column in payload["column_metadata"]}

    assert metadata["user_id"]["inferred_kind"] == "id_like"
    assert metadata["joined_at"]["inferred_kind"] == "datetime_like"
    assert metadata["is_active"]["inferred_kind"] == "boolean"
    assert metadata["score"]["inferred_kind"] == "numeric"
    assert metadata["segment"]["inferred_kind"] == "categorical"
    assert metadata["segment"]["unique_count"] == 2
    assert metadata["segment"]["sample_values"] == ["A", "B", "A"]


def test_profile_returns_404_for_unknown_session() -> None:
    client = TestClient(app)

    response = client.get("/datasets/missing-session/profile")

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset session not found."
