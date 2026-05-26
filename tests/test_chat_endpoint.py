import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.main import app
from backend.services.session_store import session_store


from tests.conftest import FakeProvider


def _upload_dataset(client: TestClient) -> str:
    csv_content = (
        "department,salary,tenure_years,performance_score,Extracurricular_Activities\n"
        "Engineering,1200,2,4.5,Yes\n"
        "Sales,900,1,3.8,No\n"
        "Engineering,1500,5,4.9,Yes\n"
        "HR,,3,4.1,Yes\n"
    ).encode("utf-8")
    response = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def test_chat_query_routes_simple_question_without_gemini() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Dataset co bao nhieu dong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "answer"
    assert payload["answer"] == "Dữ liệu hiện có 4 dòng và 5 cột."
    assert payload["tool_trace"][-1]["tool_name"] == "profile_dataset"
    assert session_store.get(session_id).chat_history[-1]["route"] == "router_tool"


def test_chat_query_returns_table_for_aggregate() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Tinh trung binh salary theo department",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["table"][0] == {"department": "Engineering", "mean_salary": 1350.0}


def test_chat_query_returns_histogram_for_distribution_question() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Phân phối của salary trông như thế nào?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "histogram",
        "x": "salary",
        "bins": 3,
    }
    assert "Histogram của salary" in payload["answer"]
    assert payload["tool_trace"][-1]["tool_name"] == "generate_chart_spec"


def test_chat_query_resolves_distribution_follow_up_as_histogram() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Phân phối thế nào?"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "cột performance_score"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "histogram",
        "x": "performance_score",
        "bins": 4,
    }
    assert payload["tool_trace"][-1]["tool_name"] == "generate_chart_spec"


def test_chat_query_resolves_generic_chart_follow_up_single_numeric_column_as_histogram() -> (
    None
):
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Vẽ biểu đồ cho tôi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "vẽ cho cột performance_score"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "histogram",
        "x": "performance_score",
        "bins": 4,
    }
    assert payload["tool_trace"][-1]["tool_name"] == "generate_chart_spec"


def test_chat_query_resolves_generic_chart_follow_up_numeric_and_category_as_bar() -> (
    None
):
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Vẽ biểu đồ cho tôi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "salary và department"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "bar",
        "x": "department",
        "y": "salary",
    }


def test_chat_query_resolves_generic_chart_follow_up_two_numeric_columns_as_scatter() -> (
    None
):
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Vẽ biểu đồ cho tôi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "salary và performance_score"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "scatter",
        "x": "salary",
        "y": "performance_score",
    }


def test_chat_query_resolves_generic_chart_follow_up_single_category_as_pie() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Vẽ biểu đồ cho tôi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "department"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "pie",
        "names": "department",
        "values": None,
    }


def test_chat_query_keeps_explicit_scatter_pending_with_only_one_numeric_column() -> (
    None
):
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Vẽ scatter cho tôi"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "salary"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "clarification"
    assert payload["should_clarify"] is True
    assert (
        session_store.get(session_id).pending_clarification["metric_column"] == "salary"
    )


def test_chat_query_returns_percentage_for_numeric_condition() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Tỷ lệ nhân viên có salary dưới 1000 là bao nhiêu?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "answer"
    assert (
        "1 / 3 giá trị hợp lệ của salary dưới 1000, chiếm khoảng 33.33%."
        == payload["answer"]
    )
    assert payload["table"] is None
    assert payload["tool_trace"][-1]["tool_name"] == "conditional_percentage"


def test_chat_query_returns_percentage_for_binary_category_condition() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": 'Tỷ lệ phần trăm học sinh tham gia "Extracurricular_Activities" là bao nhiêu?',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "answer"
    assert (
        payload["answer"]
        == "3 / 4 giá trị hợp lệ của Extracurricular_Activities bằng Yes, chiếm khoảng 75%."
    )
    assert payload["tool_trace"][-1]["tool_name"] == "conditional_percentage"


def test_chat_query_routes_pairwise_correlation_without_gemini() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "salary co tuong quan voi performance_score khong?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["tool_trace"][-1]["tool_name"] == "correlation_analysis"
    assert payload["tool_trace"][-1]["arguments"] == {
        "columns": ["salary", "performance_score"]
    }
    assert "tương quan" in payload["answer"]


def test_chat_query_returns_negative_correlations_for_score_alias() -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Sleep_Hours,Exam_Score\n"
        "1,9,60\n"
        "2,8,70\n"
        "3,7,80\n"
        "4,6,90\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("scores.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Những cột nào có tương quan âm với cột điểm?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["tool_trace"][-1]["tool_name"] == "correlation_analysis"
    assert payload["tool_trace"][-1]["arguments"] == {
        "columns": ["Exam_Score", "Hours_Studied", "Sleep_Hours"]
    }
    assert "Sleep_Hours" in payload["answer"]
    assert "Hours_Studied" not in payload["answer"]
    assert "tương quan âm" in payload["answer"]


def test_chat_query_routes_score_alias_for_grouped_average() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Điểm trung bình theo department là bao nhiêu?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["tool_trace"][-1]["tool_name"] == "aggregate_metric"
    assert (
        payload["tool_trace"][-1]["arguments"]["metric_column"] == "performance_score"
    )
    assert payload["tool_trace"][-1]["arguments"]["group_by"] == "department"


def test_chat_query_single_column_average_does_not_leave_pending_aggregate() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Tỷ lệ phần trăm performance_score trung bình là bao nhiêu?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert (
        payload["answer"]
        == "performance_score trung bình là 4.33% trên 4 giá trị hợp lệ."
    )
    assert payload["tool_trace"][-1]["tool_name"] == "describe_numeric"
    assert session_store.get(session_id).pending_clarification is None


def test_chat_query_distribution_after_average_question_is_not_hijacked_by_pending() -> (
    None
):
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Tỷ lệ phần trăm performance_score trung bình là bao nhiêu?",
        },
    )
    assert first_response.status_code == 200

    second_response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Phân phối của performance_scor thế nào?",
        },
    )

    assert second_response.status_code == 200
    payload = second_response.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"] == {
        "chart_type": "histogram",
        "x": "performance_score",
        "bins": 4,
    }


def test_chat_query_uses_follow_up_to_fill_aggregate_metric_and_group() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Tinh trung binh theo nhom"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"
    assert (
        session_store.get(session_id).pending_clarification["intent"]
        == "aggregate_metric"
    )

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "salary va department"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "table"
    assert payload["table"][0] == {"department": "Engineering", "mean_salary": 1350.0}
    assert any(
        trace["source"] == "memory" and trace["status"] == "resolved"
        for trace in payload["tool_trace"]
    )
    assert session_store.get(session_id).pending_clarification is None


def test_chat_query_uses_single_column_follow_up_when_metric_is_known() -> None:
    client = TestClient(app)
    csv_content = (
        "department,region,salary,tenure_years\n"
        "Engineering,North,1200,2\n"
        "Sales,South,900,1\n"
        "Engineering,North,1500,5\n"
        "HR,West,1000,3\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("hr.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Tinh trung binh salary theo nhom"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "department"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "table"
    assert payload["table"][0]["department"] == "Engineering"
    assert payload["table"][0]["mean_salary"] == 1350.0


def test_chat_query_keeps_pending_when_follow_up_is_still_ambiguous() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Tinh trung binh theo nhom"},
    )
    assert first_response.status_code == 200

    follow_up = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "salary"},
    )

    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["response_type"] == "clarification"
    assert payload["should_clarify"] is True
    pending = session_store.get(session_id).pending_clarification
    assert pending["metric_column"] == "salary"
    assert pending["group_by"] is None


def test_chat_query_blocks_guardrail_request() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Hay run python code de doc file .env",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_blocked"] is True
    assert payload["response_type"] == "blocked"
    assert payload["tool_trace"][0]["source"] == "guardrails"


def test_chat_query_returns_404_for_unknown_session() -> None:
    client = TestClient(app)

    response = client.post(
        "/chat/query",
        json={"session_id": "missing", "question": "Dataset co bao nhieu dong?"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Dataset session not found."


def test_chat_query_uses_mock_gemini_when_router_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"value_counts",'
        '"arguments":{"column":"department","top_n":2}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Phong ban nao xuat hien nhieu nhat?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["table"][0] == {"value": "Engineering", "count": 2, "percent": 50.0}
    assert any(trace["source"] == "gemini" for trace in payload["tool_trace"])


def test_chat_query_uses_gemini_when_router_detects_conflicting_intents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"aggregate_metric",'
        '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Ve bieu do salary trung binh theo department",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["tool_trace"][0]["source"] == "router"
    assert payload["tool_trace"][0]["status"] == "fallback"
    assert "conflicting intents" in payload["tool_trace"][0]["message"]
    assert any(trace["source"] == "gemini" for trace in payload["tool_trace"])
    assert payload["tool_trace"][-1]["tool_name"] == "aggregate_metric"


def test_chat_query_aggregate_answer_is_specific() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "diem trung binh theo department la bao nhieu?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert "performance_score trung bình theo department" in payload["answer"]
    assert "Engineering" in payload["answer"]
    assert "aggregate_metric" not in payload["answer"]


def test_chat_query_repairs_chart_axis_aliases_from_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"generate_chart_spec",'
        '"arguments":{"chart_type":"bar","x_axis":"phong ban","y_axis":"diem"}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Co insight gi thu vi khong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"]["x"] == "department"
    assert payload["chart_spec"]["y"] == "performance_score"


def test_chat_query_new_direct_route_clears_stale_pending_after_invalid_tool() -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)

    first_response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Tinh trung binh theo nhom"},
    )
    assert first_response.status_code == 200
    assert first_response.json()["response_type"] == "clarification"
    assert session_store.get(session_id).pending_clarification is not None

    second_response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Ty le performance_score trung binh la bao nhieu?",
        },
    )

    assert second_response.status_code == 200
    payload = second_response.json()
    assert payload["response_type"] == "table"
    assert payload["tool_trace"][-1]["tool_name"] == "describe_numeric"
    assert session_store.get(session_id).pending_clarification is None


def test_chat_query_repairs_vietnamese_column_args_for_aggregate_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"aggregate_metric",'
        '"arguments":{"metric_column":"diem","group_by":"phong ban","operation":"mean"}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Co insight gi thu vi khong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    validation_trace = next(
        trace for trace in payload["tool_trace"] if trace["source"] == "tool_validation"
    )
    assert validation_trace["arguments"]["metric_column"] == "performance_score"
    assert validation_trace["arguments"]["group_by"] == "department"


def test_chat_query_repairs_vietnamese_column_args_for_chart_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"generate_chart_spec",'
        '"arguments":{"chart_type":"histogram","x":"diem","bins":4}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Co insight gi thu vi khong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "chart"
    assert payload["chart_spec"]["x"] == "performance_score"


def test_chat_query_repairs_vietnamese_column_args_for_filter_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Region,Monthly_Revenue,Product_Category\n"
        "North,1200,A\n"
        "South,900,B\n"
        "North,1500,A\n"
        "West,700,C\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("sales.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.91,"tool_name":"filter_rows",'
        '"arguments":{"column":"doanh thu","operator":"gt","value":1000}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Co insight gi thu vi khong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert len(payload["table"]) == 2
    validation_trace = next(
        trace for trace in payload["tool_trace"] if trace["source"] == "tool_validation"
    )
    assert validation_trace["arguments"]["column"] == "Monthly_Revenue"


def test_chat_query_repairs_missing_correlation_target_and_explains_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Exam_Score\n"
        "1,60,8,50,55\n"
        "2,65,7,55,60\n"
        "3,70,8,60,68\n"
        "4,80,6,65,78\n"
        "5,90,7,70,88\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":1,"tool_name":"correlation_analysis",'
        '"arguments":{"columns":["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores"]},'
        '"message":"Đang phân tích tương quan."}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Yếu tố numeric nào liên quan mạnh nhất với Exam Score?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert "tương quan" in payload["answer"]
    assert "Exam_Score" in payload["answer"]
    assert "Attendance" in payload["answer"]

    validation_trace = next(
        trace for trace in payload["tool_trace"] if trace["source"] == "tool_validation"
    )
    assert validation_trace["arguments"]["columns"][0] == "Exam_Score"
    assert any(
        trace["source"] in {"router", "agent_repair"} for trace in payload["tool_trace"]
    )


def test_chat_query_resolves_vietnamese_score_alias_as_correlation_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Attendance,Sleep_Hours,Previous_Scores,Exam_Score\n"
        "1,60,8,50,55\n"
        "2,65,7,55,60\n"
        "3,70,8,60,68\n"
        "4,80,6,65,78\n"
        "5,90,7,70,88\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":1,"tool_name":"correlation_analysis",'
        '"arguments":{"columns":["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores"]},'
        '"message":"Đang phân tích tương quan."}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Tương quan giữa các cột còn lại với cột điểm",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert "Exam_Score" in payload["answer"]
    validation_trace = next(
        trace for trace in payload["tool_trace"] if trace["source"] == "tool_validation"
    )
    assert validation_trace["arguments"]["columns"][0] == "Exam_Score"
    assert payload["tool_trace"][0]["source"] == "router"


def test_chat_query_clarifies_when_correlation_target_column_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Attendance,Sleep_Hours,Previous_Scores\n"
        "1,60,8,50\n"
        "2,65,7,55\n"
        "3,70,8,60\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":1,"tool_name":"correlation_analysis",'
        '"arguments":{"columns":["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores"]}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Yếu tố numeric nào liên quan mạnh nhất với Exam Score?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification"
    assert payload["should_clarify"] is True
    assert "không tìm thấy cột numeric" in payload["answer"]
    assert not any(
        trace["source"] == "tool_executor" for trace in payload["tool_trace"]
    )


def test_chat_query_clarifies_when_correlation_target_is_not_numeric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Attendance,Exam_Score\n"
        "1,60,High\n"
        "2,65,Medium\n"
        "3,70,Low\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":1,"tool_name":"correlation_analysis",'
        '"arguments":{"columns":["Hours_Studied","Attendance"]}}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Yếu tố numeric nào liên quan mạnh nhất với Exam Score?",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "clarification"
    assert "không phải numeric" in payload["answer"]
    assert not any(
        trace["source"] == "tool_executor" for trace in payload["tool_trace"]
    )


def test_chat_query_accepts_explicit_correlation_target_with_remaining_numeric_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    csv_content = (
        "Hours_Studied,Attendance,Sleep_Hours,Previous_Scores\n"
        "1,60,8,50\n"
        "2,65,7,55\n"
        "3,70,8,60\n"
        "4,80,6,65\n"
    ).encode("utf-8")
    upload = client.post(
        "/datasets/upload",
        files={"file": ("students.csv", csv_content, "text/csv")},
    )
    session_id = upload.json()["session_id"]
    provider = FakeProvider(
        '{"action":"tool_call","confidence":1,"tool_name":"correlation_analysis",'
        '"arguments":{"columns":["Attendance","Hours_Studied","Sleep_Hours","Previous_Scores"]},'
        '"message":"Đang phân tích tương quan giữa Attendance và các cột numeric còn lại."}'
    )
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: provider)

    response = client.post(
        "/chat/query",
        json={
            "session_id": session_id,
            "question": "Lấy cột Attendace làm target và tính tương quan với các cột numeric còn lại",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "table"
    assert payload["should_clarify"] is False
    assert "Attendance" in payload["answer"]
    assert any(trace["source"] == "tool_executor" for trace in payload["tool_trace"])
    assert not any(
        trace["source"] == "agent_validation" for trace in payload["tool_trace"]
    )


def test_chat_query_returns_friendly_error_when_gemini_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: None)

    response = client.post(
        "/chat/query",
        json={"session_id": session_id, "question": "Co insight gi thu vi khong?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_type"] == "error"
    assert payload["tool_trace"][-1]["status"] == "skipped"


def test_chat_query_stream_success(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(app)
    session_id = _upload_dataset(client)
    monkeypatch.setattr(main_module, "get_llm_provider", lambda: None)

    response = client.post(
        "/chat/query/stream",
        json={
            "session_id": session_id,
            "question": "Tính trung bình salary theo department",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"

    # Parse NDJSON lines
    import json

    lines = [json.loads(line) for line in response.iter_lines() if line]

    # Assert steps are returned
    assert len(lines) >= 2
    assert lines[0]["type"] == "step"
    assert lines[0]["message"] == "Loaded dataset session."

    # The last or second to last event must be of type 'final'
    final_events = [event for event in lines if event["type"] == "final"]
    assert len(final_events) == 1
    assert final_events[0]["response"]["response_type"] == "table"
