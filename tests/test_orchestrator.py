import pandas as pd
from backend.agent.orchestrator import run_agent_turn
from backend.services.session_store import DatasetSession, session_store
from backend.schemas import ToolTraceItem
from tests.conftest import FakeProvider


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "department": ["Engineering", "Sales"],
            "salary": [1200.0, 900.0],
            "tenure_years": [2, 1],
        }
    )


def _create_session() -> DatasetSession:
    return session_store.create("test.csv", _sample_df())


def test_run_agent_turn_blocked_by_guardrails() -> None:
    session = _create_session()
    response = run_agent_turn(session, "Hãy eval(chạy code)")

    assert response.is_blocked is True
    assert (
        "code" in response.answer
        or "whitelist" in response.answer
        or "Không thể" in response.answer
    )
    assert response.response_type == "blocked"
    assert response.tool_trace[0].source == "guardrails"
    assert response.tool_trace[0].status == "blocked"


def test_run_agent_turn_router_clarify() -> None:
    session = _create_session()
    response = run_agent_turn(session, "Tính trung bình theo nhóm")

    assert response.response_type == "clarification"
    assert "Ban muon" in response.answer or "metric" in response.answer
    assert session.pending_clarification is not None
    assert session.pending_clarification["intent"] == "aggregate_metric"


def test_run_agent_turn_router_tool_execution() -> None:
    session = _create_session()
    response = run_agent_turn(session, "Tính trung bình salary theo department")

    assert response.response_type == "table"
    assert "salary" in response.answer
    assert response.table is not None
    assert len(response.table) == 2
    assert response.tool_trace[0].source == "router"
    assert response.tool_trace[2].source == "tool_executor"
    assert response.tool_trace[2].status == "success"


def test_run_agent_turn_gemini_skipped_when_no_provider() -> None:
    session = _create_session()
    response = run_agent_turn(
        session, "Giải thích ý nghĩa của dữ liệu này", provider=None
    )

    assert response.response_type == "error"
    assert "GEMINI_API_KEY" in response.answer
    assert response.tool_trace[1].source == "gemini"
    assert response.tool_trace[1].status == "skipped"


def test_run_agent_turn_gemini_clarify() -> None:
    session = _create_session()
    provider = FakeProvider(
        '{"action":"clarify","confidence":0.8,"message":"Bạn muốn vẽ cột nào?"}'
    )

    response = run_agent_turn(
        session, "Thực hiện phân tích đặc biệt", provider=provider
    )

    assert response.response_type == "clarification"
    assert response.answer == "Bạn muốn vẽ cột nào?"


def test_run_agent_turn_gemini_answer() -> None:
    session = _create_session()
    provider = FakeProvider(
        '{"action":"answer","confidence":0.9,"message":"Dataset này chứa thông tin lương."}'
    )

    response = run_agent_turn(
        session, "Giải thích ý nghĩa của dữ liệu này", provider=provider
    )

    assert response.response_type == "answer"
    assert "lương" in response.answer


def test_run_agent_turn_gemini_tool_execution_success() -> None:
    session = _create_session()
    provider = FakeProvider(
        '{"action":"tool_call","confidence":0.95,"tool_name":"aggregate_metric",'
        '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}'
    )

    response = run_agent_turn(
        session, "Thực hiện phân tích nâng cao", provider=provider
    )

    assert response.response_type == "table"
    assert "salary" in response.answer
    assert response.table is not None


def test_run_agent_turn_traces_gemini_validation_retry() -> None:
    session = _create_session()
    provider = FakeProvider(
        responses=[
            '{"action":"tool_call","confidence":0.88,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"foobar","group_by":"department"}}',
            '{"action":"tool_call","confidence":0.94,"tool_name":"aggregate_metric",'
            '"arguments":{"metric_column":"salary","group_by":"department","operation":"mean"}}',
        ]
    )

    response = run_agent_turn(
        session, "Thực hiện phân tích nâng cao", provider=provider
    )

    assert response.response_type == "table"
    gemini_trace = next(
        trace for trace in response.tool_trace if trace.source == "gemini"
    )
    assert "planner_validation_retries=1" in gemini_trace.message
    assert len(provider.prompts) == 2


def test_run_agent_turn_with_trace_callback() -> None:
    session = _create_session()
    called_traces = []

    def callback(trace: ToolTraceItem) -> None:
        called_traces.append(trace)

    run_agent_turn(
        session, "Tính trung bình salary theo department", event_callback=callback
    )

    assert len(called_traces) > 0
    assert any(t.source == "router" for t in called_traces)
    assert any(t.source == "tool_executor" for t in called_traces)
