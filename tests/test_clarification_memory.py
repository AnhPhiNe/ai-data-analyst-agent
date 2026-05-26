import pandas as pd
from backend.agent.clarification_memory import (
    column_options,
    set_pending_from_question,
    set_pending_from_tool_call,
    try_resolve_pending_clarification,
)
from backend.services.session_store import DatasetSession, session_store
from backend.schemas import ChatResponse


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


def test_column_options() -> None:
    session = _create_session()
    all_cols = column_options(session)
    assert all_cols == ["department", "salary", "tenure_years"]

    numeric_cols = column_options(session, numeric_only=True)
    assert numeric_cols == ["salary", "tenure_years"]


def test_set_pending_from_question() -> None:
    session = _create_session()
    # Question with aggregate intent, missing columns
    set_pending_from_question(session, "Tính trung bình theo nhóm", "Ban muon tinh gi?")

    assert session.pending_clarification is not None
    assert session.pending_clarification["intent"] == "aggregate_metric"
    assert session.pending_clarification["operation"] == "mean"
    assert (
        session.pending_clarification["original_question"]
        == "Tính trung bình theo nhóm"
    )


def test_set_pending_from_tool_call() -> None:
    session = _create_session()
    # Tool validation failure for aggregate_metric
    set_pending_from_tool_call(
        session,
        "Tính trung bình theo nhóm",
        "aggregate_metric",
        {"operation": "mean"},
        "Thiếu tham số",
    )

    assert session.pending_clarification is not None
    assert session.pending_clarification["intent"] == "aggregate_metric"
    assert session.pending_clarification["operation"] == "mean"


def test_try_resolve_pending_clarification_no_pending() -> None:
    session = _create_session()
    traces = []

    def mock_exec(*args):
        return ChatResponse(
            session_id="1", answer="Executed", response_type="table", tool_trace=[]
        )

    res = try_resolve_pending_clarification(session, "salary", traces, mock_exec)
    assert res is None


def test_try_resolve_pending_clarification_new_standalone_intent() -> None:
    session = _create_session()
    set_pending_from_question(session, "Tính trung bình theo nhóm", "Ban muon tinh gi?")
    traces = []

    def mock_exec(*args):
        return ChatResponse(
            session_id="1", answer="Executed", response_type="table", tool_trace=[]
        )

    # Ask for chart instead (new standalone intent)
    res = try_resolve_pending_clarification(
        session, "Vẽ biểu đồ phân phối salary", traces, mock_exec
    )
    assert res is None
    assert session.pending_clarification is None  # Cleared!


def test_try_resolve_pending_clarification_successful_aggregate_resolution() -> None:
    session = _create_session()
    set_pending_from_question(session, "Tính trung bình theo nhóm", "Ban muon tinh gi?")
    traces = []

    executed_args = None

    def mock_exec(session, question, tool_name, arguments, traces, source):
        nonlocal executed_args
        executed_args = (tool_name, arguments)
        return ChatResponse(
            session_id="1", answer="Executed", response_type="table", tool_trace=traces
        )

    # Supply "salary" and "department" in follow-up
    res = try_resolve_pending_clarification(
        session, "salary và department", traces, mock_exec
    )

    assert res is not None
    assert executed_args is not None
    assert executed_args[0] == "aggregate_metric"
    assert executed_args[1]["metric_column"] == "salary"
    assert executed_args[1]["group_by"] == "department"
    assert executed_args[1]["operation"] == "mean"
    assert any(t.status == "resolved" for t in traces)


def test_try_resolve_pending_clarification_partial_aggregate_resolution() -> None:
    session = _create_session()
    set_pending_from_question(session, "Tính trung bình theo nhóm", "Ban muon tinh gi?")
    traces = []

    def mock_exec(*args):
        return ChatResponse(
            session_id="1", answer="Executed", response_type="table", tool_trace=[]
        )

    # Supply only "salary" (missing group_by)
    res = try_resolve_pending_clarification(session, "salary", traces, mock_exec)

    assert res is not None
    assert res.response_type == "clarification"
    # Should still be pending clarification
    assert session.pending_clarification is not None
    assert session.pending_clarification["metric_column"] == "salary"
    assert session.pending_clarification["group_by"] is None
