import threading
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from backend.services.session_store import InMemorySessionStore, DatasetSession


def _dummy_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"col": [1, 2]})


def test_session_creation_and_fields() -> None:
    store = InMemorySessionStore()
    df = _dummy_dataframe()
    session = store.create("test.csv", df)

    assert isinstance(session, DatasetSession)
    assert len(session.session_id) > 0
    assert len(session.access_token) > 0
    assert session.filename == "test.csv"
    assert session.dataframe.equals(df)
    assert session.chat_history == []
    assert session.pending_clarification is None
    assert session.profile_cache is None
    assert session.suggestions_cache is None
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.last_accessed_at, datetime)
    assert isinstance(session.expires_at, datetime)


def test_configure_constraints() -> None:
    store = InMemorySessionStore()
    store.configure(ttl_seconds=120, max_sessions=5)
    # create a session and verify TTL expiration is 120s
    session = store.create("test.csv", _dummy_dataframe())
    expected_expires = session.created_at + timedelta(seconds=120)
    assert abs((session.expires_at - expected_expires).total_seconds()) < 1.0


def test_get_refreshes_expiry() -> None:
    store = InMemorySessionStore()
    store.configure(ttl_seconds=300, max_sessions=25)
    session = store.create("test.csv", _dummy_dataframe())
    orig_expires = session.expires_at

    # Access after 1 second
    time.sleep(0.01)
    retrieved = store.get(session.session_id)
    assert retrieved is not None
    assert retrieved.expires_at > orig_expires


def test_verify_access() -> None:
    store = InMemorySessionStore()
    session = store.create("test.csv", _dummy_dataframe())

    # Required = False
    assert store.verify_access(session, None, required=False) is True
    assert store.verify_access(session, "wrong", required=False) is True

    # Required = True
    assert store.verify_access(session, session.access_token, required=True) is True
    assert store.verify_access(session, "wrong", required=True) is False
    assert store.verify_access(session, None, required=True) is False


def test_add_chat_turn_and_clarifications() -> None:
    store = InMemorySessionStore()
    session = store.create("test.csv", _dummy_dataframe())

    store.add_chat_turn(session.session_id, "Q1", "A1", "route1")
    assert len(session.chat_history) == 1
    assert session.chat_history[0]["question"] == "Q1"
    assert session.chat_history[0]["answer"] == "A1"
    assert session.chat_history[0]["route"] == "route1"

    pending = {"intent": "agg", "original_question": "Q1"}
    store.set_pending_clarification(session.session_id, pending)
    assert session.pending_clarification == pending

    store.clear_pending_clarification(session.session_id)
    assert session.pending_clarification is None


def test_cleanup_expired_sessions() -> None:
    store = InMemorySessionStore()
    store.configure(ttl_seconds=1, max_sessions=2)
    session = store.create("test.csv", _dummy_dataframe())

    # Manually expire the session
    session.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)

    # Get should return None and cleanup
    assert store.get(session.session_id) is None


def test_max_sessions_eviction() -> None:
    store = InMemorySessionStore()
    store.configure(ttl_seconds=300, max_sessions=3)

    s1 = store.create("1.csv", _dummy_dataframe())
    time.sleep(0.01)
    s2 = store.create("2.csv", _dummy_dataframe())
    time.sleep(0.01)
    s3 = store.create("3.csv", _dummy_dataframe())

    # Get s1 to make it recently accessed
    store.get(s1.session_id)

    # Create fourth session, which triggers eviction.
    # Oldest by last_accessed_at should be s2 (since s1 was accessed recently, s3 was created recently)
    s4 = store.create("4.csv", _dummy_dataframe())

    assert store.get(s1.session_id) is not None
    assert store.get(s3.session_id) is not None
    assert store.get(s4.session_id) is not None
    assert store.get(s2.session_id) is None  # evicted!


def test_thread_safety_stress() -> None:
    store = InMemorySessionStore()
    store.configure(ttl_seconds=300, max_sessions=100)

    session = store.create("stress.csv", _dummy_dataframe())
    session_id = session.session_id

    def worker():
        for i in range(50):
            store.add_chat_turn(session_id, f"Q{i}", f"A{i}", "route")
            store.set_pending_clarification(session_id, {"intent": "test"})
            store.get(session_id)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(session.chat_history) == 250
