from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from secrets import token_urlsafe
from threading import RLock
from typing import Any
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetSession:
    session_id: str
    access_token: str
    filename: str
    dataframe: pd.DataFrame
    chat_history: list[dict[str, object]] = field(default_factory=list)
    pending_clarification: dict[str, object] | None = None
    profile_cache: dict[str, object] | None = None
    suggestions_cache: Any | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        self._lock = RLock()
        self._ttl_seconds = 3600
        self._max_sessions = 25

    def configure(self, ttl_seconds: int, max_sessions: int) -> None:
        self._ttl_seconds = max(60, int(ttl_seconds))
        self._max_sessions = max(1, int(max_sessions))

    def create(self, filename: str, dataframe: pd.DataFrame) -> DatasetSession:
        now = datetime.now(timezone.utc)
        session = DatasetSession(
            session_id=str(uuid4()),
            access_token=token_urlsafe(24),
            filename=filename,
            dataframe=dataframe,
            created_at=now,
            last_accessed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )
        with self._lock:
            self._cleanup_expired_locked(now)
            self._evict_if_needed_locked()
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> DatasetSession | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._cleanup_expired_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.last_accessed_at = now
            session.expires_at = now + timedelta(seconds=self._ttl_seconds)
            return session

    def verify_access(self, session: DatasetSession, token: str | None, required: bool) -> bool:
        if not required:
            return True
        return bool(token) and token == session.access_token

    def add_chat_turn(self, session_id: str, question: str, answer: str, route: str) -> None:
        session = self.get(session_id)
        if session is None:
            return
        with self._lock:
            session.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "route": route,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )

    def set_pending_clarification(self, session_id: str, pending: dict[str, object] | None) -> None:
        session = self.get(session_id)
        if session is None:
            return
        with self._lock:
            session.pending_clarification = pending

    def clear_pending_clarification(self, session_id: str) -> None:
        self.set_pending_clarification(session_id, None)

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def _cleanup_expired_locked(self, now: datetime) -> None:
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    def _evict_if_needed_locked(self) -> None:
        while len(self._sessions) >= self._max_sessions:
            oldest_session_id = min(
                self._sessions,
                key=lambda session_id: self._sessions[session_id].last_accessed_at,
            )
            self._sessions.pop(oldest_session_id, None)


session_store = InMemorySessionStore()
