from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    dataframe: pd.DataFrame
    chat_history: list[dict[str, object]] = field(default_factory=list)
    pending_clarification: dict[str, object] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}

    def create(self, filename: str, dataframe: pd.DataFrame) -> DatasetSession:
        session = DatasetSession(
            session_id=str(uuid4()),
            filename=filename,
            dataframe=dataframe,
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> DatasetSession | None:
        return self._sessions.get(session_id)

    def add_chat_turn(self, session_id: str, question: str, answer: str, route: str) -> None:
        session = self.get(session_id)
        if session is None:
            return
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
        session.pending_clarification = pending

    def clear_pending_clarification(self, session_id: str) -> None:
        self.set_pending_clarification(session_id, None)

    def clear(self) -> None:
        self._sessions.clear()


session_store = InMemorySessionStore()
