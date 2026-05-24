from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    dataframe: pd.DataFrame
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

    def clear(self) -> None:
        self._sessions.clear()


session_store = InMemorySessionStore()
