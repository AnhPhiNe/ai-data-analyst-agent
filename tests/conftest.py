import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.session_store import session_store


class FakeProvider:
    """Mock LLM provider for unit and integration testing."""

    def __init__(
        self,
        responses: list[str] | str | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        if isinstance(responses, str):
            self.responses = [responses]
        else:
            self.responses = responses or []
        self.errors = errors or []
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        if not self.responses:
            return "Fake fallback response"
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def clear_session_store() -> None:
    """Automatically clears the global session store before every test run."""
    session_store.clear()


@pytest.fixture
def client() -> TestClient:
    """Standard TestClient fixture pointing to the FastAPI app."""
    return TestClient(app)
