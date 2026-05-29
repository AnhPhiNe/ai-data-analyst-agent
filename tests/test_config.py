from backend.core.config import Settings


def test_settings_parses_allowed_origins() -> None:
    settings = Settings(
        ALLOWED_ORIGINS="https://app.streamlit.app, https://api.example.com"
    )

    assert settings.cors_allowed_origins() == [
        "https://app.streamlit.app",
        "https://api.example.com",
    ]


def test_settings_returns_empty_origins_when_unset() -> None:
    settings = Settings(ALLOWED_ORIGINS="")

    assert settings.cors_allowed_origins() == []


def test_settings_reads_planner_validation_retries() -> None:
    settings = Settings(MAX_PLANNER_VALIDATION_RETRIES=2)

    assert settings.max_planner_validation_retries == 2


def test_settings_defaults_to_gemini_flash_lite_model(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert settings.gemini_model == "gemini-2.5-flash-lite"
    assert settings.groq_model == "llama-3.3-70b-versatile"
