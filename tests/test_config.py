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
