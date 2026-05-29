from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = Field(default="AI Data Analyst Agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", alias="GEMINI_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")
    max_rows: int = Field(default=100_000, alias="MAX_ROWS")
    max_columns: int = Field(default=200, alias="MAX_COLUMNS")
    max_sessions: int = Field(default=25, alias="MAX_SESSIONS")
    session_ttl_seconds: int = Field(default=3600, alias="SESSION_TTL_SECONDS")
    require_session_token: bool = Field(default=False, alias="REQUIRE_SESSION_TOKEN")
    allowed_origins: str = Field(default="", alias="ALLOWED_ORIGINS")
    rate_limit_per_minute: int = Field(default=300, alias="RATE_LIMIT_PER_MINUTE")
    rate_limit_window_seconds: int = Field(
        default=60, alias="RATE_LIMIT_WINDOW_SECONDS"
    )
    max_planner_validation_retries: int = Field(
        default=1, alias="MAX_PLANNER_VALIDATION_RETRIES"
    )

    def cors_allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
