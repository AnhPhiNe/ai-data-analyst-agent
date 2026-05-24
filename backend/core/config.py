from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="AI Data Analyst Agent", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash-lite", alias="GEMINI_MODEL")
    max_upload_mb: int = Field(default=10, alias="MAX_UPLOAD_MB")


@lru_cache
def get_settings() -> Settings:
    return Settings()
