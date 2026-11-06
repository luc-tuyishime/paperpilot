"""Application settings, read from environment variables (or a .env file).

Every setting has a safe default so the app runs with zero configuration.
In Docker or Kubernetes we override them with environment variables,
never by editing code.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="PAPERPILOT_")

    app_name: str = "PaperPilot"
    environment: str = "local"  # local | ci | production
    version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it (cached), so .env is read only one time."""
    return Settings()
