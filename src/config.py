"""Application configuration loaded from environment variables via pydantic-settings."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Values are resolved from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- LLM Provider --
    deepseek_api_key: str = Field(default="sk-placeholder", description="API key for the LLM provider")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="OpenAI-compatible API base URL",
    )
    deepseek_model: str = Field(default="deepseek-reasoner", description="Model for reasoning tasks")
    deepseek_coder_model: str = Field(default="deepseek-coder", description="Model for code tasks")
    deepseek_max_tokens: int = Field(default=8192, description="Maximum tokens for completions")
    deepseek_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")

    # -- Databases --
    source_db_url: str = Field(
        default="postgresql://user:password@localhost:5432/source_db",
        description="Source database connection string",
    )
    target_db_url: str = Field(
        default="postgresql://user:password@localhost:5432/warehouse_db",
        description="Target warehouse connection string",
    )
    db_pool_size: int = Field(default=10, ge=1, description="Connection pool size")

    # -- Pipeline --
    max_retry_attempts: int = Field(default=3, ge=1, description="Max auto-fix retries")
    log_level: str = Field(default="INFO", description="Logging level")
    output_dir: Path = Field(default=Path("./output"), description="Directory for generated artifacts")

    # -- Airflow --
    airflow_dags_dir: Path = Field(default=Path("./dags"), description="Directory for generated DAGs")

    # -- Data Validation --
    max_null_rate: float = Field(default=0.05, ge=0.0, le=1.0, description="Max acceptable null rate")
    max_distribution_shift: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Max acceptable KS statistic"
    )
    row_count_tolerance: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Row count tolerance fraction"
    )


def get_settings() -> Settings:
    """Return a cached Settings instance (re-created each call so env changes are picked up)."""
    return Settings()
