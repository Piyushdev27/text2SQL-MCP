"""Application configuration management using Pydantic Settings.

Reads environment variables and local .env file securely, providing
type validation, sensible defaults, and masked credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path (where .env lives)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Central configuration for PostgreSQL, MCP server, and query execution."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH) if ENV_FILE_PATH.exists() else None,
        env_file_encoding="utf-8",
        env_prefix="TEXT2SQL_",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL Connection Settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host IP or domain")
    postgres_port: int = Field(default=5432, description="PostgreSQL TCP port")
    postgres_user: str = Field(default="postgres", description="Database username")
    postgres_password: Optional[SecretStr] = Field(default=None, description="Database user password")
    postgres_db: str = Field(default="lending_sample", description="Target database name")
    postgres_sslmode: str = Field(default="require", description="SSL mode (require, disable, prefer)")
    postgres_schema: str = Field(default="public", description="Default search schema")

    # Execution Bounds & Security Policy
    query_timeout_seconds: int = Field(default=60, description="Maximum execution timeout in seconds")
    max_row_limit: int = Field(default=100, description="Maximum rows returned by execute_readonly_sql")

    # Local Storage for Scratch & Ground-Truth Test Benchmarks
    local_db_path: str = Field(
        default=str(PROJECT_ROOT / "local_problem1.db"),
        description="Path to local database for local test benchmark tables",
    )

    def get_postgres_connection_dict(self) -> Dict[str, Any]:
        """Returns keyword arguments formatted for psycopg2.connect()."""
        kwargs: Dict[str, Any] = {
            "host": self.postgres_host,
            "port": self.postgres_port,
            "user": self.postgres_user,
            "dbname": self.postgres_db,
            "sslmode": self.postgres_sslmode,
            "connect_timeout": 15,
        }
        if self.postgres_password:
            kwargs["password"] = self.postgres_password.get_secret_value()
        return kwargs

    def get_masked_summary(self) -> Dict[str, Any]:
        """Returns a safe summary of configuration without revealing passwords."""
        return {
            "postgres_host": self.postgres_host,
            "postgres_port": self.postgres_port,
            "postgres_user": self.postgres_user,
            "postgres_db": self.postgres_db,
            "postgres_schema": self.postgres_schema,
            "postgres_sslmode": self.postgres_sslmode,
            "has_password": self.postgres_password is not None,
            "query_timeout_seconds": self.query_timeout_seconds,
            "max_row_limit": self.max_row_limit,
            "local_db_path": self.local_db_path,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached singleton instance of Settings."""
    return Settings()


def reset_settings() -> None:
    """Clears the settings cache (useful during testing)."""
    get_settings.cache_clear()
