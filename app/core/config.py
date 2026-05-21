from __future__ import annotations

import os

from pydantic import BaseModel


def _bool_env(name: str, default: str = "0") -> bool:
    """Прочитать булево значение из переменной окружения."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


class Settings(BaseModel):
    """Конфигурация приложения из переменных окружения."""

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
    catalog_path: str | None = os.getenv("CATALOG_PATH")
    # catalog_path: str | None = os.getenv("CATALOG_PATH", "data/items_000-001.parquet")
    catalog_size: int = int(os.getenv("CATALOG_SIZE", "500"))
    default_feed_limit: int = int(os.getenv("DEFAULT_FEED_LIMIT", "5"))
    events_max_length: int = int(os.getenv("EVENTS_MAX_LENGTH", "100000"))
    logs_path: str | None = os.getenv("LOGS_PATH")
    logs_bootstrap_reset: bool = _bool_env("LOGS_BOOTSTRAP_RESET", "1")
    candgen_strategy: str = os.getenv("CANDGEN_STRATEGY", "random")
    candgen_fallback: str | None = os.getenv("CANDGEN_FALLBACK", "random")
    ranker_strategy: str = os.getenv("RANKER_STRATEGY", "random")
    popular_eid: str | None = os.getenv("POPULAR_EID")
