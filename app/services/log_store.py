from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from redis import Redis


@dataclass(frozen=True)
class LogCache:
    """Кэш логов в виде DataFrame для bootstrap датасетов."""

    df: pl.DataFrame


def _empty_log_df() -> pl.DataFrame:
    """Создать пустой DataFrame с правильной схемой логов."""
    return pl.DataFrame({
        "timestamp": [],
        "eid": [],
        "user_id": [],
        "item_id": [],
        "session_id": [],
    })


def _validate_log_df(df: pl.DataFrame) -> pl.DataFrame:
    """Проверить, что DataFrame содержит обязательные колонки."""
    required = {"timestamp", "eid", "user_id", "item_id", "session_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Log data is missing columns: {sorted(missing)}")
    return df


def load_logs(path: str | None = None, df: pl.DataFrame | None = None) -> LogCache:
    """Загрузить логи из parquet или принять готовый DataFrame."""
    if df is not None:
        return LogCache(_validate_log_df(df))

    if not path:
        return LogCache(_empty_log_df())

    p = Path(path)
    if not p.is_file():
        return LogCache(_empty_log_df())

    df = pl.read_parquet(p)
    return LogCache(_validate_log_df(df))


class RedisLogStore:
    """Хранилище сырых логов в Redis list."""

    def __init__(self, redis: Redis, max_events: int = 100000, key_prefix: str = "events") -> None:
        self._redis = redis
        self._max_events = max(0, max_events)
        self._events_key = f"{key_prefix}:all"

    def add_event(
        self,
        session_id: str,
        item_id: int,
        eid: int,
        timestamp: int | None = None,
        user_id: int | None = None,
    ) -> int:
        """Добавить событие в общий лог и вернуть timestamp."""
        event_ts = timestamp or int(time.time() * 1000)
        payload = {
            "timestamp": event_ts,
            "session_id": session_id,
            "item_id": item_id,
            "eid": eid,
        }
        if user_id is not None:
            payload["user_id"] = user_id
        self._redis.lpush(self._events_key, json.dumps(payload))
        if self._max_events > 0:
            self._redis.ltrim(self._events_key, 0, self._max_events - 1)

        return event_ts

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Вернуть последние события из общего лога."""
        if limit <= 0:
            return []
        values = self._redis.lrange(self._events_key, 0, limit - 1)
        return [json.loads(value) for value in values]
