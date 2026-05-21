from __future__ import annotations

import json
from typing import Any

from redis import Redis


class RedisSessionStore:
    """Индекс истории сессий по ключу (user_id, session_id)."""

    def __init__(self, redis: Redis, ttl_seconds: int = 86400) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    def _key(self, user_id: int, session_id: str) -> str:
        """Сформировать ключ Redis для конкретной сессии."""
        return f"session:{user_id}:{session_id}"

    def get_session(self, user_id: int, session_id: str) -> list[dict[str, Any]]:
        """Получить историю событий конкретной сессии."""
        key = self._key(user_id, session_id)
        values = self._redis.lrange(key, 0, -1)
        history = [json.loads(value) for value in values]
        return history

    def add_event(self, user_id: int, session_id: str, item_id: int, eid: int) -> None:
        """Добавить событие в историю сессии и обновить TTL."""
        key = self._key(user_id, session_id)
        payload = {"user_id": user_id, "item_id": item_id, "eid": eid}
        self._redis.rpush(key, json.dumps(payload))
        self._redis.expire(key, self._ttl)
