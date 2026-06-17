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

    def _score_cache_key(self, user_id: int, session_id: str) -> str:
        return f"session_scores:{user_id}:{session_id}"

    def get_score_cache(self, user_id: int, session_id: str) -> dict[str, Any] | None:
        """Кэш EMA-скоров: scores, processed_events, pool_item_ids, group_x, mode."""
        raw = self._redis.get(self._score_cache_key(user_id, session_id))
        if not raw:
            return None
        data = json.loads(raw)
        scores = data.get("scores", {})
        data["scores"] = {int(k): float(v) for k, v in scores.items()}
        data["pool_item_ids"] = [int(i) for i in data.get("pool_item_ids", [])]
        return data

    def set_score_cache(
        self,
        user_id: int,
        session_id: str,
        *,
        scores: dict[int, float],
        processed_events: int,
        pool_item_ids: list[int],
        group_x: str,
        mode: str,
    ) -> None:
        payload = {
            "scores": {str(k): v for k, v in scores.items()},
            "processed_events": processed_events,
            "pool_item_ids": pool_item_ids,
            "group_x": group_x,
            "mode": mode,
        }
        key = self._score_cache_key(user_id, session_id)
        self._redis.set(key, json.dumps(payload), ex=self._ttl)

    def clear_score_cache(self, user_id: int, session_id: str) -> None:
        self._redis.delete(self._score_cache_key(user_id, session_id))
