from __future__ import annotations

import random
from typing import Protocol, Sequence

from app.ml.base import BaseCandGen, BaseRanker, UserHistory


class PopularitySource(Protocol):
    """Источник популярности для топ-кандидатов."""

    def get_top_items(self, limit: int, eid: str | int | None = None) -> list[int]: ...


class RandomCandGen(BaseCandGen):
    """Случайный генератор кандидатов из каталога."""

    def __init__(self, catalog_ids: Sequence[int], n_candidates: int) -> None:
        self._catalog_ids = list(catalog_ids)
        self._n_candidates = max(0, n_candidates)

    def get_candidates(self, user_history: UserHistory) -> list[int]:
        """Сэмплировать кандидатов без учета истории."""
        if not self._catalog_ids or self._n_candidates <= 0:
            return []

        k = min(self._n_candidates, len(self._catalog_ids))
        if k == len(self._catalog_ids):
            candidates = self._catalog_ids.copy()
            random.shuffle(candidates)
            return candidates

        return random.sample(self._catalog_ids, k)


class RandomRanker(BaseRanker):
    """Ранжировщик, случайно перемешивающий кандидатов."""

    def rank(self, user_history: UserHistory, candidates: Sequence[int]) -> list[int]:
        """Перемешать кандидатов случайным образом."""
        ranked = list(candidates)
        random.shuffle(ranked)
        return ranked


class TopPopularCandGen(BaseCandGen):
    """Генератор кандидатов из топа популярности."""

    def __init__(self, source: PopularitySource, n_candidates: int, eid: str | int | None = None) -> None:
        self._source = source
        self._n_candidates = max(0, n_candidates)
        self._eid = eid

    def get_candidates(self, user_history: UserHistory) -> list[int]:
        """Вернуть топ популярных кандидатов."""
        if self._n_candidates <= 0:
            return []
        return self._source.get_top_items(self._n_candidates, eid=self._eid)
