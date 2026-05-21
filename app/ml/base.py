from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

UserHistory = list[dict[str, Any]]


class BaseCandGen(ABC):
    """Интерфейс генератора кандидатов."""

    @abstractmethod
    def get_candidates(self, user_history: UserHistory) -> list[int]:
        """Вернуть список кандидатов по истории пользователя."""
        raise NotImplementedError


class BaseRanker(ABC):
    """Интерфейс ранжировщика кандидатов."""

    @abstractmethod
    def rank(self, user_history: UserHistory, candidates: Sequence[int]) -> list[int]:
        """Отранжировать кандидатов с учетом истории пользователя."""
        raise NotImplementedError
