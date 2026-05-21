from __future__ import annotations

import importlib
import inspect
from typing import Any, Callable

from app.core.config import Settings
from app.ml.base import BaseCandGen, BaseRanker
from app.ml.strategies import RandomCandGen, RandomRanker, TopPopularCandGen
from app.services.catalog import CatalogCache
from app.services.datasets import DatasetHub, PopularityIndex
from app.services.log_store import LogCache, RedisLogStore


class StrategyConfigError(ValueError):
    """Ошибка конфигурации стратегий."""

    pass


class StrategyRegistry:
    """Реестр для сборки candgen и ranker с нужными зависимостями."""

    def __init__(
        self,
        *,
        catalog: CatalogCache,
        log_store: RedisLogStore,
        log_cache: LogCache,
        data_hub: DatasetHub,
        popularity_index: PopularityIndex,
        settings: Settings,
    ) -> None:
        self._catalog = catalog
        self._log_store = log_store
        self._log_cache = log_cache
        self._data_hub = data_hub
        self._popularity_index = popularity_index
        self._settings = settings

    def build_candgen(self, name: str, *, n_candidates: int) -> BaseCandGen:
        """Создать candgen по имени или пути класса."""
        return _build_candgen(
            name,
            catalog=self._catalog,
            log_store=self._log_store,
            log_cache=self._log_cache,
            data_hub=self._data_hub,
            popularity_index=self._popularity_index,
            settings=self._settings,
            n_candidates=n_candidates,
        )

    def build_ranker(self, name: str) -> BaseRanker:
        """Создать ranker по имени или пути класса."""
        return _build_ranker(
            name,
            settings=self._settings,
            log_cache=self._log_cache,
            data_hub=self._data_hub,
            popularity_index=self._popularity_index,
        )


def _load_class(path: str) -> type:
    """Загрузить класс по строковому пути модуля."""
    if ":" in path:
        module_path, class_name = path.split(":", 1)
    else:
        if "." not in path:
            raise StrategyConfigError(f"Invalid class path: {path}")
        module_path, class_name = path.rsplit(".", 1)

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise StrategyConfigError(f"Class not found: {path}")
    return cls


def _instantiate(cls: type, **kwargs: Any) -> Any:
    """Инстанцировать класс, передавая только поддерживаемые аргументы."""
    signature = inspect.signature(cls)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return cls(**kwargs)

    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return cls(**accepted)


def _resolve_candgen(name: str) -> Callable[..., BaseCandGen]:
    """Получить класс candgen из реестра или по пути."""
    registry: dict[str, Callable[..., BaseCandGen]] = {
        "random": RandomCandGen,
        "popular": TopPopularCandGen,
    }
    lowered = name.lower().strip()
    if lowered in registry:
        return registry[lowered]
    return _load_class(name)


def _resolve_ranker(name: str) -> Callable[..., BaseRanker]:
    """Получить класс ranker из реестра или по пути."""
    registry: dict[str, Callable[..., BaseRanker]] = {
        "random": RandomRanker,
    }
    lowered = name.lower().strip()
    if lowered in registry:
        return registry[lowered]
    return _load_class(name)


def _build_candgen(
    name: str,
    *,
    catalog: CatalogCache,
    log_store: RedisLogStore,
    log_cache: LogCache,
    data_hub: DatasetHub,
    popularity_index: PopularityIndex,
    settings: Settings,
    n_candidates: int,
) -> BaseCandGen:
    """Собрать candgen с нужными зависимостями."""
    cls = _resolve_candgen(name)
    try:
        return _instantiate(
            cls,
            catalog=catalog,
            log_store=log_store,
            source=popularity_index,
            log_cache=log_cache,
            data_hub=data_hub,
            popularity_index=popularity_index,
            settings=settings,
            n_candidates=n_candidates,
            eid=settings.popular_eid,
            catalog_ids=catalog.item_ids,
        )
    except TypeError as exc:
        raise StrategyConfigError(f"Failed to init candgen: {name}") from exc


def _build_ranker(
    name: str,
    *,
    settings: Settings,
    log_cache: LogCache | None = None,
    data_hub: DatasetHub | None = None,
    popularity_index: PopularityIndex | None = None,
) -> BaseRanker:
    """Собрать ranker с нужными зависимостями."""
    cls = _resolve_ranker(name)
    try:
        return _instantiate(
            cls,
            settings=settings,
            log_cache=log_cache,
            data_hub=data_hub,
            popularity_index=popularity_index,
        )
    except TypeError as exc:
        raise StrategyConfigError(f"Failed to init ranker: {name}") from exc
