from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import polars as pl
from redis import Redis


@dataclass(frozen=True)
class Event:
    """Событие для онлайн-обновления датасетов."""

    timestamp: int
    session_id: str
    item_id: int
    eid: int
    user_id: int | None = None


class EventSink(Protocol):
    """Интерфейс обработчика событий датасета."""

    def on_event(self, event: Event) -> None: ...


class DatasetHub:
    """Хаб, который рассылает события и запускает bootstrap датасетов."""

    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self._sinks = list(sinks)

    def publish(self, event: Event) -> None:
        """Отправить событие всем датасетам."""
        for sink in self._sinks:
            sink.on_event(event)

    def bootstrap(self, df: pl.DataFrame, reset: bool = True) -> None:
        """Построить датасеты из исторических логов."""
        for sink in self._sinks:
            bootstrap = getattr(sink, "bootstrap", None)
            if callable(bootstrap):
                bootstrap(df, reset=reset)


class PopularityIndex:
    """Индекс популярности на базе Redis ZSET."""

    def __init__(self, redis: Redis, key_prefix: str = "popularity") -> None:
        self._redis = redis
        self._all_key = f"{key_prefix}:all"
        self._eid_key_tpl = f"{key_prefix}:eid:{{eid}}"

    def _eid_key(self, eid: str | int) -> str:
        """Сформировать ключ ZSET для конкретного eid."""
        return self._eid_key_tpl.format(eid=str(eid))

    def bootstrap(self, df: pl.DataFrame, reset: bool = True) -> None:
        """Построить индекс популярности из логов."""
        if df.is_empty() or "item_id" not in df.columns:
            return

        if reset:
            keys = [self._all_key]
            if "eid" in df.columns:
                eids = df.select("eid").drop_nulls().unique().to_series().to_list()
                for eid in eids:
                    keys.append(self._eid_key(eid))
            self._redis.delete(*keys)

        counts = df.group_by("item_id").len().rename({"len": "cnt"})
        eid_counts = None
        if "eid" in df.columns:
            eid_counts = df.group_by(["eid", "item_id"]).len().rename({"len": "cnt"})

        pipe = self._redis.pipeline()
        for row in counts.iter_rows(named=True):
            pipe.zincrby(self._all_key, float(row["cnt"]), str(int(row["item_id"])))

        if eid_counts is not None:
            for row in eid_counts.iter_rows(named=True):
                eid = row["eid"]
                if eid is None:
                    continue
                pipe.zincrby(
                    self._eid_key(eid),
                    float(row["cnt"]),
                    str(int(row["item_id"])),
                )

        pipe.execute()

    def on_event(self, event: Event) -> None:
        """Обновить популярность по одному событию."""
        member = str(event.item_id)
        self._redis.zincrby(self._all_key, 1.0, member)
        self._redis.zincrby(self._eid_key(event.eid), 1.0, member)

    def get_top_items(self, limit: int, eid: str | int | None = None) -> list[int]:
        """Вернуть топ популярных item_id."""
        if limit <= 0:
            return []
        key = self._all_key if eid is None else self._eid_key(eid)
        values = self._redis.zrevrange(key, 0, limit - 1)
        return [int(value) for value in values]
