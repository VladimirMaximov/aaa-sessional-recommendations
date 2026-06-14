from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class CatalogItem:
    """Метаданные объявления, используемые в выдаче."""

    item_id: int
    title: str
    image_url: str


class CatalogCache:
    """Кэш каталога в памяти с быстрым доступом по item_id."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df
        self._by_id: dict[int, CatalogItem] = {}

        for row in df.iter_rows(named=True):
            item_id = int(row["item_id"])
            title = str(row.get("title") or f"Item {item_id}")
            image_url = str(row.get("image_url"))
            self._by_id[item_id] = CatalogItem(
                item_id=item_id,
                title=title,
                image_url=image_url,
            )

        self._ids = list(self._by_id.keys())

    @property
    def df(self) -> pl.DataFrame:
        """Вернуть исходный DataFrame каталога."""
        return self._df

    @property
    def item_ids(self) -> list[int]:
        """Вернуть список всех item_id в каталоге."""
        return self._ids

    def get_item(self, item_id: int) -> CatalogItem | None:
        """Получить метаданные объявления по item_id."""
        return self._by_id.get(item_id)
