from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
            image_url = str(row.get("image_url") or f"https://via.placeholder.com/600x400?text=Item+{item_id}")
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


def _build_mock_df(size: int) -> pl.DataFrame:
    """Сгенерировать моковый каталог заданного размера."""
    item_ids = list(range(1, size + 1))
    titles = [f"Item {item_id}" for item_id in item_ids]
    image_urls = [f"https://via.placeholder.com/600x400?text=Item+{item_id}" for item_id in item_ids]
    return pl.DataFrame({
        "item_id": item_ids,
        "title": titles,
        "image_url": image_urls,
    })


def load_catalog(path: str | None, mock_size: int = 500) -> CatalogCache:
    """Загрузить каталог из parquet или создать моковый."""
    df: pl.DataFrame
    if path:
        p = Path(path)
        if p.is_file():
            df = pl.read_parquet(p)
        else:
            df = _build_mock_df(mock_size)
    else:
        df = _build_mock_df(mock_size)

    if "title" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("title"))
    if "image_url" not in df.columns:
        df = df.with_columns(pl.lit(None).alias("image_url"))

    return CatalogCache(df)
