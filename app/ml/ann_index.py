from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ItemEmbeddingStore:
    """Таблица эмбеддингов айтемов: загружается из parquet или заполняется детерминированными заглушками.

    Формат parquet: колонки item_id (i64) + embedding (list/array) или emb_0..emb_{D-1} (f32).
    Все векторы L2-нормированы — косинусное сходство вычисляется как скалярное произведение.
    """

    def __init__(
        self,
        emb_path: str | None,
        emb_dim: int = 768,
        item_ids: list[int] | None = None,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        """Загружает эмбеддинги из parquet (локально или из S3), опционально фильтруя по item_ids.

        Использует scan_parquet чтобы не читать весь файл в память — критично для файлов >1GB.
        storage_options передаётся в polars для чтения из S3 (s3://...).
        При ошибке оставляет таблицу пустой до вызова _setup_index().
        """
        self._dim = emb_dim
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        self._vecs: np.ndarray = np.zeros((0, emb_dim), dtype=np.float32)
        if emb_path:
            try:
                import polars as pl

                scan = pl.scan_parquet(emb_path, storage_options=storage_options)
                if item_ids is not None:
                    scan = scan.filter(pl.col("item_id").is_in(item_ids))
                df = scan.collect()
                self._ids = df["item_id"].to_numpy().astype(np.int64)
                if "embedding" in df.columns:
                    vecs = np.stack(df["embedding"].to_numpy()).astype(np.float32)
                else:
                    emb_cols = sorted(c for c in df.columns if c.startswith("emb_"))
                    vecs = df.select(emb_cols).to_numpy().astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                self._vecs = vecs / np.maximum(norms, 1e-8)
                self._dim = self._vecs.shape[1]
                logger.info(
                    "Loaded %d embeddings (dim=%d) from %s",
                    len(self._ids),
                    self._dim,
                    emb_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load embeddings from %s: %s — falling back to dummy",
                    emb_path,
                    e,
                )

    def _setup_index(self, item_ids: list[int]) -> None:
        """Заполняет таблицу детерминированными случайными векторами (seed=42).

        Вызывается при старте если parquet недоступен. No-op если parquet уже загружен.
        """
        if len(self._ids) > 0:
            return
        if hasattr(self, "_id_to_idx_cache"):
            del self._id_to_idx_cache
        self._ids = np.array(item_ids, dtype=np.int64)
        raw = np.random.default_rng(42).standard_normal((len(item_ids), self._dim)).astype(np.float32)
        self._vecs = raw / np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)

    @property
    def _id_to_idx(self) -> dict[int, int]:
        if not hasattr(self, "_id_to_idx_cache"):
            self._id_to_idx_cache: dict[int, int] = {int(iid): i for i, iid in enumerate(self._ids)}
        return self._id_to_idx_cache

    @property
    def dim(self) -> int:
        return self._dim

    def get_embs_batch(self, item_ids: list[int]) -> np.ndarray:
        """Возвращает матрицу эмбеддингов (n, D) float32. Для неизвестных айтемов — нулевой вектор."""
        mapping = self._id_to_idx
        out = np.zeros((len(item_ids), self._dim), dtype=np.float32)
        for row, iid in enumerate(item_ids):
            idx = mapping.get(int(iid), -1)
            if idx >= 0:
                out[row] = self._vecs[idx]
        return out
