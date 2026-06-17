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


class FaissANNIndex:
    """Сжатый FAISS-индекс (OPQ+IVFPQ) по полному каталогу для подмешивания соседей.

    Десериализуется из байтов S3 без записи на диск. Индекс хранит коды PQ
    (~64 байта/вектор), поэтому 2.9M айтемов занимают ~0.25 GB RAM.
    Вектора в индексе L2-нормированы → inner product = cosine.
    """

    def __init__(
        self,
        index_blob: bytes | None = None,
        ids_blob: bytes | None = None,
        nprobe: int = 32,
    ) -> None:
        self._index = None
        self._ids: np.ndarray = np.array([], dtype=np.int64)
        if index_blob is None or ids_blob is None:
            return
        try:
            import io

            import faiss

            self._index = faiss.deserialize_index(
                np.frombuffer(index_blob, dtype="uint8")
            )
            self._ids = np.load(io.BytesIO(ids_blob))
            try:
                self._index.nprobe = nprobe
            except Exception:
                pass
            logger.info(
                "FaissANNIndex loaded (ntotal=%d, ids=%d, nprobe=%d)",
                self._index.ntotal,
                len(self._ids),
                nprobe,
            )
        except Exception as e:
            logger.warning("Failed to load FAISS ANN index: %s — ANN mixing disabled", e)
            self._index = None
            self._ids = np.array([], dtype=np.int64)

    @property
    def available(self) -> bool:
        return self._index is not None and len(self._ids) > 0

    def query(
        self,
        query_vec: np.ndarray,
        top_n: int,
        exclude_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Top-N соседей для query_vec (item_id, score), с фильтрацией exclude_ids.

        query_vec должен быть L2-нормирован (одно пространство с индексом).
        """
        if not self.available:
            return []
        exclude = exclude_ids or set()
        q = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1)
        sims, idxs = self._index.search(q, top_n)
        out: list[tuple[int, float]] = []
        for score, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            iid = int(self._ids[idx])
            if iid in exclude:
                continue
            out.append((iid, float(score)))
        return out
