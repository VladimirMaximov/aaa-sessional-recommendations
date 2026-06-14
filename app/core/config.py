from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _default_data_path(filename: str) -> Optional[str]:
    p = os.path.join("data", filename)
    return p if os.path.isfile(p) else None


class Settings(BaseModel):
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
    # item_catalog.parquet (item_id, title, image_path) preferred; falls back to ranking parquet
    catalog_path: Optional[str] = (
        os.getenv("CATALOG_PATH")
        or _default_data_path("item_catalog.parquet")
        or _default_data_path("train/train.parquet")
    )
    catalog_size: int = int(os.getenv("CATALOG_SIZE", "50000"))
    default_feed_limit: int = int(os.getenv("DEFAULT_FEED_LIMIT", "5"))
    # test split used for the demo feed; val is kept for notebook-only evaluation
    feed_path: Optional[str] = (
        os.getenv("FEED_PATH")
        or _default_data_path("test/test.parquet")
        or _default_data_path("val/val.parquet")
    )
    # cap number of RS groups loaded into memory (0 = no limit); test has 14k groups vs val 1.3k
    feed_max_groups: int = int(os.getenv("FEED_MAX_GROUPS", "2000"))
    # item embeddings: parquet with item_id + emb_* columns; falls back to dummy table
    emb_path: Optional[str] = os.getenv("EMB_PATH") or _default_data_path("item_embeddings.parquet")
    emb_dim: int = int(os.getenv("EMB_DIM", "768"))
    ann_neighbors: int = int(os.getenv("ANN_NEIGHBORS", "50"))
    reranker_path: Optional[str] = os.getenv("RERANKER_PATH") or _default_data_path("models/reranker_catboost.cbm")
