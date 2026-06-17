from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel

from app.core.env import load_project_env

load_project_env()


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def item_image_key(item_id: int) -> str:
    """S3 object key: images/NNN/{item_id}.jpg."""
    return f"images/{item_id % 1000:03d}/{item_id}.jpg"


def item_image_url(item_id: int, image_base_url: str | None) -> str:
    """Public URL for an item image (S3/CDN public base)."""
    key = item_image_key(item_id)
    if image_base_url:
        return f"{image_base_url.rstrip('/')}/{key}"
    return f"/{key}"


class Settings(BaseModel):
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    session_ttl_seconds: int = int(os.getenv("SESSION_TTL_SECONDS", "86400"))
    default_feed_limit: int = int(os.getenv("DEFAULT_FEED_LIMIT", "5"))

    # cap number of RS groups loaded into memory (0 = no limit)
    feed_max_groups: int = int(os.getenv("FEED_MAX_GROUPS", "2000"))
    emb_dim: int = int(os.getenv("EMB_DIM", "768"))

    # --- Local data directory (fallback when S3 is not configured) ---
    # Place files as: data/feed.parquet, data/item_embeddings.parquet,
    # data/item_catalog.parquet, data/models/reranker_catboost.cbm, etc.
    local_data_dir: str = os.getenv("LOCAL_DATA_DIR", "./data")

    # --- S3 (takes priority over local files when configured) ---
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "https://s3.ru1.storage.beget.cloud")
    s3_bucket: Optional[str] = os.getenv("S3_BUCKET")
    s3_access_key_id: Optional[str] = os.getenv("S3_ACCESS_KEY_ID")
    s3_secret_access_key: Optional[str] = os.getenv("S3_SECRET_ACCESS_KEY")
    s3_region: str = os.getenv("S3_REGION", "ru-1")
    s3_presign_ttl_seconds: int = int(os.getenv("S3_PRESIGN_TTL_SECONDS", "3600"))

    # S3 object keys for the data artifacts (within s3_bucket)
    feed_s3_key: Optional[str] = os.getenv("FEED_S3_KEY")
    emb_s3_key: Optional[str] = os.getenv("EMB_S3_KEY")
    reranker_s3_key: Optional[str] = os.getenv("RERANKER_S3_KEY") or "models/reranker_catboost.cbm"

    # --- ANN mixing (OPQ+IVFPQ index built offline, loaded from S3 into memory) ---
    faiss_index_s3_key: Optional[str] = os.getenv("FAISS_INDEX_S3_KEY") or "models/faiss_ivfpq.index"
    faiss_ids_s3_key: Optional[str] = os.getenv("FAISS_IDS_S3_KEY") or "models/faiss_item_ids.npy"
    catalog_s3_key: Optional[str] = os.getenv("CATALOG_S3_KEY") or "item_catalog.parquet"

    ann_enabled: bool = _bool_env("ANN_ENABLED", "1")
    ann_nprobe: int = int(os.getenv("ANN_NPROBE", "32"))
    ann_pool_n: int = int(os.getenv("ANN_POOL_N", "400"))
    # размер "ближнего" окна соседей: exploit берётся из cand[:ann_exploit_k],
    # explore — из хвоста cand[ann_exploit_k:]
    ann_exploit_k: int = int(os.getenv("ANN_EXPLOIT_K", "6"))
    # каденс подмешивания: каждая ann_every-я карточка отдаётся как ANN
    # (тип exploit/explore чередуется), при условии что есть лайки и индекс доступен
    ann_every: int = int(os.getenv("ANN_EVERY", "4"))
    ann_recent_likes: int = int(os.getenv("ANN_RECENT_LIKES", "5"))

    # Public base URL for item images (optional; otherwise presigned S3 URLs are used)
    image_base_url: Optional[str] = os.getenv("IMAGE_BASE_URL") or None

    def _s3_uri(self, key: str | None) -> Optional[str]:
        if not (self.s3_bucket and key):
            return None
        return f"s3://{self.s3_bucket}/{key}"

    def _local(self, *parts: str) -> Optional[str]:
        """Returns the local path if the file exists, else None."""
        p = os.path.join(self.local_data_dir, *parts)
        return p if os.path.exists(p) else None

    @property
    def feed_uri(self) -> Optional[str]:
        return self._s3_uri(self.feed_s3_key) or self._local("feed.parquet")

    @property
    def emb_uri(self) -> Optional[str]:
        return self._s3_uri(self.emb_s3_key) or self._local("item_embeddings.parquet")

    @property
    def catalog_uri(self) -> Optional[str]:
        return self._s3_uri(self.catalog_s3_key) or self._local("item_catalog.parquet")

    def s3_storage_options(self) -> dict[str, str]:
        """storage_options for polars/object_store reads from the custom S3 endpoint."""
        opts: dict[str, str] = {
            "aws_endpoint_url": self.s3_endpoint_url,
            "aws_region": self.s3_region,
            "aws_virtual_hosted_style_request": "false",
        }
        if self.s3_access_key_id:
            opts["aws_access_key_id"] = self.s3_access_key_id
        if self.s3_secret_access_key:
            opts["aws_secret_access_key"] = self.s3_secret_access_key
        return opts
