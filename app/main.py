from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import polars as pl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis import Redis

from app.api.v1 import router as v1_router
from app.core.config import Settings
from app.ml.ann_index import FaissANNIndex, ItemEmbeddingStore
from app.ml.reranker import SessionReranker
from app.services.catalog import CatalogCache
from app.services.image_urls import ImageUrlService
from app.services.session_store import RedisSessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

settings = Settings()

_FEED_KEEP_COLS = ["x", "item_id", "title"] + [f"float_{i}" for i in range(1, 81)] + ["cat_1", "cat_2", "cat_3"]


def _load_feed_groups(
    feed_uri: str | None,
    storage_options: dict[str, str],
    max_groups: int = 0,
) -> dict[str, pl.DataFrame]:
    if not feed_uri:
        raise RuntimeError("FEED_S3_KEY / S3_BUCKET are not set — cannot load feed from S3")
    avail = pl.read_parquet(feed_uri, n_rows=0, storage_options=storage_options).columns
    cols = [c for c in _FEED_KEEP_COLS if c in avail]
    df = pl.read_parquet(feed_uri, columns=cols, storage_options=storage_options)
    groups = {str(x): grp for (x,), grp in df.group_by(["x"], maintain_order=True)}
    if max_groups > 0 and len(groups) > max_groups:
        keys = list(groups)[:max_groups]
        groups = {k: groups[k] for k in keys}
    return groups


def _build_catalog(
    catalog_uri: str | None,
    storage_options: dict[str, str],
    feed_groups: dict[str, pl.DataFrame],
) -> CatalogCache:
    """Каталог тайтлов: предпочитаем полный item_catalog.parquet из S3 (нужен для ANN-айтемов
    вне фид-групп), при недоступности — фолбэк на тайтлы из фид-групп."""
    if catalog_uri:
        try:
            df = pl.read_parquet(catalog_uri, columns=["item_id", "title"], storage_options=storage_options)
            logger.info("Catalog loaded from S3 (%s): %d items", catalog_uri, len(df))
            return CatalogCache(df)
        except Exception as e:
            logger.warning("Catalog S3 load failed (%s): %s — falling back to feed groups", catalog_uri, e)

    if not feed_groups:
        raise RuntimeError("feed_groups is empty — cannot build fallback catalog")
    frames = [grp.select(["item_id", "title"]) for grp in feed_groups.values()]
    combined = pl.concat(frames).unique("item_id").select(["item_id", "title"])
    return CatalogCache(combined)


def _s3_bytes(s: Settings, key: str | None) -> bytes | None:
    """Скачивает произвольный объект из S3 в память (bytes). None если S3 не настроен или ошибка."""
    if not (s.s3_bucket and s.s3_access_key_id and s.s3_secret_access_key and key):
        return None
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=s.s3_endpoint_url,
            aws_access_key_id=s.s3_access_key_id,
            aws_secret_access_key=s.s3_secret_access_key,
            region_name=s.s3_region,
            config=Config(signature_version="s3"),
        )
        blob = client.get_object(Bucket=s.s3_bucket, Key=key)["Body"].read()
        logger.info("Fetched from S3 (s3://%s/%s, %d bytes)", s.s3_bucket, key, len(blob))
        return blob
    except Exception as e:
        logger.warning("S3 fetch failed (s3://%s/%s): %s", s.s3_bucket, key, e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage_options = settings.s3_storage_options()

    feed_groups = _load_feed_groups(
        settings.feed_uri, storage_options, max_groups=settings.feed_max_groups
    )
    feed_group_keys = list(feed_groups.keys())

    catalog = _build_catalog(settings.catalog_uri, storage_options, feed_groups)
    image_urls = ImageUrlService(settings)

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    session_store = RedisSessionStore(redis_client, ttl_seconds=settings.session_ttl_seconds)

    feed_item_ids = list({iid for grp in feed_groups.values() for iid in grp["item_id"].to_list()})
    emb_store = ItemEmbeddingStore(
        settings.emb_uri,
        emb_dim=settings.emb_dim,
        item_ids=feed_item_ids,
        storage_options=storage_options,
    )
    emb_store._setup_index(feed_item_ids)

    reranker = SessionReranker(model_blob=_s3_bytes(settings, settings.reranker_s3_key))

    if settings.ann_enabled:
        ann_index = FaissANNIndex(
            index_blob=_s3_bytes(settings, settings.faiss_index_s3_key),
            ids_blob=_s3_bytes(settings, settings.faiss_ids_s3_key),
            nprobe=settings.ann_nprobe,
        )
    else:
        ann_index = FaissANNIndex()

    app.state.catalog = catalog
    app.state.image_urls = image_urls
    app.state.session_store = session_store
    app.state.redis = redis_client
    app.state.emb_store = emb_store
    app.state.reranker = reranker
    app.state.ann_index = ann_index
    app.state.feed_groups = feed_groups
    app.state.feed_group_keys = feed_group_keys
    app.state.settings = settings

    yield

    redis_client.close()


app = FastAPI(title="SessionRec", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root_status() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

app.include_router(v1_router, prefix="/api/v1")
