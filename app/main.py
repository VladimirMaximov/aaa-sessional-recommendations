from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis import Redis

from app.api.v1 import router as v1_router
from app.core.config import Settings
from app.ml.ann_index import ANNItemIndex, ItemEmbeddingStore
from app.ml.reranker import SessionReranker
from app.services.catalog import CatalogCache
from app.services.session_store import RedisSessionStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

settings = Settings()

# Колонки, загружаемые из parquet в каждый групповой DataFrame.
# title нужен для каталога, float_* + cat_* — признаки реранкера.
_FEED_KEEP_COLS = ["x", "item_id", "title"] + [f"float_{i}" for i in range(1, 81)] + ["cat_1", "cat_2", "cat_3"]


def _load_feed_groups(feed_path: str | None, max_groups: int = 0) -> dict[str, pl.DataFrame]:
    """Загружает parquet и возвращает {x: DataFrame} с признаками и метаданными в порядке строк.

    max_groups > 0 ограничивает число групп — важно для test (~15k групп), чтобы не OOM.
    """
    if not feed_path:
        raise RuntimeError("FEED_PATH is not set — cannot start without feed data")
    avail = pl.read_parquet(feed_path, n_rows=0).columns
    cols = [c for c in _FEED_KEEP_COLS if c in avail]
    df = pl.read_parquet(feed_path, columns=cols)
    groups = {str(x): grp for (x,), grp in df.group_by(["x"], maintain_order=True)}
    if max_groups > 0 and len(groups) > max_groups:
        keys = list(groups)[:max_groups]
        groups = {k: groups[k] for k in keys}
    return groups


def _build_catalog(val_groups: dict[str, pl.DataFrame]) -> CatalogCache:
    """Строит каталог из всех айтемов val-групп.

    Загрузка из val гарантирует 100% покрытие подаваемых айтемов без накладных расходов
    на полный item_catalog.parquet. Изображения обслуживаются как /images/{item_id}.jpg.
    """
    if not val_groups:
        raise RuntimeError("val_groups is empty — cannot build catalog")

    frames = [grp.select(["item_id", "title"]) for grp in val_groups.values()]
    combined = pl.concat(frames).unique("item_id")
    result_df = combined.with_columns(
        ("/images/" + pl.col("item_id").cast(pl.Utf8) + ".jpg").alias("image_url")
    ).select(["item_id", "title", "image_url"])

    return CatalogCache(result_df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализирует ресурсы при старте и освобождает их при остановке.

    Порядок инициализации: val-группы → каталог → Redis → эмбеддинги → реранкер.
    Все компоненты монтируются в app.state и доступны из обработчиков через request.app.state.
    """
    feed_groups = _load_feed_groups(settings.feed_path, max_groups=settings.feed_max_groups)
    feed_group_keys = list(feed_groups.keys())

    catalog = _build_catalog(feed_groups)

    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    session_store = RedisSessionStore(redis_client, ttl_seconds=settings.session_ttl_seconds)

    # Собираем item_ids до загрузки эмбеддингов, чтобы фильтровать parquet по ним
    # и не грузить весь файл (~7GB) целиком.
    feed_item_ids = list({iid for grp in feed_groups.values() for iid in grp["item_id"].to_list()})
    emb_store = ItemEmbeddingStore(settings.emb_path, emb_dim=settings.emb_dim, item_ids=feed_item_ids)
    emb_store._setup_index(feed_item_ids)  # no-op если parquet загружен успешно

    ann_index = ANNItemIndex(emb_store)

    reranker = SessionReranker(settings.reranker_path)

    app.state.catalog = catalog
    app.state.session_store = session_store
    app.state.redis = redis_client
    app.state.emb_store = emb_store
    app.state.ann_index = ann_index
    app.state.reranker = reranker
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

_images_dir = Path("data/images")
if _images_dir.is_dir():
    app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")

app.include_router(v1_router, prefix="/api/v1")
