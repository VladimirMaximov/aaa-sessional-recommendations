from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from redis import Redis

from app.api.v1 import router as v1_router
from app.core.config import Settings
from app.ml.factory import StrategyRegistry
from app.services.catalog import load_catalog
from app.services.datasets import DatasetHub, PopularityIndex
from app.services.log_store import RedisLogStore, load_logs
from app.services.session_store import RedisSessionStore

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализировать кэш каталога, логи, датасеты и стратегии."""
    catalog = load_catalog(settings.catalog_path, settings.catalog_size)
    redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
    session_store = RedisSessionStore(redis_client, ttl_seconds=settings.session_ttl_seconds)
    log_store = RedisLogStore(redis_client, max_events=settings.events_max_length)
    log_cache = load_logs(settings.logs_path)
    popularity_index = PopularityIndex(redis_client)
    data_hub = DatasetHub([popularity_index])
    data_hub.bootstrap(log_cache.df, reset=settings.logs_bootstrap_reset)
    strategy_registry = StrategyRegistry(
        catalog=catalog,
        log_store=log_store,
        log_cache=log_cache,
        data_hub=data_hub,
        popularity_index=popularity_index,
        settings=settings,
    )

    app.state.catalog = catalog
    app.state.session_store = session_store
    app.state.log_store = log_store
    app.state.log_cache = log_cache
    app.state.settings = settings
    app.state.data_hub = data_hub
    app.state.strategy_registry = strategy_registry

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
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}


app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")
app.include_router(v1_router, prefix="/api/v1")
