from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import Settings
from app.ml.factory import StrategyConfigError, StrategyRegistry
from app.schemas import FeedItem, FeedResponse, InteractionRequest, InteractionResponse
from app.services.catalog import CatalogCache
from app.services.datasets import DatasetHub, Event
from app.services.log_store import RedisLogStore
from app.services.session_store import RedisSessionStore

router = APIRouter()


@router.get("/feed", response_model=FeedResponse)
def get_feed(
    session_id: str,
    user_id: int,
    request: Request,
    limit: int = Query(5, ge=1, le=50),
    candgen: str | None = Query(None),
    ranker: str | None = Query(None),
) -> FeedResponse:
    """Вернуть ленту для указанной сессии пользователя."""
    catalog: CatalogCache = request.app.state.catalog
    session_store: RedisSessionStore = request.app.state.session_store
    # log_store: RedisLogStore = request.app.state.log_store
    settings: Settings = request.app.state.settings
    # data_hub: DatasetHub = request.app.state.data_hub
    strategy_registry: StrategyRegistry = request.app.state.strategy_registry

    history = session_store.get_session(user_id, session_id)
    seen = {h["item_id"] for h in history if "item_id" in h}

    candgen_name = candgen or settings.candgen_strategy
    ranker_name = ranker or settings.ranker_strategy

    try:
        candgen_impl = strategy_registry.build_candgen(
            candgen_name,
            n_candidates=limit * 5,
        )
        ranker_impl = strategy_registry.build_ranker(ranker_name)
    except StrategyConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    candidates = candgen_impl.get_candidates(history)
    if not candidates and settings.candgen_fallback:
        fallback_impl = strategy_registry.build_candgen(
            settings.candgen_fallback,
            n_candidates=limit * 5,
        )
        candidates = fallback_impl.get_candidates(history)

    candidates = [c for c in candidates if c not in seen]
    ranked = ranker_impl.rank(history, candidates)

    items: list[FeedItem] = []
    for item_id in ranked:
        meta = catalog.get_item(item_id)
        if not meta:
            continue
        items.append(FeedItem(item_id=meta.item_id, title=meta.title, image_url=meta.image_url))
        if len(items) >= limit:
            break

    return FeedResponse(session_id=session_id, items=items)


@router.post("/interact", response_model=InteractionResponse)
def post_interact(
    payload: InteractionRequest,
    request: Request,
) -> InteractionResponse:
    """Сохранить пользовательское действие и обновить датасеты."""
    session_store: RedisSessionStore = request.app.state.session_store
    log_store: RedisLogStore = request.app.state.log_store
    data_hub: DatasetHub = request.app.state.data_hub
    session_store.add_event(payload.user_id, payload.session_id, payload.item_id, payload.eid)
    event_ts = log_store.add_event(
        payload.session_id,
        payload.item_id,
        payload.eid,
        user_id=payload.user_id,
    )
    data_hub.publish(
        Event(
            timestamp=event_ts,
            session_id=payload.session_id,
            item_id=payload.item_id,
            eid=payload.eid,
            user_id=payload.user_id,
        )
    )
    return InteractionResponse(status="ok")
