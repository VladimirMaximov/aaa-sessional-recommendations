from __future__ import annotations

import logging
import random

import numpy as np
from fastapi import APIRouter, Query, Request

from app.ml.ann_index import FaissANNIndex, ItemEmbeddingStore
from app.ml.reranker import FLOAT_COLS, ORIG_CAT_COLS, SessionReranker
from app.schemas import (
    FeedItem,
    FeedResponse,
    InteractionRequest,
    InteractionResponse,
    RankChange,
    RerankInfo,
    SessionOverviewItem,
    SessionOverviewResponse,
)
from app.services.catalog import CatalogCache
from app.services.image_urls import ImageUrlService
from app.services.session_store import RedisSessionStore

router = APIRouter()
logger = logging.getLogger(__name__)

_GROUP_KEY_PREFIX = "group:"
_LIKE_EID = 1
_SKIP_EID = 2

# Вес негативного сигнала: query_emb = like_emb - SKIP_ALPHA * skip_emb.
_SKIP_ALPHA = 0.7

# EMA-реранкинг: статический вес нового сигнала на каждом шаге.
# Постоянный (не затухающий) вес позволяет вектору интересов сессии смещаться
# к свежим лайкам/скипам и делает инкрементальный расчёт идентичным полному реплею.
_EMA_WEIGHT = 0.70


def _get_or_assign_group(redis, session_id: str, feed_group_keys: list[str]) -> str | None:
    """Возвращает ключ feed-группы для сессии, при первом вызове назначает случайный и сохраняет в Redis."""
    if not feed_group_keys:
        return None
    key = f"{_GROUP_KEY_PREFIX}{session_id}"
    group_x = redis.get(key)
    if not group_x:
        group_x = random.choice(feed_group_keys)
        redis.set(key, group_x, ex=86400)
    return group_x


def _liked_skipped_from_events(
    events: list[tuple[int, int]], up_to: int
) -> tuple[list[int], list[int]]:
    liked: list[int] = []
    skipped: list[int] = []
    for ev_iid, eid in events[:up_to]:
        if eid == _LIKE_EID:
            liked.append(ev_iid)
        else:
            skipped.append(ev_iid)
    return liked, skipped


def _valid_score_cache(cache: dict | None, group_x: str, mode: str) -> dict | None:
    if not cache:
        return None
    if cache.get("group_x") != group_x or cache.get("mode") != mode:
        return None
    return cache


def _build_query_emb(
    liked: list[int],
    skipped: list[int],
    emb_store: ItemEmbeddingStore,
) -> np.ndarray | None:
    """Объединяет сигналы лайков и скипов в один нормированный вектор запроса.

    Работает в трёх режимах:
    - только лайки: среднее эмбеддингов лайков
    - лайки + скипы: like_emb - SKIP_ALPHA * skip_emb
    - только скипы: нулевой вектор - SKIP_ALPHA * skip_emb (отталкиваемся от скипнутых)
    Возвращает None если нет ни лайков, ни скипов, или вектор вырождается в ноль.
    """
    if not liked and not skipped:
        return None

    if liked:
        pos = emb_store.get_embs_batch(liked).mean(axis=0).astype(np.float64)
        n = np.linalg.norm(pos)
        if n > 1e-8:
            pos /= n
    else:
        pos = np.zeros(emb_store.dim, dtype=np.float64)

    if skipped:
        neg = emb_store.get_embs_batch(skipped).mean(axis=0).astype(np.float64)
        n = np.linalg.norm(neg)
        if n > 1e-8:
            neg /= n
        pos = pos - _SKIP_ALPHA * neg

    n = np.linalg.norm(pos)
    if n < 1e-8:
        return None
    return (pos / n).astype(np.float32)


def _likes_query_emb(
    liked: list[int],
    emb_store: ItemEmbeddingStore,
    recent: int,
) -> np.ndarray | None:
    """Нормированное среднее эмбеддингов последних `recent` лайков (query для ANN-поиска)."""
    if not liked:
        return None
    recent_liked = liked[-recent:] if recent > 0 else liked
    v = emb_store.get_embs_batch(recent_liked).mean(axis=0).astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return (v / n).astype(np.float32)


def _pick_ann_card(
    query_emb: np.ndarray,
    exclude_ids: set[int],
    ann_index: FaissANNIndex,
    settings,
    want_explore: bool,
) -> tuple[int, str] | None:
    """Выбирает один ANN-айтем: exploit из ближнего окна, explore из хвоста соседей.

    Возвращает (item_id, source) или None если кандидатов нет.
    """
    cand = ann_index.query(query_emb, settings.ann_pool_n, exclude_ids)
    if not cand:
        return None

    head = cand[: settings.ann_exploit_k]
    tail = cand[settings.ann_exploit_k :]

    if want_explore and tail:
        iid, _ = random.choice(tail)
        return int(iid), "ann_explore"

    iid, _ = random.choice(head or cand)
    return int(iid), "ann_exploit"


def _build_served(
    pool: list[int],
    base_index: int,
    limit: int,
    liked: list[int],
    exclude_ids: set[int],
    emb_store: ItemEmbeddingStore,
    ann_index: FaissANNIndex,
    settings,
) -> list[tuple[int, str]]:
    """Собирает выдачу с серверной каденцией ANN.

    Каждая ann_every-я карточка (по глобальной позиции base_index..) становится
    ANN-карточкой с чередованием exploit/explore. На остальных позициях — следующий
    невыбранный ranker-айтем из pool. ANN-слоты падают обратно на ranker при
    отсутствии кандидатов.
    """
    query_emb = (
        _likes_query_emb(liked, emb_store, settings.ann_recent_likes)
        if settings.ann_enabled and liked and ann_index.available
        else None
    )

    every = settings.ann_every
    served: list[tuple[int, str]] = []
    used: set[int] = set()
    pool_ptr = 0

    for pos in range(base_index, base_index + limit):
        is_ann_slot = (
            query_emb is not None and every > 0 and (pos + 1) % every == 0
        )
        if is_ann_slot:
            ann_slot_index = (pos + 1) // every - 1
            want_explore = ann_slot_index % 2 == 1
            picked = _pick_ann_card(
                query_emb, exclude_ids | used, ann_index, settings, want_explore
            )
            if picked is not None:
                served.append(picked)
                used.add(picked[0])
                continue

        while pool_ptr < len(pool):
            iid = pool[pool_ptr]
            pool_ptr += 1
            if iid not in used:
                served.append((iid, "ranker"))
                used.add(iid)
                break

    return served


def _compute_ema_scores(
    item_ids: list[int],
    group_df,
    remaining_df,
    history: list[dict],
    emb_store: ItemEmbeddingStore,
    reranker: SessionReranker,
    has_features: bool,
    score_cache: dict | None = None,
) -> tuple[dict[int, float], str, list[int], dict]:
    """Инкрементальный EMA для legacy-реранкера (session-only кандидаты).

    Обрабатывает только события после processed_events из кэша. На каждом шаге:
      scores = (1 - _EMA_WEIGHT) * scores + _EMA_WEIGHT * step_scores
    step_scores: CatBoost если есть лайки + фичи, иначе cosine_sim.
    """
    if not item_ids:
        empty: dict[int, float] = {}
        return empty, "original_order", [], {
            "scores": empty,
            "processed_events": 0,
            "pool_item_ids": [],
        }

    orig_order = {iid: i for i, iid in enumerate(group_df["item_id"].to_list())}
    n_total = len(group_df)
    events = [(h["item_id"], h["eid"]) for h in history if "item_id" in h]

    cache = score_cache or {}
    scores_list = [
        cache.get("scores", {}).get(iid, (n_total - orig_order.get(iid, n_total)) / n_total)
        for iid in item_ids
    ]
    scores_arr = np.array(scores_list, dtype=np.float64)
    processed = int(cache.get("processed_events", 0))
    used_catboost = processed > 0 and bool(cache.get("scores"))

    liked_so_far, skipped_so_far = _liked_skipped_from_events(events, processed)

    if processed == len(events) and cache.get("scores"):
        scores = {iid: float(scores_arr[i]) for i, iid in enumerate(item_ids)}
        liked_all, _ = _liked_skipped_from_events(events, len(events))
        method = "ema_catboost" if liked_all else "original_order"
        return scores, method, item_ids, {
            "scores": scores,
            "processed_events": processed,
            "pool_item_ids": item_ids,
        }

    float_mat: np.ndarray | None = None
    cat_mat_np: np.ndarray | None = None
    cand_embs: np.ndarray | None = None
    if has_features and reranker.available:
        float_mat = remaining_df.select(FLOAT_COLS).to_numpy().astype("float32")
        cat_mat_np = remaining_df.select(ORIG_CAT_COLS).to_numpy().astype(object)
        cand_embs = emb_store.get_embs_batch(item_ids)

    for step in range(processed, len(events)):
        ev_iid, eid = events[step]
        if eid == _LIKE_EID:
            liked_so_far.append(ev_iid)
        else:
            skipped_so_far.append(ev_iid)

        new_scores: np.ndarray | None = None

        if float_mat is not None and cat_mat_np is not None and cand_embs is not None and liked_so_far:
            last_liked_row = group_df.filter(group_df["item_id"] == liked_so_far[-1])
            liked_cats = (
                last_liked_row.select(ORIG_CAT_COLS).to_numpy().astype(object)[0]
                if len(last_liked_row) > 0
                else np.array(["nan"] * 3, dtype=object)
            )
            liked_cats_sets: list[set[str] | None] = [None, None, None]
            for iid in liked_so_far:
                row = group_df.filter(group_df["item_id"] == iid)
                if len(row) > 0:
                    cats = row.select(ORIG_CAT_COLS).to_numpy().astype(str)[0]
                    for ci in range(3):
                        if liked_cats_sets[ci] is None:
                            liked_cats_sets[ci] = set()
                        liked_cats_sets[ci].add(cats[ci])
            cat_str = cat_mat_np.astype(str)
            union_match = np.column_stack([
                np.array([c in liked_cats_sets[ci] for c in cat_str[:, ci]], dtype=np.float32)
                if liked_cats_sets[ci] is not None
                else np.zeros(len(cat_str), dtype=np.float32)
                for ci in range(3)
            ])
            query_emb = _build_query_emb(liked_so_far, skipped_so_far, emb_store)
            sims = (
                (cand_embs @ query_emb).astype(np.float64)
                if query_emb is not None
                else np.zeros(len(item_ids), dtype=np.float64)
            )
            raw = reranker.score_items(sims, float_mat, cat_mat_np, liked_cats, union_match).astype(np.float64)
            mn, mx = raw.min(), raw.max()
            new_scores = (raw - mn) / (mx - mn + 1e-8)
            used_catboost = True
        else:
            query_emb = _build_query_emb(liked_so_far, skipped_so_far, emb_store)
            if query_emb is not None:
                raw = (emb_store.get_embs_batch(item_ids) @ query_emb).astype(np.float64)
                mn, mx = raw.min(), raw.max()
                new_scores = (raw - mn) / (mx - mn + 1e-8)

        if new_scores is None:
            continue

        scores_arr = (1.0 - _EMA_WEIGHT) * scores_arr + _EMA_WEIGHT * new_scores

    processed = len(events)
    scores = {iid: float(scores_arr[i]) for i, iid in enumerate(item_ids)}

    if not events:
        method = "original_order"
    elif used_catboost:
        method = "ema_catboost"
    else:
        method = "ema_cosine"

    return scores, method, item_ids, {
        "scores": scores,
        "processed_events": processed,
        "pool_item_ids": item_ids,
    }


def _build_rerank_info(
    method: str,
    liked_count: int,
    skipped_count: int,
    item_ids: list[int],
    pool: list[int],
    served_ids: list[int],
) -> RerankInfo:
    orig_rank = {iid: i + 1 for i, iid in enumerate(item_ids)}
    new_rank = {iid: i + 1 for i, iid in enumerate(pool)}

    rank_changes = [
        RankChange(
            item_id=iid,
            original_rank=orig_rank[iid],
            new_rank=new_rank[iid],
            delta=orig_rank[iid] - new_rank[iid],
        )
        for iid in served_ids
        if iid in orig_rank and iid in new_rank
    ]

    served_deltas = [rc.delta for rc in rank_changes]
    mean_lift = float(np.mean(served_deltas)) if served_deltas else 0.0

    return RerankInfo(
        method=method,
        liked_count=liked_count,
        skipped_count=skipped_count,
        candidates_count=len(item_ids),
        mean_rank_lift=mean_lift,
        rank_changes=rank_changes,
    )


@router.get("/feed", response_model=FeedResponse)
def get_feed(
    session_id: str,
    user_id: int,
    request: Request,
    limit: int = Query(5, ge=1, le=50),
) -> FeedResponse:
    """Возвращает следующую порцию айтемов сессии с учётом истории лайков и скипов.

    EMA-реранкинг: инкрементальный реплей новых событий со статическим весом сигнала,
    результат кэшируется в Redis. Метод: ema_catboost / ema_cosine / original_order.
    """
    catalog: CatalogCache = request.app.state.catalog
    image_urls: ImageUrlService = request.app.state.image_urls
    session_store: RedisSessionStore = request.app.state.session_store
    redis = request.app.state.redis
    emb_store: ItemEmbeddingStore = request.app.state.emb_store
    reranker: SessionReranker = request.app.state.reranker
    ann_index: FaissANNIndex = request.app.state.ann_index
    settings = request.app.state.settings
    feed_groups: dict = request.app.state.feed_groups
    feed_group_keys: list = request.app.state.feed_group_keys

    history = session_store.get_session(user_id, session_id)
    seen = {h["item_id"] for h in history if "item_id" in h}
    liked = [h["item_id"] for h in history if h.get("eid") == _LIKE_EID]
    skipped = [h["item_id"] for h in history if h.get("eid") == _SKIP_EID]

    group_x = _get_or_assign_group(redis, session_id, feed_group_keys)

    pool: list[int]
    rerank_info: RerankInfo | None = None
    item_ids: list[int] = []
    method = "original_order"

    if group_x and group_x in feed_groups:
        group_df = feed_groups[group_x]
        remaining_df = group_df.filter(~group_df["item_id"].is_in(list(seen)))
        item_ids = remaining_df["item_id"].to_list()

        avail = set(remaining_df.columns)
        has_features = all(c in avail for c in FLOAT_COLS + ORIG_CAT_COLS)

        score_cache = _valid_score_cache(
            session_store.get_score_cache(user_id, session_id),
            group_x,
            "legacy",
        )
        score_map, method, item_ids, cache_out = _compute_ema_scores(
            item_ids,
            group_df,
            remaining_df,
            history,
            emb_store,
            reranker,
            has_features,
            score_cache=score_cache,
        )
        session_store.set_score_cache(
            user_id,
            session_id,
            scores=cache_out["scores"],
            processed_events=cache_out["processed_events"],
            pool_item_ids=cache_out["pool_item_ids"],
            group_x=group_x,
            mode="legacy",
        )
        if method != "original_order":
            pool = sorted(item_ids, key=lambda iid: score_map.get(iid, 0.0), reverse=True)
        else:
            pool = item_ids
    else:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"No feed group assigned for session {session_id!r}")

    # Серверная каденция ANN: каждая ann_every-я карточка отдаётся как ANN
    # (exploit/explore по очереди); позиция карточки = число уже пройденных событий.
    exclude_ids = set(group_df["item_id"].to_list()) | seen
    served = _build_served(
        pool,
        base_index=len(history),
        limit=limit,
        liked=liked,
        exclude_ids=exclude_ids,
        emb_store=emb_store,
        ann_index=ann_index,
        settings=settings,
    )

    items: list[FeedItem] = []
    for item_id, src in served:
        meta = catalog.get_item(item_id)
        if meta is None:
            if src == "ranker":
                continue
            title = f"Item {item_id}"
        else:
            title = meta.title
        items.append(
            FeedItem(
                item_id=item_id,
                title=title,
                image_url=image_urls.url_for(item_id),
                source=src,
            )
        )

    if method != "original_order" and item_ids:
        served_ids = [it.item_id for it in items]
        rerank_info = _build_rerank_info(
            method, len(liked), len(skipped), item_ids, pool, served_ids
        )

        rank_by_id = {rc.item_id: rc.delta for rc in rerank_info.rank_changes}
        for it in items:
            it.rank_delta = rank_by_id.get(it.item_id)

        top_movers = sorted(rerank_info.rank_changes, key=lambda r: r.delta, reverse=True)[:3]
        movers_str = "  ".join(
            f"#{r.original_rank}→#{r.new_rank}({r.delta:+d})" for r in top_movers
        )
        n_exploit = sum(1 for it in items if it.source == "ann_exploit")
        n_explore = sum(1 for it in items if it.source == "ann_explore")
        logger.info(
            "rerank  session=%-22s  method=%-18s  "
            "liked=%d  skipped=%d  candidates=%d  served_mean_lift=%+.1f  "
            "ann=[exploit=%d explore=%d]  top3=[%s]",
            session_id[:22],
            method,
            len(liked),
            len(skipped),
            len(item_ids),
            rerank_info.mean_rank_lift,
            n_exploit,
            n_explore,
            movers_str,
        )

    return FeedResponse(session_id=session_id, items=items, rerank_info=rerank_info)


@router.get("/session_overview", response_model=SessionOverviewResponse)
def get_session_overview(
    session_id: str,
    user_id: int,
    request: Request,
) -> SessionOverviewResponse:
    """Возвращает все айтемы сессии в исходном RS-порядке с их текущими статусами."""
    catalog: CatalogCache = request.app.state.catalog
    image_urls: ImageUrlService = request.app.state.image_urls
    session_store: RedisSessionStore = request.app.state.session_store
    redis = request.app.state.redis
    feed_groups: dict = request.app.state.feed_groups
    feed_group_keys: list = request.app.state.feed_group_keys

    history = session_store.get_session(user_id, session_id)
    liked_set = {h["item_id"] for h in history if h.get("eid") == _LIKE_EID}
    skipped_set = {h["item_id"] for h in history if h.get("eid") == _SKIP_EID}
    seen_set = {h["item_id"] for h in history if "item_id" in h}

    group_x = _get_or_assign_group(redis, session_id, feed_group_keys)

    item_ids_in_order: list[int] = []
    if group_x and group_x in feed_groups:
        item_ids_in_order = feed_groups[group_x]["item_id"].to_list()

    items: list[SessionOverviewItem] = []
    for rank, item_id in enumerate(item_ids_in_order, start=1):
        meta = catalog.get_item(item_id)
        if not meta:
            continue
        if item_id in liked_set:
            status = "liked"
        elif item_id in skipped_set:
            status = "skipped"
        elif item_id in seen_set:
            status = "seen"
        else:
            status = "unseen"
        items.append(
            SessionOverviewItem(
                item_id=meta.item_id,
                title=meta.title,
                image_url=image_urls.url_for(meta.item_id),
                original_rank=rank,
                status=status,
            )
        )

    return SessionOverviewResponse(session_id=session_id, items=items)


@router.post("/interact", response_model=InteractionResponse)
def post_interact(
    payload: InteractionRequest,
    request: Request,
) -> InteractionResponse:
    """Записывает событие взаимодействия (лайк/скип) в историю сессии."""
    session_store: RedisSessionStore = request.app.state.session_store
    session_store.add_event(payload.user_id, payload.session_id, payload.item_id, payload.eid)
    return InteractionResponse(status="ok")
