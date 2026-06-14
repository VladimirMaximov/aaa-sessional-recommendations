from __future__ import annotations

import logging
import random

import numpy as np
from fastapi import APIRouter, Query, Request

from app.ml.ann_index import ItemEmbeddingStore
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
from app.services.session_store import RedisSessionStore

router = APIRouter()
logger = logging.getLogger(__name__)

_GROUP_KEY_PREFIX = "group:"
_LIKE_EID = 1
_SKIP_EID = 2

# Вес негативного сигнала: query_emb = like_emb - SKIP_ALPHA * skip_emb.
_SKIP_ALPHA = 0.7

# EMA-реранкинг: вес нового сигнала убывает с каждым событием.
_EMA_BASE_WEIGHT = 0.70
_EMA_DECAY = 0.70


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
        # skip-only: стартуем с нуля, скип-вектор отклонит нас от нежеланных айтемов
        pos = np.zeros(emb_store._dim, dtype=np.float64)

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


def _compute_ema_scores(
    item_ids: list[int],
    group_df,
    remaining_df,
    history: list[dict],
    emb_store: ItemEmbeddingStore,
    reranker: SessionReranker,
    has_features: bool,
) -> tuple[np.ndarray, str]:
    """EMA-реранкинг: реплей истории событий с затухающим весом нового сигнала.

    На каждом шаге t:
      new_weight = _EMA_BASE_WEIGHT * _EMA_DECAY^t
      scores = (1 - new_weight) * scores + new_weight * step_scores

    step_scores: CatBoost если есть лайки + фичи, иначе cosine_sim.
    Базовые scores = нормированные исходные позиции RS (позиция 1 → 1.0).
    """
    n = len(item_ids)
    if n == 0:
        return np.array([]), "original_order"

    orig_order = {iid: i for i, iid in enumerate(group_df["item_id"].to_list())}
    n_total = len(group_df)
    scores = np.array(
        [(n_total - orig_order.get(iid, n_total)) / n_total for iid in item_ids],
        dtype=np.float64,
    )

    # Предвычисляем матрицы признаков один раз — они не меняются между шагами
    float_mat: np.ndarray | None = None
    cat_mat_np: np.ndarray | None = None
    cand_embs: np.ndarray | None = None
    if has_features and reranker.available:
        float_mat = remaining_df.select(FLOAT_COLS).to_numpy().astype("float32")
        cat_mat_np = remaining_df.select(ORIG_CAT_COLS).to_numpy().astype(object)
        cand_embs = emb_store.get_embs_batch(item_ids)

    events = [(h["item_id"], h["eid"]) for h in history if "item_id" in h]
    liked_so_far: list[int] = []
    skipped_so_far: list[int] = []
    used_catboost = False

    for step, (ev_iid, eid) in enumerate(events):
        if eid == _LIKE_EID:
            liked_so_far.append(ev_iid)
        else:
            skipped_so_far.append(ev_iid)

        new_scores: np.ndarray | None = None

        if float_mat is not None and cat_mat_np is not None and cand_embs is not None and liked_so_far:
            # liked_cat_* — категории последнего лайка (совместимо с обученной моделью)
            last_liked_row = group_df.filter(group_df["item_id"] == liked_so_far[-1])
            liked_cats = (
                last_liked_row.select(ORIG_CAT_COLS).to_numpy().astype(object)[0]
                if len(last_liked_row) > 0
                else np.array(["nan"] * 3, dtype=object)
            )
            # cat_*_match — union по всем лайкнутым: 1 если кандидат совпадает хотя бы с одним
            liked_cats_sets = [None, None, None]
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
                if liked_cats_sets[ci] is not None else np.zeros(len(cat_str), dtype=np.float32)
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

        w = _EMA_BASE_WEIGHT * (_EMA_DECAY ** step)
        scores = (1.0 - w) * scores + w * new_scores

    if not events:
        method = "original_order"
    elif used_catboost:
        method = "ema_catboost"
    else:
        method = "ema_cosine"

    return scores, method


def _build_rerank_info(
    method: str,
    liked_count: int,
    skipped_count: int,
    item_ids: list[int],
    pool: list[int],
    served_ids: list[int],
) -> RerankInfo:
    """Формирует статистику реранкинга: дельты позиций для каждого айтема в выдаче.

    delta = original_rank - new_rank: положительное значение означает подъём.
    mean_rank_lift считается только по отданным айтемам, чтобы отражать реальную пользу для пользователя.
    """
    orig_rank = {iid: i + 1 for i, iid in enumerate(item_ids)}
    new_rank  = {iid: i + 1 for i, iid in enumerate(pool)}

    rank_changes = [
        RankChange(
            item_id=iid,
            original_rank=orig_rank[iid],
            new_rank=new_rank[iid],
            delta=orig_rank[iid] - new_rank[iid],
        )
        for iid in served_ids
        if iid in orig_rank
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

    Использует EMA-реранкинг: реплей всех событий сессии с затухающим весом нового сигнала.
    На каждом шаге реранкинг блендируется с текущей позицией, которая уже несёт в себе
    результаты предыдущих шагов. Метод: ema_catboost / ema_cosine / original_order.
    """
    catalog: CatalogCache = request.app.state.catalog
    session_store: RedisSessionStore = request.app.state.session_store
    redis = request.app.state.redis
    emb_store: ItemEmbeddingStore = request.app.state.emb_store
    reranker: SessionReranker = request.app.state.reranker
    feed_groups: dict = request.app.state.feed_groups
    feed_group_keys: list = request.app.state.feed_group_keys

    history = session_store.get_session(user_id, session_id)
    seen    = {h["item_id"] for h in history if "item_id" in h}
    liked   = [h["item_id"] for h in history if h.get("eid") == _LIKE_EID]
    skipped = [h["item_id"] for h in history if h.get("eid") == _SKIP_EID]

    group_x = _get_or_assign_group(redis, session_id, feed_group_keys)

    pool: list[int]
    rerank_info: RerankInfo | None = None

    if group_x and group_x in feed_groups:
        group_df = feed_groups[group_x]
        remaining_df = group_df.filter(~group_df["item_id"].is_in(list(seen)))
        item_ids = remaining_df["item_id"].to_list()

        avail = set(remaining_df.columns)
        has_features = all(c in avail for c in FLOAT_COLS + ORIG_CAT_COLS)

        ema_scores, method = _compute_ema_scores(
            item_ids, group_df, remaining_df, history, emb_store, reranker, has_features
        )

        if len(ema_scores) > 0 and method != "original_order":
            pool = [item_ids[i] for i in np.argsort(ema_scores)[::-1]]
        else:
            pool = item_ids
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"No feed group assigned for session {session_id!r}")

    items: list[FeedItem] = []
    for item_id in pool:
        meta = catalog.get_item(item_id)
        if not meta:
            continue
        items.append(FeedItem(item_id=meta.item_id, title=meta.title, image_url=meta.image_url))
        if len(items) >= limit:
            break

    if method != "original_order" and item_ids:
        served_ids  = [it.item_id for it in items]
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
        logger.info(
            "rerank  session=%-22s  method=%-18s  "
            "liked=%d  skipped=%d  candidates=%d  served_mean_lift=%+.1f  top3=[%s]",
            session_id[:22], method,
            len(liked), len(skipped), len(item_ids),
            rerank_info.mean_rank_lift, movers_str,
        )

    return FeedResponse(session_id=session_id, items=items, rerank_info=rerank_info)


@router.get("/session_overview", response_model=SessionOverviewResponse)
def get_session_overview(
    session_id: str,
    user_id: int,
    request: Request,
) -> SessionOverviewResponse:
    """Возвращает все айтемы сессии в исходном RS-порядке с их текущими статусами.

    Статусы: liked / skipped / seen (показан, нет реакции) / unseen (ещё не показан).
    """
    catalog: CatalogCache = request.app.state.catalog
    session_store: RedisSessionStore = request.app.state.session_store
    redis = request.app.state.redis
    feed_groups: dict = request.app.state.feed_groups
    feed_group_keys: list = request.app.state.feed_group_keys

    history = session_store.get_session(user_id, session_id)
    liked_set   = {h["item_id"] for h in history if h.get("eid") == _LIKE_EID}
    skipped_set = {h["item_id"] for h in history if h.get("eid") == _SKIP_EID}
    seen_set    = {h["item_id"] for h in history if "item_id" in h}

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
        items.append(SessionOverviewItem(
            item_id=meta.item_id,
            title=meta.title,
            image_url=meta.image_url,
            original_rank=rank,
            status=status,
        ))

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
