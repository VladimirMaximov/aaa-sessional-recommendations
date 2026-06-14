from __future__ import annotations

from pydantic import BaseModel, Field


class InteractionRequest(BaseModel):
    session_id: str = Field(min_length=1)
    user_id: int
    item_id: int
    eid: int


class InteractionResponse(BaseModel):
    status: str


class FeedItem(BaseModel):
    item_id: int
    title: str
    image_url: str
    rank_delta: int | None = None  # positive = moved up after reranking


class RankChange(BaseModel):
    item_id: int
    original_rank: int  # 1-based position in original RS order
    new_rank: int       # 1-based position after reranking
    delta: int          # original_rank - new_rank  (positive = moved up)


class RerankInfo(BaseModel):
    method: str          # "catboost_yetirank" | "cosine_sim" | "original_order"
    liked_count: int
    skipped_count: int
    candidates_count: int
    mean_rank_lift: float  # avg delta for served items (positive = moved up)
    rank_changes: list[RankChange]  # only for items actually served


class FeedResponse(BaseModel):
    session_id: str
    items: list[FeedItem]
    rerank_info: RerankInfo | None = None


class SessionOverviewItem(BaseModel):
    item_id: int
    title: str
    image_url: str
    original_rank: int   # 1-based position in RS order
    status: str          # "liked" | "skipped" | "seen" | "unseen"


class SessionOverviewResponse(BaseModel):
    session_id: str
    items: list[SessionOverviewItem]
