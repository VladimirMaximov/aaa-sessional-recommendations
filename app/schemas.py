from __future__ import annotations

from pydantic import BaseModel, Field


class InteractionRequest(BaseModel):
    """Запрос на запись пользовательского события."""

    session_id: str = Field(min_length=1)
    user_id: int
    item_id: int
    eid: int


class InteractionResponse(BaseModel):
    """Ответ на запись события."""

    status: str


class FeedItem(BaseModel):
    """Элемент ленты рекомендаций."""

    item_id: int
    title: str
    image_url: str


class FeedResponse(BaseModel):
    """Ответ с рекомендациями для сессии."""

    session_id: str
    items: list[FeedItem]
