from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Action = Literal["like", "skip", "seen"]
CandidateSource = Literal["ranker", "ann_exploit", "ann_explore", "baseline", "unknown"]
Winner = Literal["A", "B", "tie"]


class JudgeHistoryItem(BaseModel):
    item_id: int
    title: str
    action: Action
    source: CandidateSource = "unknown"
    original_rank: int | None = None


class JudgeCandidateItem(BaseModel):
    item_id: int
    title: str
    source: CandidateSource = "unknown"
    original_rank: int | None = None
    new_rank: int | None = None
    rank_delta: int | None = None


class JudgeComparisonExample(BaseModel):
    session_id: str
    step: int = Field(ge=0)
    user_id: int | None = None
    list_a_name: str
    list_b_name: str
    history: list[JudgeHistoryItem]
    list_a: list[JudgeCandidateItem]
    list_b: list[JudgeCandidateItem]


class JudgeVerdict(BaseModel):
    winner: Winner
    confidence: float = Field(ge=0.0, le=1.0)
    relevance_A: int = Field(ge=1, le=5)
    relevance_B: int = Field(ge=1, le=5)
    diversity_A: int = Field(ge=1, le=5)
    diversity_B: int = Field(ge=1, le=5)
    exploration_quality_A: int = Field(ge=1, le=5)
    exploration_quality_B: int = Field(ge=1, le=5)
    bad_items_A: list[int] = Field(default_factory=list)
    bad_items_B: list[int] = Field(default_factory=list)
    reason: str


class JudgeResult(BaseModel):
    session_id: str
    step: int
    list_a_name: str
    list_b_name: str
    verdict: JudgeVerdict

