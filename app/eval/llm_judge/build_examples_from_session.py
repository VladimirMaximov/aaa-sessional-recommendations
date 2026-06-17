from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from redis import Redis

from app.core.config import Settings
from app.eval.llm_judge.schemas import (
    JudgeCandidateItem,
    JudgeComparisonExample,
    JudgeHistoryItem,
)

_LIKE_EID = 1
_SKIP_EID = 2
_TTL_SECONDS = 600
_POLICIES = ("original_order", "rerank_only", "rerank_plus_faiss")
_PAIRWISE = (
    ("original_order", "rerank_only"),
    ("rerank_only", "rerank_plus_faiss"),
    ("original_order", "rerank_plus_faiss"),
)


def _get_json(base_url: str, path: str, params: dict) -> dict:
    url = f"{base_url.rstrip('/')}{path}?{urlencode(params)}"
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _session_key(user_id: int, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


def _score_key(user_id: int, session_id: str) -> str:
    return f"session_scores:{user_id}:{session_id}"


def _group_key(session_id: str) -> str:
    return f"group:{session_id}"


def _read_history(redis: Redis, user_id: int, session_id: str) -> list[dict]:
    values = redis.lrange(_session_key(user_id, session_id), 0, -1)
    return [json.loads(value) for value in values]


def _write_temp_session(
    redis: Redis,
    *,
    source_session_id: str,
    temp_session_id: str,
    user_id: int,
    events: list[dict],
) -> None:
    pipe = redis.pipeline()
    pipe.delete(_session_key(user_id, temp_session_id))
    pipe.delete(_score_key(user_id, temp_session_id))
    pipe.delete(_group_key(temp_session_id))

    group_x = redis.get(_group_key(source_session_id))
    if group_x:
        pipe.set(_group_key(temp_session_id), group_x, ex=_TTL_SECONDS)

    for event in events:
        payload = {
            "user_id": user_id,
            "item_id": int(event["item_id"]),
            "eid": int(event["eid"]),
        }
        pipe.rpush(_session_key(user_id, temp_session_id), json.dumps(payload))

    pipe.expire(_session_key(user_id, temp_session_id), _TTL_SECONDS)
    pipe.execute()


def _delete_temp_session(redis: Redis, *, temp_session_id: str, user_id: int) -> None:
    redis.delete(
        _session_key(user_id, temp_session_id),
        _score_key(user_id, temp_session_id),
        _group_key(temp_session_id),
    )


def _overview_maps(overview: dict) -> dict[int, dict]:
    by_id = {int(item["item_id"]): item for item in overview.get("items", [])}
    return by_id


def _history_items(events: list[dict], title_by_id: dict[int, dict]) -> list[JudgeHistoryItem]:
    out: list[JudgeHistoryItem] = []
    for event in events:
        item_id = int(event["item_id"])
        meta = title_by_id.get(item_id, {})
        eid = int(event["eid"])
        out.append(
            JudgeHistoryItem(
                item_id=item_id,
                title=str(meta.get("title") or f"Item {item_id}"),
                action="like" if eid == _LIKE_EID else "skip",
                source="ranker" if meta else "unknown",
                original_rank=meta.get("original_rank"),
            )
        )
    return out


def _feed_candidates(feed: dict, policy: str, overview_by_id: dict[int, dict]) -> list[JudgeCandidateItem]:
    candidates: list[JudgeCandidateItem] = []
    for idx, item in enumerate(feed.get("items", []), start=1):
        item_id = int(item["item_id"])
        overview_meta = overview_by_id.get(item_id, {})
        original_rank = overview_meta.get("original_rank")
        rank_delta = item.get("rank_delta")
        if policy == "original_order" and original_rank is not None:
            rank_delta = 0
        candidates.append(
            JudgeCandidateItem(
                item_id=item_id,
                title=str(item.get("title") or f"Item {item['item_id']}"),
                source=item.get("source") or ("baseline" if policy == "original_order" else "unknown"),
                original_rank=original_rank,
                new_rank=idx,
                rank_delta=rank_delta,
            )
        )
    return candidates


def build_examples(
    *,
    api_base_url: str,
    redis_url: str,
    user_id: int,
    session_id: str,
    output_path: Path,
    limit: int,
    min_step: int,
    max_steps: int,
    pairs: tuple[tuple[str, str], ...] = _PAIRWISE,
) -> int:
    redis = Redis.from_url(redis_url, decode_responses=True)
    full_history = _read_history(redis, user_id, session_id)
    if not full_history:
        raise RuntimeError(f"No Redis history for user_id={user_id}, session_id={session_id!r}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as out:
        processed_steps = 0
        for step in range(max(0, min_step), len(full_history) + 1):
            if max_steps and processed_steps >= max_steps:
                break

            prefix = full_history[:step]
            temp_session_id = f"judge_{session_id}_{step}_{uuid.uuid4().hex[:8]}"
            _write_temp_session(
                redis,
                source_session_id=session_id,
                temp_session_id=temp_session_id,
                user_id=user_id,
                events=prefix,
            )
            try:
                overview = _get_json(
                    api_base_url,
                    "/api/v1/session_overview",
                    {"session_id": temp_session_id, "user_id": user_id},
                )
                by_id = _overview_maps(overview)
                history_items = _history_items(prefix, by_id)

                lists_by_policy: dict[str, list[JudgeCandidateItem]] = {}
                for policy in _POLICIES:
                    feed = _get_json(
                        api_base_url,
                        "/api/v1/feed",
                        {
                            "session_id": temp_session_id,
                            "user_id": user_id,
                            "limit": limit,
                            "policy": policy,
                        },
                    )
                    lists_by_policy[policy] = _feed_candidates(feed, policy, by_id)

                for left, right in pairs:
                    example = JudgeComparisonExample(
                        session_id=session_id,
                        user_id=user_id,
                        step=step,
                        list_a_name=left,
                        list_b_name=right,
                        history=history_items,
                        list_a=lists_by_policy[left],
                        list_b=lists_by_policy[right],
                    )
                    out.write(example.model_dump_json(ensure_ascii=False) + "\n")
                    written += 1
                processed_steps += 1
            finally:
                _delete_temp_session(redis, temp_session_id=temp_session_id, user_id=user_id)

            # Avoid hammering the local API when S3 presigned URLs or ANN are slow.
            time.sleep(0.05)

    return written


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(
        description="Build LLM judge JSONL examples from a real Redis-backed UI session."
    )
    parser.add_argument("--session-id", required=True, help="Browser localStorage session_id.")
    parser.add_argument("--user-id", type=int, default=1000001)
    parser.add_argument("--output", default="data/eval/judge_examples.jsonl")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--redis-url", default=settings.redis_url)
    parser.add_argument("--limit", type=int, default=10, help="Items per compared list.")
    parser.add_argument("--min-step", type=int, default=1, help="First history prefix length.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means all history prefixes.")
    parser.add_argument(
        "--pairs",
        default="original_order:rerank_only,rerank_only:rerank_plus_faiss,original_order:rerank_plus_faiss",
        help="Comma-separated pairwise comparisons, e.g. rerank_only:rerank_plus_faiss.",
    )
    args = parser.parse_args()
    pairs = tuple(
        tuple(pair.split(":", 1))  # type: ignore[misc]
        for pair in args.pairs.split(",")
        if pair.strip()
    )
    for left, right in pairs:
        if left not in _POLICIES or right not in _POLICIES:
            raise ValueError(f"Unknown pair {left}:{right}; allowed policies: {', '.join(_POLICIES)}")

    n = build_examples(
        api_base_url=args.api_base_url,
        redis_url=args.redis_url,
        user_id=args.user_id,
        session_id=args.session_id,
        output_path=Path(args.output),
        limit=args.limit,
        min_step=args.min_step,
        max_steps=args.max_steps,
        pairs=pairs,
    )
    print(f"Wrote {n} examples to {args.output}")


if __name__ == "__main__":
    main()
