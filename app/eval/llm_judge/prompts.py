from __future__ import annotations

import json

from app.eval.llm_judge.schemas import JudgeComparisonExample


SYSTEM_PROMPT = """Ты оцениваешь качество сессионных рекомендаций объявлений.

Контекст:
- изначально есть уже отранжированный пул объявлений;
- после каждого like/skip система переранжирует оставшуюся часть пула;
- дополнительно система может подмешивать FAISS-кандидатов из общего каталога;
- source=ranker означает айтем из исходного пула после rerank;
- source=ann_exploit означает близкий ANN-кандидат по лайкам;
- source=ann_explore означает более дальний ANN-кандидат для exploration.

Твоя задача: сравнить две следующие выдачи для одного и того же состояния сессии.
Оценивай не объявления сами по себе, а то, какая выдача лучше подходит пользователю
после показанной истории like/skip.

Критерии:
1. Релевантность лайкам пользователя.
2. Учет скипов: явно похожие на skipped объявления хуже.
3. Качество rerank: поднятые вверх объявления должны быть оправданы историей.
4. Качество FAISS-подмешивания: ANN-кандидаты должны быть либо близкими, либо
   разумным exploration, а не случайным шумом.
5. Баланс exploitation/exploration: слишком однообразная выдача хуже, но слишком
   далекий exploration тоже хуже.

Верни только валидный JSON без markdown и без пояснений вне JSON.
"""


def build_user_prompt(example: JudgeComparisonExample) -> str:
    payload = {
        "session_id": example.session_id,
        "step": example.step,
        "history": [item.model_dump() for item in example.history],
        "list_A": {
            "name": example.list_a_name,
            "items": [item.model_dump() for item in example.list_a],
        },
        "list_B": {
            "name": example.list_b_name,
            "items": [item.model_dump() for item in example.list_b],
        },
        "required_output_schema": {
            "winner": "A | B | tie",
            "confidence": "float from 0 to 1",
            "relevance_A": "integer 1..5",
            "relevance_B": "integer 1..5",
            "diversity_A": "integer 1..5",
            "diversity_B": "integer 1..5",
            "exploration_quality_A": "integer 1..5",
            "exploration_quality_B": "integer 1..5",
            "bad_items_A": "list of item_id",
            "bad_items_B": "list of item_id",
            "reason": "short Russian explanation",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

