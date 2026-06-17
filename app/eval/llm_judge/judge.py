from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.env import load_project_env
from app.eval.llm_judge.prompts import SYSTEM_PROMPT, build_user_prompt
from app.eval.llm_judge.schemas import JudgeComparisonExample, JudgeResult, JudgeVerdict

load_project_env()

DEFAULT_BASE_URL = "https://routerai.ru/api/v1"
DEFAULT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM response does not contain JSON: {text[:300]}")
    return json.loads(stripped[start : end + 1])


class LLMJudge:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_api_key = api_key or os.getenv("ROUTERAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError("ROUTERAI_API_KEY is not set")

        self._model_name = model or os.getenv("LLM_JUDGE_MODEL", DEFAULT_MODEL)
        self._llm = ChatOpenAI(
            api_key=resolved_api_key,
            base_url=base_url or os.getenv("ROUTERAI_BASE_URL", DEFAULT_BASE_URL),
            model=self._model_name,
            temperature=temperature,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def judge_pairwise(self, example: JudgeComparisonExample) -> JudgeResult:
        response = self._llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(example)),
            ]
        )
        verdict = JudgeVerdict.model_validate(_extract_json(str(response.content)))
        return JudgeResult(
            session_id=example.session_id,
            step=example.step,
            list_a_name=example.list_a_name,
            list_b_name=example.list_b_name,
            verdict=verdict,
        )
