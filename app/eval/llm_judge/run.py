from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.eval.llm_judge.judge import LLMJudge
from app.eval.llm_judge.prompts import build_user_prompt
from app.eval.llm_judge.schemas import JudgeComparisonExample


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge pairwise evaluation.")
    parser.add_argument("--input", required=True, help="Input JSONL with JudgeComparisonExample rows.")
    parser.add_argument("--output", required=True, help="Output JSONL with JudgeResult rows.")
    parser.add_argument("--limit", type=int, default=0, help="Max examples to process. 0 means all.")
    parser.add_argument("--dry-run", action="store_true", help="Print first built prompt and exit.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples: list[JudgeComparisonExample] = []
    for _, raw in _iter_jsonl(input_path):
        examples.append(JudgeComparisonExample.model_validate(raw))
        if args.limit and len(examples) >= args.limit:
            break

    if args.dry_run:
        if not examples:
            raise RuntimeError("Input file has no examples")
        print(build_user_prompt(examples[0]))
        return

    judge = LLMJudge()
    with output_path.open("w", encoding="utf-8") as out:
        for example in examples:
            result = judge.judge_pairwise(example)
            out.write(result.model_dump_json(ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

