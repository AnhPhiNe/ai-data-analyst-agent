import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.agent.orchestrator import run_agent_turn  # noqa: E402
from backend.services.session_store import session_store  # noqa: E402


DEFAULT_EVAL_SET = ROOT / "docs" / "golden_answer_eval_set.jsonl"


def main() -> None:
    examples = [
        json.loads(line)
        for line in DEFAULT_EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct = 0
    failures = []
    for item in examples:
        dataframe = pd.read_csv(ROOT / item["dataset"])
        session = session_store.create(str(item["dataset"]), dataframe)
        response = run_agent_turn(session, item["question"], provider=None)
        passed, reasons = _check_response(item, response.model_dump())
        correct += int(passed)
        if not passed:
            failures.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "reasons": reasons,
                    "response_type": response.response_type,
                    "answer": response.answer,
                }
            )

    accuracy = correct / len(examples) if examples else 0.0
    print(f"Golden answer eval: {correct}/{len(examples)} passed ({accuracy:.1%})")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(
                f"- {failure['id']}: {', '.join(failure['reasons'])} :: "
                f"{failure['question']} -> {failure['response_type']} | {failure['answer']}"
            )


def _check_response(
    expected: dict[str, Any], response: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if response["response_type"] != expected["expected_response_type"]:
        reasons.append(
            f"expected response_type={expected['expected_response_type']}, "
            f"got {response['response_type']}"
        )

    final_tool = _final_tool(response)
    if final_tool != expected.get("expected_tool"):
        reasons.append(
            f"expected tool={expected.get('expected_tool')}, got {final_tool}"
        )

    for fragment in expected.get("answer_contains", []):
        if fragment not in response.get("answer", ""):
            reasons.append(f"answer missing '{fragment}'")

    table_contains = expected.get("table_contains")
    if table_contains and not _table_has_row(
        response.get("table") or [], table_contains
    ):
        reasons.append(f"table missing row subset {table_contains}")

    chart_contains = expected.get("chart_contains")
    if chart_contains:
        chart_spec = response.get("chart_spec") or {}
        for key, value in chart_contains.items():
            if chart_spec.get(key) != value:
                reasons.append(
                    f"chart_spec[{key}] expected {value}, got {chart_spec.get(key)}"
                )

    return not reasons, reasons


def _final_tool(response: dict[str, Any]) -> str | None:
    for trace in reversed(response.get("tool_trace") or []):
        tool_name = trace.get("tool_name")
        if tool_name:
            return str(tool_name)
    return None


def _table_has_row(table: list[dict[str, Any]], subset: dict[str, Any]) -> bool:
    return any(
        all(row.get(key) == value for key, value in subset.items()) for row in table
    )


if __name__ == "__main__":
    main()
