import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.agent.router import route_question  # noqa: E402

DEFAULT_DATASET = ROOT / "data" / "sample_student_performance.csv"
DEFAULT_EVAL_SET = ROOT / "docs" / "route_eval_set.jsonl"


def main() -> None:
    dataframe = pd.read_csv(DEFAULT_DATASET)
    examples = [
        json.loads(line)
        for line in DEFAULT_EVAL_SET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correct = 0
    rows: list[dict[str, object]] = []
    for item in examples:
        decision = route_question(dataframe, item["question"])
        predicted_route = "tool" if decision.should_use_tool else decision.route_type
        predicted_tool = decision.tool_name if predicted_route == "tool" else None
        expected_route = item["expected_route"]
        expected_tool = item.get("expected_tool")
        passed = predicted_route == expected_route and predicted_tool == expected_tool
        correct += int(passed)
        rows.append(
            {
                "id": item["id"],
                "passed": passed,
                "expected_route": expected_route,
                "expected_tool": expected_tool,
                "predicted_route": predicted_route,
                "predicted_tool": predicted_tool,
                "question": item["question"],
            }
        )

    accuracy = correct / len(examples) if examples else 0.0
    print(f"Router eval: {correct}/{len(examples)} passed ({accuracy:.1%})")
    failures = [row for row in rows if not row["passed"]]
    if failures:
        print("\nFailures:")
        for row in failures:
            print(
                f"- {row['id']}: expected {row['expected_route']}/{row['expected_tool']}, "
                f"got {row['predicted_route']}/{row['predicted_tool']} :: {row['question']}"
            )


if __name__ == "__main__":
    main()
