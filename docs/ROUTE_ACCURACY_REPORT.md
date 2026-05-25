# Router Accuracy Report

This report documents a lightweight deterministic routing eval for the portfolio demo.

## Scope

- Dataset: `data/sample_student_performance.csv`
- Eval file: `docs/route_eval_set.jsonl`
- Size: 60 Vietnamese/English questions
- Metric: exact match on `expected_route` and `expected_tool`

## Run

```bash
python scripts/evaluate_router.py
```

Current local result:

```text
Router eval: 60/60 passed (100.0%)
```

## Why This Exists

The router is heuristic, so test count alone does not prove language-routing quality. This eval set makes the current intent coverage explicit and gives reviewers a concrete artifact to discuss.

## Expected Failure Modes

- Broad requests such as "anything interesting?" should fall back to the LLM layer.
- Mixed chart + aggregation requests intentionally fall back because the deterministic router detects conflicting intents.
- Ambiguous chart/aggregate requests should clarify instead of guessing.

## Interpreting Results

This is not a production NLU benchmark. It is a small regression suite for portfolio-level routing behavior. A production version should add:

- Multiple datasets with different schemas.
- More natural paraphrases from real users.
- Per-intent precision/recall.
- Golden answer evaluation after tool execution.
