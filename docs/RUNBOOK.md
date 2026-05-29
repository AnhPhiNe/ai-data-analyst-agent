# Runbook

This runbook documents the local development and stable production deployment flow.

## 1. Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

`GEMINI_API_KEY` and `GROQ_API_KEY` are optional. Without an LLM key, the deterministic router and fallback suggestions still work.

## 2. Start Backend

```bash
uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## 3. Start Frontend

```bash
streamlit run frontend/streamlit_app.py
```

Use `data/sample_student_performance.csv` for a quick smoke test.

Useful demo questions:

- `Dataset có vấn đề chất lượng dữ liệu gì?`
- `Nhóm nào có salary trung bình cao nhất và có outlier không?`
- `So sánh salary theo department và vẽ biểu đồ`
- `Liệt kê top 10 user có salary cao nhất`

## 3b. One-command Docker Demo

```bash
docker compose up --build
```

Open `http://localhost:8501`.

## 4. Run Tests

```bash
pytest
```

Expected result at the time of writing:

```text
All tests passed
```

Router eval:

```bash
python scripts/evaluate_router.py
```

Golden answer eval:

```bash
python scripts/evaluate_golden_answers.py
```

## 5. Troubleshooting

### Backend is unreachable

Make sure FastAPI is running on the URL configured by `BACKEND_URL`.

### LLM fallback is skipped

Set `LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-2.5-flash-lite`, and `GEMINI_API_KEY` in `.env`, then restart the backend. To test Groq instead, set `LLM_PROVIDER=groq`, `GROQ_MODEL=llama-3.3-70b-versatile`, and `GROQ_API_KEY`.

### Uploaded session disappears

The project currently uses an in-memory session store with TTL and max-session eviction. Re-upload the dataset after backend restart or expiration.

### A chart does not render

Open the tool trace and check whether `generate_chart_spec` returned a validated chart spec. Invalid specs are rejected before rendering.

### SQL fallback is rejected

SQL fallback is intentionally read-only. Use one `SELECT` or `WITH ... SELECT` query against the `dataset` table only. File access, write statements, multiple statements, and non-`dataset` tables are blocked by validation.

## 6. Production Notes

This project is currently scoped as a stable, lightweight release. It now includes structured request logs, request ids, lightweight session ownership support, controlled multi-step planning, read-only SQL fallback, and basic per-process rate limits, but full production scaling still needs persistent storage, real auth, centralized monitoring, and deployment secrets management.
