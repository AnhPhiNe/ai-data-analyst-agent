# Runbook

This runbook documents the local development and portfolio demo flow.

## 1. Install

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

`GEMINI_API_KEY` is optional. Without it, the deterministic router and fallback suggestions still work.

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

## 4. Run Tests

```bash
pytest
```

Expected result at the time of writing:

```text
140 passed
```

## 5. Troubleshooting

### Backend is unreachable

Make sure FastAPI is running on the URL configured by `BACKEND_URL`.

### Gemini fallback is skipped

Set `GEMINI_API_KEY` in `.env` and restart the backend.

### Uploaded session disappears

The project currently uses an in-memory session store. Re-upload the dataset after backend restart.

### A chart does not render

Open the tool trace and check whether `generate_chart_spec` returned a validated chart spec. Invalid specs are rejected before rendering.

## 6. Production Notes

This project is intentionally scoped as a portfolio MVP. For production, add persistent storage, auth, dataset lifecycle management, request logging, rate limits, and deployment secrets management.
