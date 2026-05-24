# AI Data Analyst Agent

MVP portfolio project for a safe AI agent that analyzes tabular data with FastAPI, Streamlit, pandas, Plotly, and Gemini.

## Current Scope

Current scope:

- FastAPI backend skeleton
- Streamlit frontend skeleton
- Environment example
- Pytest setup
- Health check test
- CSV/XLSX upload endpoint
- In-memory dataset sessions
- Dataset profiling endpoint
- Streamlit profiling dashboard
- Whitelisted safe pandas tool layer
- Validated chart specs and Streamlit Plotly renderer
- Guardrails and high-confidence rule-based router
- Gemini runtime with mockable provider, JSON parsing, retry, and tool-call validation

## Run Backend

```bash
uvicorn backend.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run Frontend

```bash
streamlit run frontend/streamlit_app.py
```

## Run Tests

```bash
pytest
```
