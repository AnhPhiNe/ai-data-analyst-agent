# AI Data Analyst Agent

An end-to-end portfolio project for a safe AI agent that analyzes uploaded tabular datasets.

The app combines a FastAPI backend, Streamlit UI, pandas-based analysis tools, Plotly charts, guardrails, a hybrid router, and optional Gemini fallback. It is designed as an AI Engineer Intern portfolio project: small enough to understand, but complete enough to show practical agent engineering decisions.

## What It Does

- Upload CSV or XLSX datasets.
- Profile a dataset with schema, missing values, numeric summaries, top categories, and distributions.
- Ask natural-language questions about the uploaded data.
- Route clear requests to safe deterministic pandas tools.
- Fall back to Gemini for lower-confidence or ambiguous requests when `GEMINI_API_KEY` is configured.
- Keep Gemini optional: the app still works with deterministic routing and fallback suggestions without an API key.
- Generate validated Plotly chart specs for frontend rendering.
- Provide suggested questions and grounded analytical insights from profiling signals.
- Use guardrails to block unsafe or out-of-scope requests.
- Keep a tool trace for debugging and AI engineering review.

## Architecture

```text
Streamlit UI
    |
    | upload / profile / suggestions / chat
    v
FastAPI Backend
    |
    +-- Dataset loader and in-memory session store
    +-- Profiling service
    +-- Suggested questions and deterministic insights
    +-- Guardrails
    +-- Hybrid candidate-scoring router
    +-- Optional Gemini runtime
    +-- Tool validation and argument repair
    +-- Safe pandas tool registry
    +-- Chart spec validation
```

## Agent Flow

```text
User question
  -> Guardrails
  -> Clarification memory check
  -> Candidate-scoring router
      -> high-confidence tool call
      -> conflict or low confidence
  -> Optional Gemini fallback
  -> Column argument repair
  -> Tool validation
  -> Safe pandas execution
  -> User-facing answer + table/chart + trace
```

## Key Engineering Features

### Safe Tool Layer

The agent cannot execute arbitrary Python. It can only call whitelisted tools:

- `profile_dataset`
- `list_columns`
- `describe_numeric`
- `detect_missing_values`
- `value_counts`
- `aggregate_metric`
- `sort_values`
- `filter_rows`
- `conditional_percentage`
- `correlation_analysis`
- `generate_chart_spec`

Tool arguments are validated before execution, dangerous keys are rejected, and chart specs are schema-validated before rendering.

### Hybrid Router

The router no longer routes only by the first matched keyword. It builds route candidates, scores them by intent priority, confidence, and evidence, then falls back to Gemini when intents conflict.

This keeps common questions fast and deterministic while reducing brittle rule-based behavior for ambiguous wording.

### Multilingual Column Resolution

The agent can resolve user wording such as Vietnamese aliases, loose column descriptions, and near-matches to actual dataset columns. This is used both by the router and by the argument repair layer for LLM-generated tool calls.

### Suggested Analysis

Suggested insights are deterministic and grounded in profiling signals:

- highest missing values
- salient numeric summaries
- dominant categories
- outlier or unusual range signals
- correlation only when the absolute correlation is meaningful

Gemini can help diversify questions, but insights remain deterministic to reduce hallucination.

## Project Structure

```text
backend/
  agent/          guardrails, router, Gemini runtime, agent loop, column repair
  services/       upload/session/profiling services
  tools/          safe pandas tool registry
  visualization/ chart spec validation
frontend/
  streamlit_app.py
tests/
  pytest suite for upload, profiling, tools, router, Gemini, suggestions, chat
data/
  sample_student_performance.csv
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Copy environment variables:

```bash
copy .env.example .env
```

Gemini is optional. Leave `GEMINI_API_KEY` empty to use deterministic routing and fallback content only.

## Run Locally

Start the backend:

```bash
uvicorn backend.main:app --reload
```

Start the frontend in another terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Open the Streamlit URL, upload a CSV/XLSX file, and ask questions about the dataset.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `AI Data Analyst Agent` | FastAPI service name |
| `APP_ENV` | `development` | Runtime environment label |
| `GEMINI_API_KEY` | empty | Optional Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash-lite` | Gemini model name |
| `MAX_UPLOAD_MB` | `10` | Maximum upload size |
| `BACKEND_URL` | `http://localhost:8000` | Frontend backend URL |

## Run Tests

```bash
pytest
```

Current local status:

```text
140 passed
```

## Example Questions

- `Dataset co bao nhieu dong?`
- `Cot nao thieu du lieu nhieu nhat?`
- `Diem trung binh theo Parental_Involvement la bao nhieu?`
- `Phan phoi cua Attendance the nao?`
- `Ve heatmap tuong quan`
- `Hours_Studied co tuong quan voi Exam_Score khong?`
- `Ty le hoc sinh co Hours_Studied duoi 16 gio la bao nhieu?`

## Current Limitations

- Sessions are stored in memory, so uploaded datasets are lost when the backend restarts.
- The app is built for single-user local demos, not production multi-tenant workloads.
- There is no database, vector store, RAG pipeline, or autonomous background analysis.
- Gemini is optional and only used for fallback tool selection/question generation.
- The app analyzes uploaded tabular data only; it does not browse the web or access external data sources.

## Suggested CV Bullet

Built a safe AI data analyst agent for tabular datasets using FastAPI, Streamlit, pandas, Plotly, and Gemini, with guarded tool execution, schema-grounded routing, multilingual column resolution, deterministic analytical insights, chart generation, clarification memory, and 140 automated tests.
