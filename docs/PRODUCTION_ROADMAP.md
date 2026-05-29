# Production Roadmap

This project is intentionally scoped as a portfolio MVP. The items below are the production direction without pretending the demo is already a production platform.

## Implemented Now

- In-memory sessions have TTL refresh and max-session eviction.
- Uploads are bounded by file size, row count, and column count.
- Profile and suggested-content responses are cached per session.
- Optional session token ownership can be enabled with `REQUIRE_SESSION_TOKEN=true`.
- Backend emits structured JSON log events with request ids and latency for HTTP requests, upload, profile, suggestions, and chat turns.
- Upload and chat endpoints have a basic per-process rate limiter.
- Router behavior has a small JSONL eval set.
- Golden answer behavior has a small multi-dataset JSONL eval set.

## Next Production Steps

### Persistent Dataset Store

Move uploaded dataframes out of process memory into object storage or a database-backed artifact store. Keep only lightweight session metadata in the API process.

### Worker Queue For Heavy Profiling

Large files should be profiled asynchronously. The API should return a job id, and the frontend should poll job status instead of blocking a request.

### Observability Dashboard

Track route type, selected tool, latency, LLM fallback rate, validation failures, blocked requests, upload size, and session eviction count.

### LLM Provider Abstraction

Keep the current `LLMProvider` protocol, then add provider-specific structured-output implementations for every LLM-backed feature. The current tool-selection path supports Gemini structured output and Groq-compatible JSON object mode, then validates parsed output locally before any tool execution.

### Golden Answer Evaluation

Add a second eval layer after router eval:

- question
- expected route/tool
- expected key table/chart fields
- expected answer constraints
- forbidden hallucinated claims

This should be run against multiple datasets, not only the sample student dataset.
