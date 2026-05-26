import json
import logging
from queue import Queue
from threading import Thread
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agent.orchestrator import run_agent_turn
from backend.agent.gemini_runtime import GeminiProvider, LLMProvider
from backend.agent.suggestions import generate_suggested_content
from backend.core.config import get_settings
from backend.core.logging import configure_logging, log_event
from backend.core.rate_limit import InMemoryRateLimiter
from backend.schemas import (
    AutoAnalysisResponse,
    ChatRequest,
    ChatResponse,
    DatasetProfileResponse,
    DatasetUploadResponse,
    SuggestedContentResponse,
)
from backend.services.auto_analysis import generate_auto_analysis
from backend.services.dataset_loader import DatasetLoadError, load_dataframe
from backend.services.profiling import dataframe_preview, profile_dataset
from backend.services.session_store import DatasetSession, session_store


settings = get_settings()
configure_logging()
logger = logging.getLogger("backend.main")
session_store.configure(
    ttl_seconds=settings.session_ttl_seconds, max_sessions=settings.max_sessions
)
rate_limiter = InMemoryRateLimiter(
    max_requests=settings.rate_limit_per_minute,
    window_seconds=settings.rate_limit_window_seconds,
)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Safe AI data analyst agent backend for uploaded tabular datasets.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    started_at = perf_counter()
    client_host = request.client.host if request.client else "unknown"
    rate_limited_paths = (
        "/datasets/upload",
        "/chat/query",
        "/chat/query/stream",
    )

    if request.url.path in rate_limited_paths:
        rate_key = f"{client_host}:{request.url.path}"
        if not rate_limiter.allow(rate_key):
            log_event(
                logger,
                "http_request_rate_limited",
                request_id=request_id,
                path=request.url.path,
                client_host=client_host,
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Please try again shortly."},
                headers={"X-Request-ID": request_id},
            )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.exception(
            "http_request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
            },
        )
        raise

    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    log_event(
        logger,
        "http_request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


@app.post(
    "/datasets/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    content = await file.read()

    try:
        dataframe = load_dataframe(
            filename=file.filename or "",
            content=content,
            max_upload_mb=settings.max_upload_mb,
            max_rows=settings.max_rows,
            max_columns=settings.max_columns,
        )
    except DatasetLoadError as exc:
        log_event(
            logger, "dataset_upload_rejected", filename=file.filename, reason=str(exc)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    session = session_store.create(
        filename=file.filename or "uploaded_dataset", dataframe=dataframe
    )
    log_event(
        logger,
        "dataset_uploaded",
        session_id=session.session_id,
        filename=session.filename,
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
    )

    return DatasetUploadResponse(
        session_id=session.session_id,
        session_token=session.access_token,
        expires_at=session.expires_at.isoformat(),
        filename=session.filename,
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns],
        preview=dataframe_preview(dataframe),
    )


@app.get("/datasets/{session_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(
    session_id: str,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> DatasetProfileResponse:
    session = _get_session_or_404(session_id, x_session_token)

    with session._lock:
        if session.profile_cache is None:
            session.profile_cache = profile_dataset(session.dataframe)
            log_event(logger, "dataset_profile_computed", session_id=session.session_id)
        else:
            log_event(
                logger, "dataset_profile_cache_hit", session_id=session.session_id
            )
        profile = session.profile_cache
    return DatasetProfileResponse(
        session_id=session.session_id,
        filename=session.filename,
        **profile,
    )


@app.get("/datasets/{session_id}/suggestions", response_model=SuggestedContentResponse)
def get_dataset_suggestions(
    session_id: str,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> SuggestedContentResponse:
    session = _get_session_or_404(session_id, x_session_token)

    with session._lock:
        if session.suggestions_cache is None:
            suggested = generate_suggested_content(
                session.dataframe, provider=get_llm_provider()
            )
            session.suggestions_cache = suggested
            log_event(
                logger,
                "dataset_suggestions_computed",
                session_id=session.session_id,
                source=suggested.source,
            )
        else:
            suggested = session.suggestions_cache
            log_event(
                logger,
                "dataset_suggestions_cache_hit",
                session_id=session.session_id,
                source=suggested.source,
            )
    return SuggestedContentResponse(
        session_id=session.session_id,
        questions=suggested.questions,
        insights=suggested.insights,
        source=suggested.source,
    )


@app.get("/datasets/{session_id}/auto-analysis", response_model=AutoAnalysisResponse)
def get_dataset_auto_analysis(
    session_id: str,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> AutoAnalysisResponse:
    session = _get_session_or_404(session_id, x_session_token)
    with session._lock:
        if session.profile_cache is None:
            session.profile_cache = profile_dataset(session.dataframe)
        profile = session.profile_cache
    provider = get_llm_provider()
    analysis = generate_auto_analysis(
        session.dataframe, profile=profile, provider=provider
    )

    log_event(
        logger,
        "dataset_auto_analysis_completed",
        session_id=session.session_id,
        recommended_charts=len(analysis["recommended_charts"]),
    )
    return AutoAnalysisResponse(session_id=session.session_id, **analysis)


_llm_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider | None:
    global _llm_provider
    if _llm_provider is None:
        if settings.gemini_api_key:
            _llm_provider = GeminiProvider(
                api_key=settings.gemini_api_key, model=settings.gemini_model
            )
    return _llm_provider


@app.post("/chat/query", response_model=ChatResponse)
def chat_query(
    request: ChatRequest,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> ChatResponse:
    session = _get_session_or_404(request.session_id, x_session_token)
    response = run_agent_turn(
        session=session,
        question=request.question,
        provider=get_llm_provider(),
    )
    last_trace = response.tool_trace[-1] if response.tool_trace else None
    log_event(
        logger,
        "chat_turn_completed",
        session_id=session.session_id,
        response_type=response.response_type,
        final_source=last_trace.source if last_trace else None,
        final_tool=last_trace.tool_name if last_trace else None,
        is_blocked=response.is_blocked,
        should_clarify=response.should_clarify,
    )
    return response


@app.post("/chat/query/stream")
def chat_query_stream(
    request: ChatRequest,
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> StreamingResponse:
    session = _get_session_or_404(request.session_id, x_session_token)

    def event_stream():
        events: Queue[dict[str, Any] | None] = Queue()

        def emit_trace(trace) -> None:
            events.put({"type": "trace", "trace": _model_to_dict(trace)})

        def run_turn() -> None:
            try:
                response = run_agent_turn(
                    session=session,
                    question=request.question,
                    provider=get_llm_provider(),
                    event_callback=emit_trace,
                )
                last_trace = response.tool_trace[-1] if response.tool_trace else None
                log_event(
                    logger,
                    "chat_turn_completed",
                    session_id=session.session_id,
                    response_type=response.response_type,
                    final_source=last_trace.source if last_trace else None,
                    final_tool=last_trace.tool_name if last_trace else None,
                    is_blocked=response.is_blocked,
                    should_clarify=response.should_clarify,
                )
                events.put({"type": "final", "response": _model_to_dict(response)})
            except Exception:
                logger.exception(
                    "chat_stream_failed", extra={"session_id": session.session_id}
                )
                events.put(
                    {
                        "type": "error",
                        "message": "Chat request failed while the agent was running.",
                    }
                )
            finally:
                events.put(None)

        worker = Thread(target=run_turn, daemon=True)
        worker.start()
        yield _encode_stream_event(
            {"type": "step", "message": "Loaded dataset session."}
        )
        while True:
            event = events.get()
            if event is None:
                break
            yield _encode_stream_event(event)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _get_session_or_404(
    session_id: str, session_token: str | None = None
) -> DatasetSession:
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Dataset session not found."
        )
    if not session_store.verify_access(
        session, session_token, required=settings.require_session_token
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid session token."
        )
    return session


def _encode_stream_event(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


def _model_to_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()
