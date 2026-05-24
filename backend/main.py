import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status

from backend.core.config import get_settings
from backend.schemas import DatasetUploadResponse
from backend.services.dataset_loader import DatasetLoadError, load_dataframe
from backend.services.session_store import session_store


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="MVP backend for a safe AI agent that analyzes tabular datasets.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
    }


@app.post("/datasets/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetUploadResponse:
    content = await file.read()

    try:
        dataframe = load_dataframe(
            filename=file.filename or "",
            content=content,
            max_upload_mb=settings.max_upload_mb,
        )
    except DatasetLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session = session_store.create(filename=file.filename or "uploaded_dataset", dataframe=dataframe)
    preview_frame = dataframe.head(10).astype(object)
    preview = preview_frame.where(pd.notna(preview_frame), None)

    return DatasetUploadResponse(
        session_id=session.session_id,
        filename=session.filename,
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns],
        preview=preview.to_dict(orient="records"),
    )
