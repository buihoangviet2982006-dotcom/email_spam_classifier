"""FastAPI application for Spam Email Classifier demo."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.predictor import SpamClassifierService
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
service = SpamClassifierService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="Spam Email Classifier API",
    description=(
        "REST API phân loại Email Spam / Ham sử dụng mô hình máy học tốt nhất "
        "được lựa chọn tự động từ so sánh Decision Tree, Logistic Regression và SVM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, tags=["Demo UI"])
def demo_page() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>Spam Email Classifier API</h1><p>Xem tài liệu tại <a href='/docs'>/docs</a>.</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    best_metrics = service.metadata.get("best_metrics", {}) if service.metadata else {}
    return HealthResponse(
        status="ok" if service.is_loaded else "degraded",
        model_loaded=service.is_loaded,
        model_name=service.model_name if service.is_loaded else None,
        accuracy=best_metrics.get("accuracy"),
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info() -> ModelInfoResponse:
    if not service.is_loaded or not service.metadata:
        raise HTTPException(status_code=503, detail="Mô hình chưa sẵn sàng.")
    return ModelInfoResponse(**service.metadata)


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_email(payload: PredictRequest) -> PredictResponse:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Mô hình chưa được nạp.")
    return service.predict_one(payload)


@app.post("/api/v1/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Mô hình chưa được nạp.")
    results, elapsed_ms = service.predict_batch(payload.emails)
    spam_count = sum(1 for r in results if r.label_id == 1)
    return BatchPredictResponse(
        total=len(results),
        spam_count=spam_count,
        ham_count=len(results) - spam_count,
        model_name=service.model_name,
        processing_time_ms=elapsed_ms,
        results=results,
    )
