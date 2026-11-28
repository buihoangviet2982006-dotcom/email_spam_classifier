"""Pydantic request and response schemas for the Spam Email Classifier API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    subject: str = Field(default="", description="Tiêu đề email (tùy chọn)")
    text: str = Field(..., min_length=1, description="Nội dung email cần phân loại")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ngưỡng xác suất để kết luận SPAM (mặc định 0.5)",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Nội dung email (text) không được để trống.")
        return value


class KeywordContribution(BaseModel):
    keyword: str
    weight: float
    direction: Literal["SPAM", "HAM"]


class PredictResponse(BaseModel):
    label: Literal["SPAM", "HAM"]
    label_id: int = Field(..., description="1 = SPAM, 0 = HAM")
    spam_probability: float
    ham_probability: float
    confidence: float
    threshold: float
    model_name: str
    processing_time_ms: float
    top_keywords: list[KeywordContribution] = Field(default_factory=list)


class BatchPredictRequest(BaseModel):
    emails: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Danh sách email cần phân loại (tối đa 100 email/lần)",
    )


class BatchPredictResponse(BaseModel):
    total: int
    spam_count: int
    ham_count: int
    model_name: str
    processing_time_ms: float
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    accuracy: float | None = None


class ModelInfoResponse(BaseModel):
    best_model_key: str
    best_model_name: str
    source_model_file: str
    selection_criterion: str
    test_samples: int
    selected_at: str
    best_metrics: dict[str, Any]
    comparison: list[dict[str, Any]]
