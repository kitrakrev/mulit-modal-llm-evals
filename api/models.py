"""Pydantic models for LLM Evaluation API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    """Evaluation result for a single model response."""

    id: str = Field(..., description="Unique identifier for the evaluation")
    model_name: str = Field(..., description="Name of the evaluated model")
    prompt: str = Field(..., description="Input prompt")
    response: str = Field(..., description="Model response")
    metrics: dict[str, float] = Field(default_factory=dict, description="Metric scores")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    cost_usd: float = Field(..., description="Cost in USD")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelComparison(BaseModel):
    """Comparison of multiple models on the same prompt."""

    prompt: str = Field(..., description="Input prompt")
    models: list[ModelResult] = Field(..., description="Results for each model")
    best_model: str = Field(..., description="Model with highest average score")


class ModelResult(BaseModel):
    """Result for a single model in comparison."""

    model_name: str
    response: str
    metrics: dict[str, float]
    latency_ms: float
    cost_usd: float
    average_score: float


class RegressionAlert(BaseModel):
    """Alert for model regression."""

    model_name: str = Field(..., description="Model that regressed")
    metric_name: str = Field(..., description="Metric that degraded")
    previous_score: float = Field(..., description="Previous average score")
    current_score: float = Field(..., description="Current average score")
    percent_change: float = Field(..., description="Percentage change")
    detected_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.1.0"