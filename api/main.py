"""FastAPI application for LLM Evaluation Dashboard."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="LLM Evaluation Dashboard",
    description="Dashboard API for viewing evaluation results and model comparisons.",
    version="0.1.0",
)

from api.routes import evaluations, models  # noqa: E402

app.include_router(evaluations.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}