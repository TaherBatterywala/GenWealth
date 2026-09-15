"""
GenWealth — Health Check Router
================================
File: app/routers/health.py

GET /api/v1/health
    Returns API server status, DB connectivity, API key counts,
    model artifact health, and process uptime.
"""

import time
import logging
from fastapi import APIRouter

from app.config import get_settings
from app.schemas.response_schemas import HealthResponse
from src.advisor.model_loader import check_artifact_health, MODEL_CACHE

logger = logging.getLogger("genwealth.api.health")

router = APIRouter(prefix="/api/v1/health", tags=["Health"])

# Process start time for uptime calculation
_START_TIME = time.time()


@router.get("", response_model=HealthResponse, summary="API Health Diagnostics")
async def health_check() -> HealthResponse:
    """
    Comprehensive health check for all GenWealth subsystems.

    Checks:
    - MongoDB Atlas connectivity (ping)
    - Groq & Gemini API key pool counts
    - Model artifact file integrity (no deserialization — fast)
    - Current model cache contents
    - Process uptime

    Returns HTTP 200 always; ``status`` field signals degraded state.
    """
    settings = get_settings()
    artifacts = check_artifact_health()

    # ── MongoDB ping (non-blocking, 3 s timeout) ─────────────────────────────
    mongo_status = "not_checked"
    try:
        from src.advisor.vector_store import get_mongo_collection
        col = get_mongo_collection()
        col.database.client.admin.command("ping", serverSelectionTimeoutMS=3_000)
        mongo_status = "connected"
    except Exception as exc:
        mongo_status = f"error: {type(exc).__name__}"
        logger.warning("[Health] MongoDB ping failed: %s", exc)

    # ── Determine overall status ─────────────────────────────────────────────
    critical_ok = (
        artifacts["lstm_ok"]
        and artifacts["rf_ok"]
        and mongo_status == "connected"
    )
    status = "ok" if critical_ok else "degraded"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        mongodb=mongo_status,
        groq_keys=len(settings.groq_keys),
        gemini_keys=len(settings.gemini_keys),
        lstm_ok=artifacts["lstm_ok"],
        rf_ok=artifacts["rf_ok"],
        ppo_zip_ok=artifacts["ppo_zip_ok"],
        ppo_conflict=artifacts["ppo_conflict"],
        cache_loaded=artifacts["cache_loaded"],
        uptime_s=round(time.time() - _START_TIME, 1),
    )
