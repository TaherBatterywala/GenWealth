"""
GenWealth AI — FastAPI Application Entrypoint
==============================================
File: app/main.py

Production-ready FastAPI application serving:
  • REST & SSE API endpoints under /api/v1/...
  • Vanilla HTML5/CSS/JS SPA from frontend/ (single-port deployment)

Startup (lifespan):
  1. Validate environment configuration (Settings).
  2. Pre-load LSTM + RF model artifacts into model_loader cache.
  3. Log cache warm status.

Shutdown (lifespan):
  1. Close MongoDB connection if open.

Static Files:
  The frontend/ directory is mounted at "/" AFTER all API routers are
  registered, so /api/v1/... routes take priority over the static catch-all.
  Requires: aiofiles (`pip install aiofiles`)

Usage:
  Development:
    uvicorn app.main:app --reload --port 8000

  Production:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Logging — configure before any imports that use logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)-35s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("genwealth.main")

# ---------------------------------------------------------------------------
# Project root on sys.path (enables `from src.advisor...` imports)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Internal imports (after sys.path setup)
# ---------------------------------------------------------------------------
from app.config import get_settings
from app.routers import health, advisor, portfolio, simulation, chat, market


# ===========================================================================
# Lifespan (startup / shutdown)
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:
      - Validate Settings (raises EnvironmentError if keys missing).
      - Pre-load LSTM + RF model artifacts.
      - Log cache status.

    Shutdown:
      - Close MongoDB client if a connection was opened.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  GenWealth AI — FastAPI Server Starting")
    logger.info("=" * 60)

    # 1. Validate settings (will raise EnvironmentError on missing keys)
    try:
        settings = get_settings()
        logger.info("Settings loaded: %r", settings)
    except EnvironmentError as exc:
        logger.critical("STARTUP FAILED — configuration error: %s", exc)
        raise

    # 2. Pre-load model artifacts into cache
    logger.info("Pre-loading model artifacts…")
    try:
        from src.advisor.model_loader import warm_cache
        cache_status = warm_cache()
        for name, status in cache_status.items():
            icon = "[OK]" if status is True else "[!]"
            logger.info("  %s  %s: %s", icon, name, status)
    except Exception as exc:
        logger.warning("Model warm-cache failed (non-fatal): %s", exc)

    logger.info("Server ready.  Docs: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield   # ← Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("GenWealth AI — shutting down…")
    try:
        from src.advisor.vector_store import _mongo_client
        if _mongo_client is not None:
            _mongo_client.close()
            logger.info("MongoDB connection closed.")
    except Exception:
        pass
    logger.info("Shutdown complete.")


# ===========================================================================
# Application Factory
# ===========================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    Called once at module load. The singleton ``app`` object below is what
    uvicorn uses.
    """
    settings = None
    try:
        settings = get_settings()
    except EnvironmentError:
        pass   # Settings validation runs fully in lifespan

    app = FastAPI(
        title="GenWealth AI API",
        version="4.0.0",
        description=(
            "Production-grade quantitative trading + AI advisory engine. "
            "Phases 1–3 (LSTM, RF, FinBERT, PPO, LLM) wrapped in a REST + SSE API."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS Middleware ───────────────────────────────────────────────────────
    cors_origins = settings.cors_origins if settings else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Latency-S"],
    )

    # ── Global Exception Handlers ─────────────────────────────────────────────
    @app.exception_handler(ValidationError)
    async def pydantic_validation_error_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": "Validation failed", "detail": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path, exc, exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    # ── API Routers (MUST be registered BEFORE StaticFiles mount) ────────────
    app.include_router(health.router)
    app.include_router(advisor.router)
    app.include_router(portfolio.router)
    app.include_router(simulation.router)
    app.include_router(chat.router)
    app.include_router(market.router)

    # ── Static Files (Vanilla JS SPA) ─────────────────────────────────────────
    # Mounted at "/" — catches everything not matched by the API routers above.
    # `html=True` enables serving index.html for all unmatched paths (SPA routing).
    frontend_dir = _ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
        logger.info("Frontend static files mounted from '%s'", frontend_dir)
    else:
        logger.warning(
            "Frontend directory not found at '%s'. "
            "Only API endpoints will be available.", frontend_dir
        )

    return app


# ===========================================================================
# Application Singleton
# ===========================================================================

app = create_app()


# ===========================================================================
# Dev Entry Point
# ===========================================================================

if __name__ == "__main__":
    """
    Direct-run entry point for development convenience.
    Prefer: uvicorn app.main:app --reload --port 8000
    """
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
