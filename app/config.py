"""
GenWealth — FastAPI Application Config
=======================================
File: app/config.py

Loads and validates all environment variables required by the API server.
Uses Pydantic Settings for validation with automatic .env loading.

Environment variables expected in .env:
    MONGODB_URI        — Atlas connection string
    GROQ_API_KEY       — Primary Groq key
    GROQ_API_KEY2      — (optional) secondary Groq key
    GEMINI_API_KEY     — Primary Gemini key
    GEMINI_API_KEY2–6  — (optional) additional Gemini keys
    CORS_ORIGINS       — comma-separated allowed origins (default: *)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class Settings:
    """
    Central configuration object for the GenWealth API.

    All values are read from environment variables (loaded from .env).
    Missing required values raise EnvironmentError at startup.
    """

    def __init__(self) -> None:
        # ── MongoDB ───────────────────────────────────────────────────────────
        self.mongodb_uri: str = self._require("MONGODB_URI")

        # ── Groq key pool ─────────────────────────────────────────────────────
        self.groq_keys: list[str] = self._collect_pool("GROQ_API_KEY")
        if not self.groq_keys:
            raise EnvironmentError(
                "No GROQ_API_KEY* found in environment. "
                "Add at least GROQ_API_KEY to .env."
            )

        # ── Gemini key pool ───────────────────────────────────────────────────
        self.gemini_keys: list[str] = self._collect_pool("GEMINI_API_KEY")
        if not self.gemini_keys:
            raise EnvironmentError(
                "No GEMINI_API_KEY* found in environment. "
                "Add at least GEMINI_API_KEY to .env."
            )

        # ── CORS ──────────────────────────────────────────────────────────────
        raw_origins = os.getenv("CORS_ORIGINS", "*")
        if raw_origins.strip() == "*":
            self.cors_origins: list[str] = ["*"]
        else:
            self.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

        # ── App metadata ──────────────────────────────────────────────────────
        self.app_title:   str = "GenWealth AI API"
        self.app_version: str = "4.0.0"
        self.frontend_dir: Path = _PROJECT_ROOT / "frontend"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _require(key: str) -> str:
        """Return env var value or raise EnvironmentError."""
        val = os.getenv(key, "").strip()
        if not val:
            raise EnvironmentError(
                f"Required environment variable '{key}' is missing or empty."
            )
        return val

    @staticmethod
    def _collect_pool(prefix: str) -> list[str]:
        """
        Collect all env vars whose names START WITH ``prefix``.
        E.g. prefix="GROQ_API_KEY" matches GROQ_API_KEY, GROQ_API_KEY2, GROQ_API_KEY3 …
        Returns a list of non-empty values in natural sort order.
        """
        keys = sorted(
            k for k in os.environ if k.startswith(prefix)
        )
        return [os.environ[k].strip() for k in keys if os.environ[k].strip()]

    def __repr__(self) -> str:
        return (
            f"Settings(groq_keys={len(self.groq_keys)}, "
            f"gemini_keys={len(self.gemini_keys)}, "
            f"cors={self.cors_origins})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Uses ``functools.lru_cache`` so the .env is only read once per process.
    FastAPI dependency injection usage::

        from app.config import get_settings
        @router.get("/")
        def endpoint(settings: Settings = Depends(get_settings)):
            ...
    """
    return Settings()
