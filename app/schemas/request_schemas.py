"""
GenWealth — Pydantic Request Schemas
=====================================
File: app/schemas/request_schemas.py

All inbound API request bodies are validated here before reaching
the route handlers. Validation failures automatically return HTTP 422.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# Advisory Router
# ===========================================================================

class TickerQuery(BaseModel):
    """
    Request body for ``POST /api/v1/advisor/analyze``.

    Attributes:
        ticker:          Yahoo Finance ticker symbol (e.g. "NVDA", "RELIANCE.NS").
        use_live_engine: When True (default), triggers LiveQuantEngine for
                         tickers outside the Phase 1 CSV training set.
    """
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=["NVDA", "RELIANCE.NS", "ASML.AS"],
        description="Yahoo Finance ticker symbol",
    )
    use_live_engine: bool = Field(
        default=True,
        description="Use real-time LSTM+RF+FinBERT inference for non-CSV tickers",
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        return v.strip().upper()


# ===========================================================================
# Portfolio Router
# ===========================================================================

class PortfolioAllocRequest(BaseModel):
    """
    Request body for ``POST /api/v1/portfolio/allocate``.

    Attributes:
        tickers:  List of Yahoo Finance ticker symbols (1–20 tickers).
        capital:  Total portfolio capital in the base currency.
        currency: ISO currency code (e.g. "USD", "INR").
    """
    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        examples=[["NVDA", "AAPL", "MSFT"]],
        description="List of ticker symbols to allocate across",
    )
    capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Total capital to allocate (must be positive)",
    )
    currency: str = Field(
        default="USD",
        max_length=5,
        description="Base currency code (e.g. USD, INR)",
    )

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("tickers list must contain at least one non-empty symbol")
        return list(dict.fromkeys(cleaned))  # deduplicate preserving order


# ===========================================================================
# Simulation Router
# ===========================================================================

class SimulationRequest(BaseModel):
    """
    Request body for ``POST /api/v1/simulate/dynamic``.

    Runs the 30-trading-day DynamicTradeSimulator.
    """
    portfolio_name: str = Field(
        default="GenWealth Portfolio",
        max_length=60,
        description="Display name for this portfolio",
    )
    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        examples=[["NVDA", "AAPL"]],
    )
    capital: float = Field(
        default=100_000.0,
        gt=0,
        description="Starting capital",
    )

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v if t.strip()]


class WalkForwardRequest(BaseModel):
    """
    Request body for ``POST /api/v1/simulate/walkforward``.

    Runs the 5-day zero-lookahead WalkForwardSimulator over a historical regime.
    """
    portfolio_name: str = Field(
        default="GenWealth WF Portfolio",
        max_length=60,
    )
    tickers: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        examples=[["NVDA", "AAPL"]],
    )
    capital: float = Field(
        default=100_000.0,
        gt=0,
    )
    regime_start: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-01-02"],
        description="Walk-forward window start date (YYYY-MM-DD)",
    )
    regime_end: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2024-01-08"],
        description="Walk-forward window end date (YYYY-MM-DD, max 5 trading days after start)",
    )

    @field_validator("tickers")
    @classmethod
    def normalise_tickers(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v if t.strip()]


# ===========================================================================
# Chat Router
# ===========================================================================

class ChatMessage(BaseModel):
    """
    Request body for ``POST /api/v1/chat/stream``.

    Attributes:
        message:              Natural-language user query.
        ticker:               Optional ticker context (e.g. the active ticker
                              in the frontend's Advisory tab).
        conversation_history: Previous turns as list of {"role","content"} dicts.
                              Max 10 turns retained for context window management.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's natural-language query",
        examples=["Why did you recommend trimming NVDA?"],
    )
    ticker: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Active ticker for context grounding (optional)",
    )
    conversation_history: list[dict] = Field(
        default_factory=list,
        max_length=20,
        description="Prior conversation turns: [{role: user|assistant, content: str}]",
    )

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().upper() if v and v.strip() else None
