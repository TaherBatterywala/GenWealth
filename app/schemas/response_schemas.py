"""
GenWealth — Pydantic Response Schemas
======================================
File: app/schemas/response_schemas.py

All outbound API response bodies are defined here.
These are also used as OpenAPI documentation via FastAPI's automatic schema generation.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ===========================================================================
# Sub-models
# ===========================================================================

class Phase1Result(BaseModel):
    """Quantitative signal snapshot from Phase 1 (LSTM + RF + FinBERT)."""
    ticker:          str
    as_of_date:      str
    close_price:     float
    currency:        str
    exchange:        str = "US Markets"   # human-readable exchange name
    log_ret:         float
    vol_20:          float
    vol_ratio:       float
    ret_1m:          float
    efficiency:      float
    sentiment:       float
    lstm_prob:       Optional[float] = None
    rf_prob:         Optional[float] = None
    phase1_signal:   float
    signal_label:    str              # "BULLISH" | "NEUTRAL" | "BEARISH"
    inference_mode:  str              # "LIVE_MODEL" | "MOMENTUM_PROXY" | "CSV_CACHE"
    news_headlines:  list[str] = Field(default_factory=list)


class Phase2Result(BaseModel):
    """PPO backtest metrics and implied portfolio allocation from Phase 2."""
    strategy:              str
    total_return_pct:      float
    equal_weight_ret_pct:  float
    sharpe_ratio:          float
    max_drawdown_pct:      float
    calmar_ratio:          float
    final_value_usd:       float
    alpha_vs_equal_weight: str
    allocation:            dict[str, float]   # {ticker: weight_pct}
    cash_pct:              float
    allocation_method:     str = "BACKTEST_METRICS"   # or "PPO_LIVE" | "SMART_ALLOCATOR"


class ComplianceInfo(BaseModel):
    """SEBI / SEC compliance guardrail output."""
    flags_detected:    int
    flag_details:      list[str]
    disclaimer_added:  bool


class RAGDocument(BaseModel):
    """Single RAG knowledge base document."""
    title:            str
    source:           str
    url:              str
    snippet:          str
    similarity_score: float
    timestamp:        Optional[str] = None


# ===========================================================================
# Primary Response Models
# ===========================================================================

class AdvisoryReportResponse(BaseModel):
    """
    Full advisory pipeline response for ``POST /api/v1/advisor/analyze``.

    Returned as the final SSE event payload (``event: complete``).
    """
    ticker:       str
    report:       str           # Full 4-stage Markdown advisory report
    phase1:       Optional[Phase1Result]  = None
    phase2:       Optional[Phase2Result]  = None
    rag_docs:     list[RAGDocument]       = Field(default_factory=list)
    compliance:   Optional[ComplianceInfo] = None
    intent:       str = "SINGLE_TICKER_NEWS"
    latency_s:    float = 0.0
    error:        Optional[str] = None


class AllocationWeight(BaseModel):
    """Single ticker allocation entry."""
    ticker:   str
    weight:   float       # 0.0 – 1.0
    capital:  float       # Allocated capital in base currency
    currency: str
    signal:   float
    stance:   str


class PortfolioAllocResponse(BaseModel):
    """Response for ``POST /api/v1/portfolio/allocate``."""
    portfolio_name:    str
    total_capital:     float
    currency:          str
    cash_reserved:     float
    cash_pct:          float
    allocations:       list[AllocationWeight]
    ppo_metrics:       Optional[Phase2Result] = None
    allocation_method: str       # "PPO_LIVE" | "SMART_ALLOCATOR"
    latency_s:         float = 0.0


class LedgerEntryResponse(BaseModel):
    """Single trade ledger entry for simulation responses."""
    step:          int
    date:          str
    ticker:        str
    action:        str       # BUY | TRIM_PROFIT | STOP_LOSS_EXIT | VOL_EXIT | CLOSE_ALL
    trigger:       str
    exec_price:    float
    shares_traded: float
    capital_in:    float
    capital_out:   float
    realized_pnl:  float
    cash_balance:  float
    position_roi:  float
    signal:        float
    vol_ratio:     float
    ai_reason:     str


class SimResultResponse(BaseModel):
    """Response for ``POST /api/v1/simulate/dynamic`` and ``/walkforward``."""
    portfolio_name:        str
    currency:              str
    symbol:                str
    initial_capital:       float
    final_cash:            float
    final_positions_value: float
    total_realized_pnl:    float
    total_roi_pct:         float
    win_rate_pct:          float
    max_drawdown_pct:      float
    n_trades:              int
    ledger:                list[LedgerEntryResponse] = Field(default_factory=list)
    portfolio_curve:       list[list[Any]]           = Field(default_factory=list)
    simulation_type:       str = "DYNAMIC"           # "DYNAMIC" | "WALK_FORWARD"
    latency_s:             float = 0.0


class HealthResponse(BaseModel):
    """Response for ``GET /api/v1/health``."""
    status:            str          # "ok" | "degraded" | "error"
    version:           str
    mongodb:           str          # "connected" | "error: ..."
    groq_keys:         int          # count of available Groq keys
    gemini_keys:       int          # count of available Gemini keys
    lstm_ok:           bool
    rf_ok:             bool
    ppo_zip_ok:        bool
    ppo_conflict:      str          # "OK" | "DETECTED: ..." | "UNKNOWN"
    cache_loaded:      list[str]
    uptime_s:          float


class ChatStreamChunk(BaseModel):
    """
    Single SSE chunk from ``POST /api/v1/chat/stream``.

    Yielded as JSON for each token/sentence fragment:
        data: {"type": "token", "content": "..."}
    Final message:
        data: {"type": "done", "content": "", "disclaimer_added": true}
    """
    type:              str   # "token" | "done" | "error"
    content:           str
    disclaimer_added:  bool = False
