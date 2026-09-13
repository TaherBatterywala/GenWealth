# GenWealth — Advisor Engine Package
# Phase 3: Hybrid RAG Pipeline + Multi-LLM Reasoning + Dynamic Trading Simulator
#
# Public API:
#   from src.advisor.vector_store            import query_knowledge_base, get_live_stock_news
#   from src.advisor.context_builder         import ContextAggregator
#   from src.advisor.llm_engine              import LLMAdvisorEngine, generate_groq_report
#   from src.advisor.guardrails              import sanitize_and_append_disclaimer
#   from src.advisor.trade_execution_simulator import (
#       DynamicTradeSimulator, WalkForwardSimulator,
#       DayEvent, RegimePortfolioResult,
#       fetch_regime_data, split_regime,
#       compute_ticker_signals, compute_allocation_weights,
#       discover_tickers,
#   )

from src.advisor.vector_store    import query_knowledge_base, get_live_stock_news, FRESHNESS_HOURS
from src.advisor.context_builder import ContextAggregator
from src.advisor.llm_engine      import (
    LLMAdvisorEngine,
    classify_query_intent,
    generate_gemini_report,
    generate_groq_report,
    verify_report_accuracy,
)
from src.advisor.guardrails      import sanitize_and_append_disclaimer, get_compliance_flags
from src.advisor.trade_execution_simulator import (
    # Dynamic 30-day step simulator
    DynamicTradeSimulator,
    SmartPortfolioAllocator,
    LedgerEntry,
    Position,
    SimulationResult,
    # Walk-forward 5-day historical backtester
    WalkForwardSimulator,
    DayEvent,
    RegimePortfolioResult,
    # Data utilities
    fetch_regime_data,
    split_regime,
    # Signal + allocation utilities
    compute_ticker_signals,
    compute_allocation_weights,
    momentum_signal,
    rolling_vol_ratio,
    extract_stance,
    extract_ai_reason,
    # Autonomous discovery
    discover_tickers,
    # Price loading
    load_price_history,
)

__all__ = [
    # ── vector_store ──────────────────────────────────────────────────────────
    "query_knowledge_base",
    "get_live_stock_news",
    "FRESHNESS_HOURS",
    # ── context_builder ───────────────────────────────────────────────────────
    "ContextAggregator",
    # ── llm_engine ────────────────────────────────────────────────────────────
    "LLMAdvisorEngine",
    "classify_query_intent",
    "generate_gemini_report",
    "generate_groq_report",
    "verify_report_accuracy",
    # ── guardrails ────────────────────────────────────────────────────────────
    "sanitize_and_append_disclaimer",
    "get_compliance_flags",
    # ── trade_execution_simulator — dynamic ───────────────────────────────────
    "DynamicTradeSimulator",
    "SmartPortfolioAllocator",
    "LedgerEntry",
    "Position",
    "SimulationResult",
    # ── trade_execution_simulator — walk-forward ──────────────────────────────
    "WalkForwardSimulator",
    "DayEvent",
    "RegimePortfolioResult",
    # ── data utilities ────────────────────────────────────────────────────────
    "fetch_regime_data",
    "split_regime",
    # ── signal + allocation utilities ────────────────────────────────────────
    "compute_ticker_signals",
    "compute_allocation_weights",
    "momentum_signal",
    "rolling_vol_ratio",
    "extract_stance",
    "extract_ai_reason",
    # ── autonomous discovery ──────────────────────────────────────────────────
    "discover_tickers",
    # ── price loading ─────────────────────────────────────────────────────────
    "load_price_history",
]
