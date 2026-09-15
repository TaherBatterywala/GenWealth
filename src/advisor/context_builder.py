"""
GenWealth Advisor Engine — Phase 3, Step 3.2
============================================
Module: src/advisor/context_builder.py

Context Aggregation Engine: bridges Phase 1 (Quant), Phase 2 (RL/PPO),
and Phase 3.1 (RAG Vector Store) into a single unified payload ready
for LLM reasoning (Google Gemini / Groq).

Data Flow:
  ┌────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
  │  Phase 1 CSV       │   │  Phase 2 Backtest   │   │  Phase 3.1 MongoDB   │
  │  enriched_rl_data  │   │  rl_backtest_metrics │   │  financial_knowledge │
  │  .csv              │   │  _v2.pkl            │   │  (RAG vector store)  │
  └────────┬───────────┘   └──────────┬──────────┘   └──────────┬───────────┘
           │                          │                          │
           ▼                          ▼                          ▼
  get_phase1_snapshot()    get_phase2_allocation()      get_rag_context()
           │                          │                          │
           └──────────────────────────┴──────────────────────────┘
                                      │
                                      ▼
                          ContextAggregator.build_ticker_context()
                                      │
                                      ▼
                          ContextAggregator.build_llm_prompt_context()
                          (Markdown string → ready for Gemini / Groq)
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
from src.advisor.vector_store import query_knowledge_base

# Live inference engine (lazy import to avoid circular deps at module load)
# LiveSignalResult is imported only when needed.
_live_engine_available: bool = False
try:
    from src.advisor.live_inference import LiveQuantEngine, LiveSignalResult, _signal_label
    _live_engine_available = True
except Exception:
    pass  # Live inference not available — CSV-only mode

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("genwealth.context_builder")

# ---------------------------------------------------------------------------
# Path Constants (resolved relative to project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/advisor/ → root
_CSV_PATH     = _PROJECT_ROOT / "data" / "enriched_rl_data.csv"
_PKL_PATH     = _PROJECT_ROOT / "model_artifacts" / "rl_backtest_metrics_v2.pkl"

# Tickers tracked by the RL environment (order matches the env's asset list)
_RL_TICKERS: list[str] = ["HDFCBANK.NS", "NVDA", "RELIANCE.NS", "TCS.NS"]

# ---------------------------------------------------------------------------
# Module-level cache (lazy-loaded once per process)
# ---------------------------------------------------------------------------
_phase1_df: Optional[pd.DataFrame] = None
_phase2_metrics: Optional[dict] = None
_live_engine_instance: Optional[Any] = None  # LiveQuantEngine singleton


# ---------------------------------------------------------------------------
# Safe Numeric Formatter
# ---------------------------------------------------------------------------

def _fmt_num(value: Any, fmt: str = ".4f", prefix: str = "", suffix: str = "") -> str:
    """
    Safely format a numeric value with a given format spec.

    Returns 'N/A' if the value is not a valid number (e.g., when Phase 1 data
    is unavailable and a string placeholder is returned instead of a float).
    This prevents ``ValueError: Unknown format code 'f' for object of type 'str'``
    when tickers outside the Phase 1 training set are queried.

    Args:
        value  : The value to format (ideally numeric).
        fmt    : Python format spec string (default ``'.4f'``).
        prefix : Optional prefix to prepend if value is valid (e.g. ``'$'``).
        suffix : Optional suffix to append if value is valid (e.g. ``'%'``).

    Returns:
        str: Formatted value string, or ``'N/A'`` on type mismatch.
    """
    try:
        return f"{prefix}{value:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


# ===========================================================================
# Helpers — data loaders with caching
# ===========================================================================

def _load_phase1_df() -> pd.DataFrame:
    """
    Load and cache the Phase 1 enriched feature CSV.

    Returns:
        pd.DataFrame: Full dataset with columns Date, Ticker, Close_Price,
                      Log_Ret, Vol_20, Vol_200, Vol_Ratio, Ret_1M, Ret_3M,
                      Efficiency, Vol_Shock, Ret_Lag*, Sentiment, Phase1_Signal.

    Raises:
        FileNotFoundError: If enriched_rl_data.csv is missing.
    """
    global _phase1_df
    if _phase1_df is None:
        if not _CSV_PATH.exists():
            raise FileNotFoundError(
                f"Phase 1 data not found at: {_CSV_PATH}\n"
                "Run the Phase 1 pipeline to generate enriched_rl_data.csv."
            )
        logger.info("Loading Phase 1 CSV from '%s'…", _CSV_PATH)
        _phase1_df = pd.read_csv(_CSV_PATH, parse_dates=["Date"])
        logger.info(
            "Phase 1 data loaded: %d rows, %d tickers.",
            len(_phase1_df), _phase1_df["Ticker"].nunique(),
        )
    return _phase1_df


def _load_phase2_metrics() -> dict:
    """
    Load and cache the Phase 2 PPO backtest metrics pickle.

    The pickle contains three strategy keys:
        'PPO Agent', 'Equal-Weight', 'Buy-and-Hold'
    Each maps to a dict with:
        total_return, sharpe, max_drawdown, calmar, portfolio (list[float])

    Returns:
        dict: Full backtest metrics dictionary.

    Raises:
        FileNotFoundError: If rl_backtest_metrics_v2.pkl is missing.
    """
    global _phase2_metrics
    if _phase2_metrics is None:
        if not _PKL_PATH.exists():
            raise FileNotFoundError(
                f"Phase 2 backtest metrics not found at: {_PKL_PATH}\n"
                "Run the Phase 2 PPO training pipeline first."
            )
        logger.info("Loading Phase 2 backtest metrics from '%s'…", _PKL_PATH)
        _phase2_metrics = joblib.load(_PKL_PATH)
        logger.info("Phase 2 metrics loaded. Strategies: %s", list(_phase2_metrics.keys()))
    return _phase2_metrics


# ===========================================================================
# 1. Phase 1 Snapshot
# ===========================================================================

def get_phase1_snapshot(
    ticker: str,
    live_signal_override: "Optional[Any]" = None,
) -> dict[str, Any]:
    """
    Extract the most recent Phase 1 feature snapshot for a given ticker.

    Priority:
        1. If ``live_signal_override`` is provided (a ``LiveSignalResult``), use it
           directly — bypasses the CSV entirely. Used for non-CSV tickers.
        2. Otherwise reads the latest row from ``data/enriched_rl_data.csv``.

    Args:
        ticker (str): Ticker symbol exactly as stored in the CSV,
                      e.g. "NVDA", "HDFCBANK.NS", "RELIANCE.NS", "TCS.NS".
        live_signal_override: Optional ``LiveSignalResult`` dataclass from
                              ``live_inference.LiveQuantEngine.compute_live_signal()``.
                              When supplied, skips CSV lookup entirely.

    Returns:
        dict: A snapshot dictionary with the following keys:
            - ``ticker``         (str):   Requested ticker.
            - ``as_of_date``     (str):   ISO date string of the latest row.
            - ``close_price``    (float): Latest closing price.
            - ``log_ret``        (float): Last daily log return.
            - ``vol_20``         (float): 20-day realised volatility.
            - ``vol_ratio``      (float): Vol_20 / Vol_200 ratio (regime signal).
            - ``ret_1m``         (float): 1-month cumulative return.
            - ``efficiency``     (float): Efficiency ratio (trend quality metric).
            - ``sentiment``      (float): FinBERT composite sentiment score (-1 to 1).
            - ``phase1_signal``  (float): Blended LSTM + RF signal (0 to 1).
            - ``inference_mode`` (str):   "LIVE_MODEL" | "MOMENTUM_PROXY" | "CSV_CACHE".

    Raises:
        ValueError: If the ticker is not found in the CSV and no override is given.
    """
    ticker = ticker.upper()

    # ── Path 1: Live inference override ─────────────────────────────────────
    if live_signal_override is not None:
        lr = live_signal_override
        snapshot = {
            "ticker":         lr.ticker,
            "as_of_date":     lr.as_of_date,
            "close_price":    lr.close_price,
            "log_ret":        lr.log_ret,
            "vol_20":         lr.vol_20,
            "vol_ratio":      lr.vol_ratio,
            "ret_1m":         lr.ret_1m,
            "efficiency":     lr.efficiency,
            "sentiment":      lr.sentiment,
            "phase1_signal":  lr.phase1_signal,
            "inference_mode": lr.inference_mode,
        }
        logger.info(
            "Phase 1 snapshot for '%s' via live engine: signal=%.4f, sentiment=%.4f [%s]",
            ticker, snapshot["phase1_signal"], snapshot["sentiment"],
            snapshot["inference_mode"],
        )
        return snapshot

    # ── Path 2: CSV lookup ────────────────────────────────────────────────────
    df = _load_phase1_df()
    ticker_df = df[df["Ticker"] == ticker].sort_values("Date")

    if ticker_df.empty:
        available = df["Ticker"].unique().tolist()
        raise ValueError(
            f"Ticker '{ticker}' not found in Phase 1 data.\n"
            f"Available tickers: {available}"
        )

    latest = ticker_df.iloc[-1]
    snapshot = {
        "ticker":         ticker,
        "as_of_date":     str(latest["Date"].date()),
        "close_price":    round(float(latest["Close_Price"]), 4),
        "log_ret":        round(float(latest["Log_Ret"]), 6),
        "vol_20":         round(float(latest["Vol_20"]), 6),
        "vol_ratio":      round(float(latest["Vol_Ratio"]), 4),
        "ret_1m":         round(float(latest["Ret_1M"]), 6),
        "efficiency":     round(float(latest["Efficiency"]), 4),
        "sentiment":      round(float(latest["Sentiment"]), 4),
        "phase1_signal":  round(float(latest["Phase1_Signal"]), 4),
        "inference_mode": "CSV_CACHE",
    }

    logger.info(
        "Phase 1 snapshot for '%s' as of %s: signal=%.4f, sentiment=%.4f [CSV]",
        ticker, snapshot["as_of_date"], snapshot["phase1_signal"], snapshot["sentiment"],
    )
    return snapshot


# ===========================================================================
# 2. Phase 2 Portfolio Allocation
# ===========================================================================

def get_phase2_allocation(ticker: str) -> dict[str, Any]:
    """
    Derive the PPO agent's implied portfolio allocation and key risk metrics.

    Since the backtest pickle stores the portfolio value time-series (not
    per-step weights), this function calculates:
      - Equal-weight allocation across RL tickers as the baseline.
      - Final portfolio value and total return from the PPO agent.
      - Sharpe ratio, max drawdown, and Calmar ratio.
      - The PPO agent's performance vs the Equal-Weight benchmark.

    Args:
        ticker (str): The ticker of interest (used for contextual logging).

    Returns:
        dict: A portfolio context dictionary with:
            - ``strategy``          (str):   "PPO Agent".
            - ``total_return_pct``  (float): Total backtest return in %.
            - ``sharpe_ratio``      (float): Annualised Sharpe ratio.
            - ``max_drawdown_pct``  (float): Maximum drawdown in %.
            - ``calmar_ratio``      (float): Calmar ratio (return / max DD).
            - ``final_value_usd``   (float): Final portfolio value ($).
            - ``vs_equal_weight``   (str):   Outperformance vs benchmark.
            - ``allocation``        (dict):  Equal-weight % across RL tickers.
            - ``cash_pct``          (float): Cash buffer percentage.
    """
    metrics = _load_phase2_metrics()
    ppo_data = metrics["PPO Agent"]
    ew_data  = metrics["Equal-Weight"]

    # Equal-weight allocation across the 4 RL tickers (5% cash buffer)
    cash_pct   = 5.0
    equity_pct = (100.0 - cash_pct) / len(_RL_TICKERS)
    allocation = {t: round(equity_pct, 2) for t in _RL_TICKERS}

    # PPO vs EW outperformance
    ppo_ret = ppo_data["total_return"]
    ew_ret  = ew_data["total_return"]
    # Alpha = PPO return minus benchmark return (percentage points)
    alpha_pp = round((ppo_ret - ew_ret) * 100, 2)
    alpha_sign = "+" if alpha_pp >= 0 else ""

    portfolio_series = ppo_data["portfolio"]
    final_value = portfolio_series[-1] if portfolio_series else 0.0

    result = {
        "strategy":              "PPO Agent",
        "total_return_pct":      round(ppo_ret * 100, 2),
        "equal_weight_ret_pct":  round(ew_ret * 100, 2),
        "sharpe_ratio":          round(ppo_data["sharpe"], 4),
        "max_drawdown_pct":      round(ppo_data["max_drawdown"] * 100, 2),
        "calmar_ratio":          round(ppo_data["calmar"], 4),
        "final_value_usd":       round(final_value, 2),
        # Alpha in percentage points vs equal-weight benchmark
        "alpha_vs_equal_weight": f"{alpha_sign}{alpha_pp}pp",
        "allocation":            allocation,
        "cash_pct":              cash_pct,
    }

    logger.info(
        "Phase 2 allocation (context for '%s'): return=%.1f%%, sharpe=%.3f, "
        "maxDD=%.1f%%",
        ticker, result["total_return_pct"], result["sharpe_ratio"],
        result["max_drawdown_pct"],
    )
    return result


# ===========================================================================
# 3. Phase 3.1 RAG Context
# ===========================================================================

def get_rag_context(ticker: str, top_k: int = 2) -> list[dict]:
    """
    Query the Phase 3.1 vector knowledge base for relevant news context.

    Automatically seeds MongoDB with live DuckDuckGo news if no documents
    exist for this ticker (cold-start fallback handled inside vector_store).

    Args:
        ticker (str): Ticker to search for, e.g. "NVDA".
        top_k  (int): Number of top-ranked documents to return (default 2).

    Returns:
        list[dict]: Top-k documents from the vector store, each containing:
            ``text_content``, ``source_type``, ``timestamp``,
            ``metadata`` (url, source, title), ``similarity_score``.
    """
    query = (
        f"{ticker} stock market outlook earnings revenue guidance "
        "analyst sentiment recent news"
    )
    logger.info("Querying RAG knowledge base for '%s' (top_k=%d)…", ticker, top_k)
    results = query_knowledge_base(ticker=ticker, query_text=query, top_k=top_k)
    logger.info("RAG returned %d document(s) for '%s'.", len(results), ticker)
    return results


# ===========================================================================
# 4. Context Aggregator Class
# ===========================================================================

class ContextAggregator:
    """
    Orchestrates data retrieval from all three GenWealth pipeline phases and
    assembles a unified context payload for LLM reasoning.

    Usage:
        agg = ContextAggregator()

        # Get structured JSON payload
        ctx = agg.build_ticker_context("NVDA")

        # Get formatted Markdown prompt ready for Gemini / Groq
        prompt = agg.build_llm_prompt_context("NVDA")
    """

    def build_ticker_context(
        self,
        ticker: str,
        use_live_engine: bool = True,
    ) -> dict[str, Any]:
        """
        Aggregate Phase 1 + Phase 2 + Phase 3.1 data for a single ticker.

        For tickers NOT in the Phase 1 CSV (i.e., outside the 4 trained tickers),
        the live inference engine (``LiveQuantEngine``) is invoked when
        ``use_live_engine=True`` to compute a real-time signal.

        Args:
            ticker (str):           Stock ticker symbol (e.g. "NVDA", "AAPL", "ASML.AS").
            use_live_engine (bool): If True, trigger ``LiveQuantEngine`` for non-CSV tickers.
                                    Defaults to True.

        Returns:
            dict: Unified context payload with three top-level keys:
                - ``phase1_quant``     (dict): Quantitative feature snapshot.
                - ``phase2_portfolio`` (dict): RL portfolio metrics & allocation.
                - ``phase3_rag``       (list): RAG news documents.
        """
        ticker = ticker.upper()
        logger.info("Building unified context for ticker '%s'…", ticker)

        # --- Phase 1: Quant snapshot -----------------------------------------
        # Try CSV first; if the ticker is not found AND live engine is enabled,
        # compute a real-time signal via LiveQuantEngine.
        live_signal = None
        try:
            phase1 = get_phase1_snapshot(ticker)   # CSV path
        except (ValueError, FileNotFoundError) as exc:
            logger.info(
                "Phase 1 CSV miss for '%s' (%s). Attempting live engine…", ticker, exc
            )
            if use_live_engine and _live_engine_available:
                try:
                    global _live_engine_instance
                    if _live_engine_instance is None:
                        from src.advisor.live_inference import LiveQuantEngine
                        _live_engine_instance = LiveQuantEngine()
                    live_signal = _live_engine_instance.compute_live_signal(ticker)
                    phase1 = get_phase1_snapshot(
                        ticker, live_signal_override=live_signal
                    )
                except Exception as live_exc:
                    logger.warning(
                        "Live engine also failed for '%s': %s", ticker, live_exc
                    )
                    phase1 = {"error": str(exc)}   # surface the original CSV error
            else:
                phase1 = {"error": str(exc)}

        # --- Phase 2: RL portfolio state -------------------------------------
        try:
            phase2 = get_phase2_allocation(ticker)
        except FileNotFoundError as exc:
            logger.warning("Phase 2 allocation failed: %s", exc)
            phase2 = {"error": str(exc)}

        # --- Phase 3.1: RAG news context ------------------------------------
        try:
            rag_docs = get_rag_context(ticker, top_k=2)
        except Exception as exc:
            logger.warning("RAG context failed for '%s': %s", ticker, exc)
            rag_docs = []

        context = {
            "ticker":           ticker,
            "phase1_quant":     phase1,
            "phase2_portfolio": phase2,
            "phase3_rag":       rag_docs,
            "live_signal":      live_signal,   # None for CSV-scope tickers
        }

        logger.info(
            "Unified context built for '%s' (%d RAG doc(s), live=%s).",
            ticker, len(rag_docs), live_signal is not None,
        )
        return context

    def build_llm_prompt_context(self, ticker: str) -> str:
        """
        Build a structured Markdown prompt string ready for Gemini / Groq.

        Calls ``build_ticker_context()`` internally and formats every field
        into a clean Markdown document that an LLM can directly read to
        generate an investment advisory response.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            str: A multi-section Markdown string with four sections:
                 [1] Quantitative Signal Summary
                 [2] RL Portfolio Allocation
                 [3] Live News & Market Sentiment (RAG)
                 [4] Advisory Task Instruction

        Example:
            >>> prompt = ContextAggregator().build_llm_prompt_context("NVDA")
            >>> print(prompt[:200])
        """
        ctx    = self.build_ticker_context(ticker)
        p1     = ctx.get("phase1_quant", {})
        p2     = ctx.get("phase2_portfolio", {})
        rag    = ctx.get("phase3_rag", [])
        ticker = ctx["ticker"]

        # ── Signal interpretation helpers ────────────────────────────────────
        signal_val = p1.get("phase1_signal", 0.5)
        if signal_val >= 0.65:
            signal_label = "BULLISH"
        elif signal_val <= 0.35:
            signal_label = "BEARISH"
        else:
            signal_label = "NEUTRAL"

        sentiment_val = p1.get("sentiment", 0.0)
        if sentiment_val >= 0.15:
            sent_label = "Positive"
        elif sentiment_val <= -0.15:
            sent_label = "Negative"
        else:
            sent_label = "Neutral"

        # ── Format RAG news snippets ─────────────────────────────────────────
        if rag:
            rag_section_lines = []
            for i, doc in enumerate(rag, 1):
                meta  = doc.get("metadata", {})
                title = meta.get("title", "Article")
                src   = meta.get("source", "Unknown")
                url   = meta.get("url", "N/A")
                score = doc.get("similarity_score", 0.0)
                text  = doc.get("text_content", "")[:350]
                rag_section_lines.append(
                    f"**[{i}] {title}**\n"
                    f"- Source: {src}  |  Relevance Score: {score:.3f}\n"
                    f"- URL: {url}\n"
                    f"- Snippet: {text}…\n"
                )
            rag_section = "\n".join(rag_section_lines)
        else:
            rag_section = "_No recent news articles found in the knowledge base._"

        # ── Format portfolio allocation table ────────────────────────────────
        alloc = p2.get("allocation", {})
        cash  = p2.get("cash_pct", 5.0)
        alloc_lines = [f"| {t:<14} | {w:>6.1f}% |" for t, w in alloc.items()]
        alloc_lines.append(f"| {'CASH':<14} | {cash:>6.1f}% |")
        alloc_table = (
            "| Ticker         | Weight  |\n"
            "|----------------|---------|\n"
            + "\n".join(alloc_lines)
        )

        # ── Assemble the full Markdown prompt ─────────────────────────────────
        # All numeric fields use _fmt_num() to safely handle the case where
        # Phase 1 data is unavailable (p1 = {"error": "..."}) and values
        # are None / string placeholders rather than floats.
        prompt = f"""# GenWealth AI — Advisory Context for {ticker}
> **As of:** {p1.get("as_of_date", "N/A")}  |  **Model:** PPO Multi-Asset v2  |  **Signal:** {signal_label}

---

## [1] Quantitative Signal Summary (Phase 1 — LSTM + RF + FinBERT)

| Metric                    | Value                        |
|---------------------------|------------------------------|
| Ticker                    | {ticker}                    |
| Closing Price             | {_fmt_num(p1.get("close_price"), ",.4f", prefix="$")}  |
| Daily Log Return          | {_fmt_num(p1.get("log_ret", 0), ".4%")}          |
| 20-Day Volatility         | {_fmt_num(p1.get("vol_20", 0), ".4%")}           |
| Vol Regime Ratio (20/200) | {_fmt_num(p1.get("vol_ratio", 0), ".3f")}        |
| 1-Month Return            | {_fmt_num(p1.get("ret_1m", 0), ".4%")}           |
| Efficiency Ratio          | {_fmt_num(p1.get("efficiency", 0), ".4f")}       |
| FinBERT Sentiment Score   | {_fmt_num(p1.get("sentiment", 0), ".4f")} ({sent_label}) |
| **Phase 1 Blended Signal**| **{_fmt_num(p1.get("phase1_signal", 0), ".4f")} ({signal_label})** |

---

## [2] RL Portfolio Allocation (Phase 2 — PPO Agent)

### Strategy Performance
| Metric              | PPO Agent                                         |
|---------------------|---------------------------------------------------|
| Total Return        | {p2.get('total_return_pct', 'N/A')}%              |
| Equal-Weight Return | {p2.get('equal_weight_ret_pct', 'N/A')}%          |
| Alpha vs EW         | {p2.get('alpha_vs_equal_weight', 'N/A')} (pp)     |
| Sharpe Ratio        | {p2.get('sharpe_ratio', 'N/A')}                   |
| Max Drawdown        | {p2.get('max_drawdown_pct', 'N/A')}%              |
| Calmar Ratio        | {p2.get('calmar_ratio', 'N/A')}                   |
| Final Portfolio     | ${p2.get('final_value_usd', 0):,.2f}              |

### Current Target Allocation
{alloc_table}

---

## [3] Live News & Market Sentiment (Phase 3.1 — RAG Vector Store)

{rag_section}

---

## [4] Advisory Task

You are **GenWealth AI**, an institutional-grade AI investment advisor.

Using **only** the quantitative data and news context provided above, generate a 
concise, evidence-based investment advisory response for **{ticker}** covering:

1. **Signal Interpretation**: What does the Phase 1 signal ({signal_label}, {_fmt_num(signal_val, ".4f")}) 
   combined with the FinBERT sentiment ({sent_label}) suggest about near-term price action?

2. **Portfolio Recommendation**: Given the PPO agent's allocation framework, should the 
   position in {ticker} be increased, maintained, or reduced? Justify with specific metrics.

3. **Risk Flags**: Identify any red flags from the volatility ratio ({_fmt_num(p1.get("vol_ratio", 0), ".3f")}), 
   drawdown ({p2.get("max_drawdown_pct", "N/A")}%), or news sentiment that warrant caution.

4. **Actionable Summary**: Provide a 2–3 sentence executive summary with a clear 
   directional stance: BUY / HOLD / SELL / REDUCE.

**Constraints**: Do not fabricate data. Base every claim on the figures above.
Respond in professional financial English. Be concise and direct.
"""
        return prompt


# ===========================================================================
# 5. Self-Test Execution Block
# ===========================================================================

if __name__ == "__main__":
    """
    Self-test for context_builder.py.

    Run from project root:
        python -m src.advisor.context_builder

    Tests:
        1. Phase 1 snapshot retrieval for NVDA
        2. Phase 2 portfolio allocation
        3. RAG context query (seeds MongoDB if needed)
        4. Full ContextAggregator.build_ticker_context()
        5. build_llm_prompt_context() — print the final Markdown prompt
    """
    import json

    TICKER = "NVDA"

    print("\n" + "=" * 65)
    print("  GenWealth — Phase 3 | context_builder.py Self-Test")
    print("=" * 65 + "\n")

    # --- Test 1: Phase 1 Snapshot --------------------------------------------
    print(f"[TEST 1] Phase 1 Snapshot for '{TICKER}'...")
    try:
        snap = get_phase1_snapshot(TICKER)
        print(f"  PASS  as_of_date    : {snap['as_of_date']}")
        print(f"        close_price   : ${snap['close_price']:,.4f}")
        print(f"        sentiment     : {snap['sentiment']:.4f}")
        print(f"        phase1_signal : {snap['phase1_signal']:.4f}\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")

    # --- Test 2: Phase 2 Portfolio -------------------------------------------
    print(f"[TEST 2] Phase 2 Portfolio Allocation...")
    try:
        alloc = get_phase2_allocation(TICKER)
        print(f"  PASS  total_return    : {alloc['total_return_pct']}%")
        print(f"        equal_weight    : {alloc['equal_weight_ret_pct']}%")
        print(f"        alpha_vs_ew     : {alloc['alpha_vs_equal_weight']}")
        print(f"        sharpe_ratio    : {alloc['sharpe_ratio']}")
        print(f"        max_drawdown    : {alloc['max_drawdown_pct']}%")
        print(f"        final_value     : ${alloc['final_value_usd']:,.2f}")
        print(f"        allocation      : {alloc['allocation']}\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")

    # --- Test 3: RAG Context -------------------------------------------------
    print(f"[TEST 3] RAG Context for '{TICKER}'...")
    try:
        rag = get_rag_context(TICKER, top_k=2)
        if rag:
            print(f"  PASS  Retrieved {len(rag)} document(s):")
            for i, doc in enumerate(rag, 1):
                print(f"    [{i}] score={doc['similarity_score']:.4f} | "
                      f"{doc['text_content'][:80]}...")
        else:
            print("  WARN  No RAG documents returned.")
        print()
    except Exception as e:
        print(f"  FAIL  {e}\n")

    # --- Test 4: Full Context Build ------------------------------------------
    print(f"[TEST 4] ContextAggregator.build_ticker_context('{TICKER}')...")
    try:
        agg = ContextAggregator()
        ctx = agg.build_ticker_context(TICKER)
        print(f"  PASS  Top-level keys: {list(ctx.keys())}")
        print(f"        Phase 3 RAG docs: {len(ctx['phase3_rag'])}\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")

    # --- Test 5: LLM Prompt --------------------------------------------------
    print(f"[TEST 5] build_llm_prompt_context('{TICKER}')...")
    print("-" * 65)
    try:
        prompt = agg.build_llm_prompt_context(TICKER)
        print(prompt)
    except Exception as e:
        print(f"  FAIL  {e}")

    print("=" * 65)
    print("  Context builder self-test complete.")
    print("=" * 65 + "\n")
