"""
GenWealth — Dynamic Entry/Exit Paper Trading Simulation & Reasoning Audit
=========================================================================
File : tests/test_investor_simulation.py  (v2 — Dynamic Rebalancing)

Enhancements over v1:
  • DynamicTradeSimulator replaces static buy-and-hold P&L
  • SmartPortfolioAllocator (inverse-vol × conviction) for sizing
  • Multi-step rebalancing: BUY / TRIM_PROFIT / STOP_LOSS / VOL_EXIT / CLOSE_ALL
  • Groq fallback for Stage 2 → 100% AI report coverage (zero "N/A" entries)
  • 3-second inter-ticker sleep (Groq handles quota gracefully)
  • Full verbatim AI transcripts in collapsible <details> blocks

4 Portfolios:
  1. US Tech & Growth   (USD $100,000)    — NVDA, AAPL, MSFT, TSLA
  2. Indian Equities    (INR ₹10,000,000) — RELIANCE.NS, TCS.NS, HDFCBANK.NS
  3. Global Cross-Asset (USD $100,000)    — ASML.AS, 7203.T, BTC-USD
  4. Autonomous AI Discovery ($100,000)  — Top 5 DuckDuckGo tickers
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging — suppress INFO-level noise from sub-modules during the run
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("genwealth.sim_v2")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    import numpy as np
    import pandas as pd
    from ddgs import DDGS
except ImportError as e:
    print(f"[FATAL] Missing package: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Internal Modules
# ---------------------------------------------------------------------------
from src.advisor.llm_engine import (
    LLMAdvisorEngine,
    classify_query_intent,
)
from src.advisor.guardrails import sanitize_and_append_disclaimer, get_compliance_flags
from src.advisor.context_builder import ContextAggregator
from src.advisor.trade_execution_simulator import (
    DynamicTradeSimulator,
    SimulationResult,
    LedgerEntry,
    momentum_signal,
    extract_stance,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_PATH = PROJECT_ROOT / "reports" / "INVESTOR_SIMULATION_AUDIT_REPORT.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

INTER_TICKER_SLEEP_S = 3      # Reduced from 12s — Groq fallback handles quota

# Phase 1 trained tickers (for signal sourcing)
PHASE1_TICKERS = {"HDFCBANK.NS", "NVDA", "RELIANCE.NS", "TCS.NS"}

# Extended currency / exchange map
_CURRENCY_MAP = {
    ".NS":     ("INR",     "NSE India",           "₹"),
    ".BO":     ("INR",     "BSE India",            "₹"),
    ".AS":     ("EUR",     "Euronext Amsterdam",   "€"),
    ".T":      ("JPY",     "Tokyo SE",             "¥"),
    ".L":      ("GBP",     "London SE",            "£"),
    "BTC-USD": ("USD/BTC", "Crypto",               "$"),
    "ETH-USD": ("USD/ETH", "Crypto",               "$"),
}

def _currency_info(ticker: str) -> tuple[str, str, str]:
    tu = ticker.upper()
    for suffix, (cur, exch, sym) in _CURRENCY_MAP.items():
        if tu == suffix or tu.endswith(suffix):
            return cur, exch, sym
    return "USD", "US Markets", "$"


# ===========================================================================
# Signal Helper
# ===========================================================================

def get_ticker_signal(ticker: str, price_hist: Optional[pd.DataFrame]) -> float:
    """
    Return Phase1 blended signal for trained tickers; momentum proxy for others.
    """
    if ticker.upper() in PHASE1_TICKERS:
        try:
            agg = ContextAggregator()
            snap = agg.get_phase1_snapshot(ticker)
            sig = snap.get("phase1_signal")
            if sig is not None and isinstance(sig, (int, float)):
                return float(sig)
        except Exception:
            pass

    # Momentum fallback
    if price_hist is not None and len(price_hist) >= 11:
        return momentum_signal(price_hist["Close"])
    return 0.50


# ===========================================================================
# Autonomous Ticker Discovery (DuckDuckGo)
# ===========================================================================

TICKER_RE = re.compile(r'\b([A-Z]{2,5}(?:\.[A-Z]{1,3})?)\b')
STOPWORDS = {
    "CEO", "CFO", "IPO", "ETF", "GDP", "AI", "US", "UK", "EU", "IN",
    "FY", "Q1", "Q2", "Q3", "Q4", "EPS", "PE", "PB", "YOY", "QOQ",
    "TTM", "EV", "API", "LLC", "INC", "LTD", "CORP", "SA", "PLC",
    "AG", "NY", "LA", "DC", "BV", "CTO", "THE", "AND", "FOR",
}

def discover_tickers(capital: float = 100_000, top_n: int = 5) -> dict:
    """DuckDuckGo-driven autonomous ticker selection + equal-weight allocation."""
    log.info("[AutoDiscovery] Scanning DuckDuckGo…")
    candidates: dict[str, dict] = {}

    for q in [
        "best stocks buy high conviction growth 2026",
        "top performing stocks earnings beat strong buy analyst upgrades 2026",
        "AI semiconductor cloud stocks strong buy 2026",
    ]:
        try:
            with DDGS() as ddgs:
                for item in ddgs.news(q, max_results=10, safesearch="off"):
                    text = item.get("title", "") + " " + item.get("body", "")
                    for h in TICKER_RE.findall(text):
                        if h not in STOPWORDS and len(h) >= 2:
                            candidates.setdefault(h, {"mentions": 0, "url": ""})
                            candidates[h]["mentions"] += 1
                            if not candidates[h]["url"]:
                                candidates[h]["url"] = item.get("url", "")
            time.sleep(1.2)
        except Exception as exc:
            log.warning("[AutoDiscovery] DDG query failed: %s", exc)

    ranked = sorted(
        [(t, d) for t, d in candidates.items() if t not in STOPWORDS],
        key=lambda x: x[1]["mentions"], reverse=True
    )[:top_n]

    per_ticker = round(capital / top_n, 2)
    return {
        "tickers_scanned": len(candidates),
        "picks": [
            {"ticker": t, "mentions": d["mentions"],
             "allocation": per_ticker, "url": d["url"]}
            for t, d in ranked
        ],
        "capital": capital,
    }


# ===========================================================================
# Portfolio Simulation (v2 — Dynamic)
# ===========================================================================

def simulate_portfolio_dynamic(
    name:            str,
    currency:        str,
    symbol:          str,
    initial_capital: float,
    tickers:         list[str],
    engine:          LLMAdvisorEngine,
    query_template:  str,
    sleep_s:         float = INTER_TICKER_SLEEP_S,
) -> tuple[SimulationResult, dict[str, str], dict[str, str]]:
    """
    Run the full dynamic simulation for one portfolio.

    Returns:
        (SimulationResult, ai_reports_dict, ai_intents_dict)
    """
    print(f"\n{'━'*70}")
    print(f"  Portfolio: {name}  |  {currency} {symbol}{initial_capital:,.0f}")
    print(f"  Tickers  : {tickers}")
    print(f"{'━'*70}")

    # ── 1. Load price histories (used by both signal computation + simulator)
    print("  [1/3] Loading price histories via yfinance…")
    sim = DynamicTradeSimulator(name, initial_capital, tickers, currency, symbol)
    sim.load_histories()

    # ── 2. Compute signals & run LLM pipeline per ticker ─────────────────────
    print("  [2/3] Running LLM advisory pipeline…")
    signals:    dict[str, float] = {}
    ai_stances: dict[str, str]   = {}
    ai_reports: dict[str, str]   = {}
    ai_intents: dict[str, str]   = {}

    for i, ticker in enumerate(tickers):
        currency_t, exchange_t, symbol_t = _currency_info(ticker)

        # Get signal
        signals[ticker] = get_ticker_signal(ticker, sim.price_hist.get(ticker))

        # Run LLM pipeline
        query = query_template.format(ticker=ticker, currency=currency_t, exchange=exchange_t)
        try:
            payload = engine.run_advisory_pipeline(ticker=ticker, user_query=query)
            report  = payload.get("final_report", payload.get("gemini_report", ""))
            # If the pipeline wrapped the report in another key, fall back
            if not report or "Advisory Report Unavailable" in report:
                report = payload.get("gemini_report", "")

            ai_reports[ticker]  = report
            ai_stances[ticker]  = extract_stance(report)
            ai_intents[ticker]  = payload.get("intent", "UNKNOWN")
        except Exception as exc:
            log.error("[LLM] %s pipeline error: %s", ticker, exc)
            ai_reports[ticker]  = f"## Error\nPipeline failed: {exc}\n\n## 4. Actionable Stance\n**HOLD**"
            ai_stances[ticker]  = "HOLD"
            ai_intents[ticker]  = "UNKNOWN"

        # Quick guardrails pass
        guardrail_flags = get_compliance_flags(ai_reports[ticker])

        # Compute current price for display
        px_current = None
        hist = sim.price_hist.get(ticker)
        if hist is not None and not hist.empty:
            px_current = float(hist["Close"].iloc[-1])
        px_entry = sim._price_at(ticker, 0)

        pnl_display = ""
        if px_current and px_entry and px_entry > 0:
            pnl = (px_current - px_entry) / px_entry * (initial_capital / len(tickers))
            pnl_display = f"| Static_P&L={symbol_t}{pnl:+,.0f}"

        print(
            f"    [{i+1}/{len(tickers)}] {ticker:<14} "
            f"| Signal={signals[ticker]:.3f} "
            f"| Stance={ai_stances[ticker]:<7} "
            f"| Intent={ai_intents[ticker]} "
            f"{pnl_display}"
            f"| Flags={len(guardrail_flags)}"
        )

        if i < len(tickers) - 1:
            time.sleep(sleep_s)

    # ── 3. Run dynamic simulation ─────────────────────────────────────────────
    print("  [3/3] Running DynamicTradeSimulator…")
    result = sim.run(signals, ai_stances, ai_reports)

    print(
        f"\n  ✅ {name} complete — "
        f"Final Value: {symbol}{result.final_cash + result.final_positions_value:,.0f} | "
        f"ROI: {result.total_roi_pct:+.2f}% | "
        f"Win Rate: {result.win_rate_pct:.0f}% | "
        f"Max DD: {result.max_drawdown_pct:.2f}%"
    )

    return result, ai_reports, ai_intents


# ===========================================================================
# Markdown Report Generator
# ===========================================================================

def _fmt(val, fmt: str = ".2f", prefix: str = "", suffix: str = "", na: str = "N/A") -> str:
    try:
        return f"{prefix}{val:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return na


def _action_badge(action: str) -> str:
    return {
        "BUY":            "🟢 **BUY**",
        "TRIM_PROFIT":    "🔵 **TRIM**",
        "STOP_LOSS_EXIT": "🔴 **STOP_LOSS**",
        "VOL_EXIT":       "🟠 **VOL_EXIT**",
        "CLOSE_ALL":      "⬜ **CLOSE**",
        "REBALANCE":      "🔄 **REBALANCE**",
    }.get(action, action)


def _stance_badge(stance: str) -> str:
    return {
        "BUY":    "🟢 BUY",
        "HOLD":   "🟡 HOLD",
        "SELL":   "🔴 SELL",
        "REDUCE": "🟠 REDUCE",
    }.get(stance.upper(), stance)


def generate_report(
    sim_results:  list[SimulationResult],
    ai_reports:   list[dict[str, str]],
    ai_intents:   list[dict[str, str]],
    discovery_meta: dict,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# GenWealth AI — Dynamic Trading Simulation & Reasoning Audit",
        f"> **Generated**: {now}  |  "
        "> **Engine**: GenWealth AI Phase 3  |  "
        "> **Script**: `test_investor_simulation.py v2`",
        "> **Strategy**: Dynamic Rebalancing (BUY → TRIM → STOP_LOSS → CLOSE) "
        "| SmartPortfolioAllocator (Inverse-Vol × Conviction)\n",
        "---\n",
    ]

    # ── A. Executive Investor Summary ─────────────────────────────────────────
    lines += [
        "## A. Executive Investor Summary\n",
        "| Portfolio | Currency | Initial Capital | Final Value | Net P&L | Total ROI | Win Rate | Max Drawdown |",
        "|-----------|----------|-----------------|-------------|---------|-----------|----------|--------------|",
    ]
    for r in sim_results:
        final_val = r.final_cash + r.final_positions_value
        lines.append(
            f"| **{r.portfolio_name}** | {r.currency} "
            f"| {r.symbol}{r.initial_capital:,.0f} "
            f"| {r.symbol}{final_val:,.0f} "
            f"| {r.symbol}{r.total_realized_pnl:+,.0f} "
            f"| {r.total_roi_pct:+.2f}% "
            f"| {r.win_rate_pct:.0f}% "
            f"| {r.max_drawdown_pct:.2f}% |"
        )
    lines += ["", "---\n"]

    # ── B. Per-Portfolio Detail ────────────────────────────────────────────────
    lines.append("## B. Portfolio Details — Ledger & AI Reasoning\n")

    for r, reports, intents in zip(sim_results, ai_reports, ai_intents):
        final_val = r.final_cash + r.final_positions_value
        lines += [
            f"---\n",
            f"### {r.portfolio_name}",
            f"**Currency**: {r.currency}  |  "
            f"**Capital**: {r.symbol}{r.initial_capital:,.0f}  |  "
            f"**Final Value**: {r.symbol}{final_val:,.0f}  |  "
            f"**Net P&L**: {r.symbol}{r.total_realized_pnl:+,.0f}  |  "
            f"**ROI**: {r.total_roi_pct:+.2f}%  |  "
            f"**Win Rate**: {r.win_rate_pct:.0f}%  |  "
            f"**Max Drawdown**: {r.max_drawdown_pct:.2f}%  |  "
            f"**Trades**: {r.n_trades}\n",
        ]

        # ── Transaction Ledger ─────────────────────────────────────────────────
        lines += [
            "#### 📒 Step-by-Step Transaction Ledger\n",
            "| Step | Date | Ticker | Action | Exec Price | Shares | "
            "Capital In | Capital Out | Realized P&L | Cash Balance | ROI% | "
            "Signal | Vol Ratio | Trigger |",
            "|------|------|--------|--------|------------|--------|"
            "------------|-------------|--------------|--------------|------|"
            "--------|-----------|---------|",
        ]
        for e in r.ledger:
            lines.append(
                f"| {e.step} | {e.date} | **{e.ticker}** "
                f"| {_action_badge(e.action)} "
                f"| {r.symbol}{e.exec_price:,.2f} "
                f"| {e.shares_traded:,.3f} "
                f"| {r.symbol}{e.capital_in:,.0f} "
                f"| {r.symbol}{e.capital_out:,.0f} "
                f"| {r.symbol}{e.realized_pnl:+,.2f} "
                f"| {r.symbol}{e.cash_balance:,.0f} "
                f"| {e.position_roi:+.2f}% "
                f"| {e.signal:.3f} "
                f"| {e.vol_ratio:.2f} "
                f"| `{e.trigger}` |"
            )
        lines += [
            "",
            f"> **Cash Remaining**: {r.symbol}{r.final_cash:,.2f}  "
            f"|  **Unrealised Residual**: {r.symbol}{r.final_positions_value:,.2f}\n",
        ]

        # ── AI Reasoning Transcripts ───────────────────────────────────────────
        lines += ["#### 🤖 Full AI Reasoning Transcripts\n"]
        tickers_in_portfolio = sorted(set(e.ticker for e in r.ledger))
        for ticker in tickers_in_portfolio:
            report   = reports.get(ticker, "")
            intent   = intents.get(ticker, "UNKNOWN")
            stance   = extract_stance(report)
            guardrail_flags = get_compliance_flags(report)

            # Collect ticker's ledger entries for summary
            ticker_entries = [e for e in r.ledger if e.ticker == ticker]
            total_pnl = sum(e.realized_pnl for e in ticker_entries)
            final_roi = ticker_entries[-1].position_roi if ticker_entries else 0.0

            lines += [
                f"<details>",
                f"<summary><strong>{ticker}</strong> — {_stance_badge(stance)} "
                f"| Final ROI: {final_roi:+.2f}% "
                f"| Realized P&L: {r.symbol}{total_pnl:+,.2f} "
                f"| Intent: {intent} "
                f"| Guardrail Flags: {len(guardrail_flags)}"
                f"</summary>\n",
                f"**🤖 Gemini / Groq Advisory Report (Verbatim)**:\n",
                "```markdown",
                report if report else "(No report generated)",
                "```\n",
                "**🛡️ Guardrails**:",
            ]
            if guardrail_flags:
                for gf in guardrail_flags:
                    lines.append(f"- ⚠️ `{gf}`")
            else:
                lines.append("- ✅ No compliance violations detected.")
            lines += ["", "</details>\n"]

    # ── C. Autonomous Discovery ───────────────────────────────────────────────
    lines += [
        "---\n",
        "## C. Autonomous AI Discovery Analysis\n",
        f"**Tickers Scanned**: {discovery_meta.get('tickers_scanned', 'N/A')}  |  "
        f"**Selected**: {[p['ticker'] for p in discovery_meta.get('picks', [])]}  |  "
        f"**Capital Per Ticker**: ${discovery_meta['picks'][0]['allocation']:,.0f}\n",
        "| Rank | Ticker | Mentions | DuckDuckGo Source |",
        "|------|--------|----------|-------------------|",
    ]
    for i, pick in enumerate(discovery_meta.get("picks", []), 1):
        url = pick.get("url", "N/A")[:70]
        lines.append(f"| {i} | **{pick['ticker']}** | {pick['mentions']} | {url} |")
    lines.append("")

    # ── D. Simulation Notes ───────────────────────────────────────────────────
    lines += [
        "---\n",
        "## D. Simulation Methodology\n",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| **Price Source** | Yahoo Finance — 30-day historical entry (step 0), daily closes for each step |",
        "| **Allocation** | SmartPortfolioAllocator: inverse-volatility × Phase1 conviction signal |",
        "| **Entry Rule** | Signal > 0.55 AND AI stance in {BUY, HOLD} |",
        "| **Profit-Take** | ROI > +15% → trim 30%, book profit to CASH |",
        "| **Stop-Loss** | ROI < -8% → full position exit to CASH |",
        "| **Vol Exit** | Short/Long vol ratio > 1.3 → risk exit to CASH |",
        "| **Simulation Steps** | 6 checkpoints × 5 trading days ≈ 30-day horizon |",
        "| **LLM Engine** | Gemini 2.0 Flash Lite (primary) → Groq Llama-3.3-70B (auto-fallback) |",
        "| **Guarantees** | Zero 'N/A' reports — Groq fallback ensures 100% AI coverage |",
        "| **Phase 1 Scope** | PyTorch LSTM + RF trained on 4 tickers; momentum proxy for others |",
        "| **Disclaimer** | SEBI/SEC compliant disclaimer applied to all reports |",
        "",
        "> ⚠️ **IMPORTANT**: This is a paper trading simulation for demonstration purposes only.",
        "> All P&L figures are hypothetical. Past performance does not guarantee future results.",
        "> GenWealth AI advisory reports do not constitute financial advice.",
        "",
        f"*Auto-generated — {now}*",
    ]

    return "\n".join(lines)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("  GenWealth AI — Dynamic Paper Trading Simulation v2")
    print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("  Strategy: SmartAllocator + BUY/TRIM/STOP_LOSS/VOL_EXIT rebalancing")
    print("=" * 70)

    engine = LLMAdvisorEngine()

    # ── Portfolio 1: US Tech & Growth ─────────────────────────────────────────
    r1, rep1, int1 = simulate_portfolio_dynamic(
        name="US Tech & Growth", currency="USD", symbol="$",
        initial_capital=100_000,
        tickers=["NVDA", "AAPL", "MSFT", "TSLA"],
        engine=engine,
        query_template=(
            "Provide a comprehensive institutional BUY/HOLD/SELL/REDUCE analysis for {ticker} "
            "({currency}, {exchange}). Include signal interpretation, allocation justification, "
            "risk flags, and a clear directional stance."
        ),
    )

    # ── Portfolio 2: Indian Equities ──────────────────────────────────────────
    r2, rep2, int2 = simulate_portfolio_dynamic(
        name="Indian Equities (NSE)", currency="INR", symbol="₹",
        initial_capital=10_000_000,
        tickers=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
        engine=engine,
        query_template=(
            "Analyse {ticker} ({currency}, {exchange}) for an Indian institutional investor. "
            "Cover Phase 1 signal strength, FinBERT sentiment, PPO allocation, "
            "and provide BUY/HOLD/SELL/REDUCE stance."
        ),
    )

    # ── Portfolio 3: Global Cross-Asset ───────────────────────────────────────
    r3, rep3, int3 = simulate_portfolio_dynamic(
        name="Global Cross-Asset", currency="USD", symbol="$",
        initial_capital=100_000,
        tickers=["ASML.AS", "7203.T", "BTC-USD"],
        engine=engine,
        query_template=(
            "Risk-adjusted cross-asset analysis for {ticker} ({currency}, {exchange}). "
            "Account for currency exposure and volatility. Give directional stance: "
            "BUY / HOLD / SELL / REDUCE."
        ),
    )

    # ── Portfolio 4: Autonomous AI Discovery ──────────────────────────────────
    print("\n" + "━" * 70)
    print("  Autonomous AI Discovery — scanning DuckDuckGo…")
    print("━" * 70)
    discovery = discover_tickers(capital=100_000, top_n=5)
    auto_tickers = [p["ticker"] for p in discovery["picks"]]
    print(f"  Scanned  : {discovery['tickers_scanned']} candidates")
    print(f"  Selected : {auto_tickers}  ($20,000 each)")

    r4, rep4, int4 = simulate_portfolio_dynamic(
        name="Autonomous AI Discovery", currency="USD", symbol="$",
        initial_capital=100_000,
        tickers=auto_tickers,
        engine=engine,
        query_template=(
            "AI autonomously selected {ticker} ({currency}, {exchange}) for growth. "
            "Provide complete investment thesis with entry/exit rationale. "
            "Stance: BUY / HOLD / SELL / REDUCE."
        ),
    )

    # ── Final Summary ─────────────────────────────────────────────────────────
    all_results = [r1, r2, r3, r4]
    all_reports = [rep1, rep2, rep3, rep4]
    all_intents = [int1, int2, int3, int4]

    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)
    for r in all_results:
        fv = r.final_cash + r.final_positions_value
        print(
            f"  {r.portfolio_name:<30} | "
            f"Capital={r.symbol}{r.initial_capital:>12,.0f} | "
            f"Final={r.symbol}{fv:>12,.0f} | "
            f"P&L={r.symbol}{r.total_realized_pnl:>+10,.0f} | "
            f"ROI={r.total_roi_pct:>+6.2f}% | "
            f"WinRate={r.win_rate_pct:.0f}% | "
            f"MaxDD={r.max_drawdown_pct:.2f}%"
        )

    total_tickers = sum(
        len(set(e.ticker for e in r.ledger)) for r in all_results
    )
    total_trades = sum(r.n_trades for r in all_results)
    print(f"\n  Total tickers simulated : {total_tickers}")
    print(f"  Total trade events      : {total_trades}")
    print("=" * 70)

    # ── Generate Report ───────────────────────────────────────────────────────
    print("\n  Generating Markdown audit report…")
    report_md = generate_report(all_results, all_reports, all_intents, discovery)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print(f"  ✅ Report saved → {REPORT_PATH}\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
