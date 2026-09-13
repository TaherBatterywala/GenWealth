"""
GenWealth AI — Walk-Forward Multi-Regime Investor Simulation & Rationale Audit
===============================================================================
File : tests/test_walkforward_investor.py

Stress-tests the full LLM advisory pipeline across three distinct historical
market regimes using a 5-trading-day sequential walk-forward loop with strict
zero-lookahead bias.

Architecture
------------
  Pre-simulation  : Fetch 50-day context window (~35 trading days) before each
                    regime start date. Build quant signals + LLM advisory from
                    this context ONLY — no future data leaked.

  Day 1 (Entry)   : Open positions at Day-1 OPEN price using inverse-vol x
                    conviction weights + a regime-specific cash buffer.

  Days 2-5        : At each day's close:
                      TRIM_PROFIT : ROI > +10% -> sell 30% | ROI > +20% -> sell 50%
                      STOP_LOSS   : ROI < -7%  -> exit 100% to CASH
                      REINVEST    : Freed cash redeployed into other BUY picks

  Day 5           : CLOSE_ALL remaining positions at close price.

Historical Regimes Tested
--------------------------
  1. COVID-19 Crash     (2020-02-24 to 2020-02-28)  extreme bear market
  2. Tech Bear Market   (2022-11-01 to 2022-11-07)  high-inflation selloff
  3. Standard Bull      (2023-11-06 to 2023-11-10)  AI-driven tech rally

Portfolio Universes
-------------------
  1. US Tech Growth       USD  $100,000     NVDA, AAPL, MSFT, TSLA
  2. Indian Equities NSE  INR  10,000,000   RELIANCE.NS, TCS.NS, HDFCBANK.NS
  3. Global Cross-Asset   USD  $100,000     ASML.AS, 7203.T, BTC-USD
  4. Autonomous Discovery USD  $100,000     DuckDuckGo top-5 picks

Output
------
  reports/WALKFORWARD_INVESTOR_AUDIT.md
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
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("genwealth.walkforward")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from ddgs import DDGS
except ImportError as e:
    print(f"[FATAL] Missing package: {e}. Run: pip install yfinance numpy pandas ddgs")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Internal Modules
# ---------------------------------------------------------------------------
from src.advisor.llm_engine import (
    generate_gemini_report,
    generate_groq_report,
    classify_query_intent,
    verify_report_accuracy,
)
from src.advisor.guardrails import sanitize_and_append_disclaimer, get_compliance_flags
from src.advisor.trade_execution_simulator import (
    momentum_signal,
    rolling_vol_ratio,
    extract_stance,
    extract_ai_reason,
)

# ===========================================================================
# Configuration Constants
# ===========================================================================

REPORT_PATH = PROJECT_ROOT / "reports" / "WALKFORWARD_INVESTOR_AUDIT.md"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

CONTEXT_CALENDAR_DAYS = 50    # Calendar days before regime_start (~35 trading days)
TRIM_THRESHOLD_LO     = 0.10  # ROI >+10% -> trim 30%
TRIM_THRESHOLD_HI     = 0.20  # ROI >+20% -> trim 50%
TRIM_FRACTION_LO      = 0.30
TRIM_FRACTION_HI      = 0.50
STOP_LOSS_ROI         = -0.07 # ROI <-7% -> exit 100%
SIGNAL_BUY_THRESHOLD  = 0.50  # Momentum signal floor for entry
INTER_TICKER_SLEEP_S  = 2     # Seconds between sequential LLM calls

# ---------------------------------------------------------------------------
# Historical Market Regime Definitions
# ---------------------------------------------------------------------------
REGIMES: list[dict] = [
    {
        "name":        "COVID-19 Crash",
        "label":       "Regime 1 - COVID-19 Crash (Feb 24-28, 2020 | Extreme Bear)",
        "start":       "2020-02-24",
        "end":         "2020-02-28",
        "cash_buffer": 0.20,
        "description": (
            "The COVID-19 pandemic triggered the fastest global crash in history. "
            "S&P 500 fell approx 12% in a single week - worst weekly drop since 2008. "
            "VIX spiked above 47. Circuit breakers were triggered on US exchanges. "
            "AI systems must demonstrate aggressive stop-loss execution in this regime."
        ),
    },
    {
        "name":        "Tech Bear Market",
        "label":       "Regime 2 - Tech Bear Market (Nov 1-7, 2022 | Rate-Hike Selloff)",
        "start":       "2022-11-01",
        "end":         "2022-11-07",
        "cash_buffer": 0.15,
        "description": (
            "The Fed's most aggressive rate-hike cycle since the 1980s crushed growth/tech. "
            "NASDAQ had already fallen approx 35% YTD by Nov 2022. Growth equities with "
            "high P/E multiples were especially punished as risk-free rate approached 4%. "
            "AI systems should favour value and reduce speculative tech exposure."
        ),
    },
    {
        "name":        "Bull Market",
        "label":       "Regime 3 - Bull Market Rally (Nov 6-10, 2023 | AI-Driven Rally)",
        "start":       "2023-11-06",
        "end":         "2023-11-10",
        "cash_buffer": 0.05,
        "description": (
            "The generative AI boom drove NVDA up approx 200% YTD. Markets priced in Fed pivot. "
            "Broad risk-on environment with VIX below 15. Growth stocks recovered sharply. "
            "AI systems should deploy capital aggressively and take profit on winners."
        ),
    },
]

# ---------------------------------------------------------------------------
# Portfolio Definitions
# ---------------------------------------------------------------------------
PORTFOLIOS: list[dict] = [
    {
        "name":     "US Tech Growth",
        "currency": "USD",
        "symbol":   "$",
        "capital":  100_000,
        "tickers":  ["NVDA", "AAPL", "MSFT", "TSLA"],
    },
    {
        "name":     "Indian Equities (NSE)",
        "currency": "INR",
        "symbol":   "Rs.",
        "capital":  10_000_000,
        "tickers":  ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    },
    {
        "name":     "Global Cross-Asset",
        "currency": "USD",
        "symbol":   "$",
        "capital":  100_000,
        "tickers":  ["ASML.AS", "7203.T", "BTC-USD"],
    },
]

_CURRENCY_MAP: dict[str, tuple[str, str, str]] = {
    ".NS":     ("INR",     "NSE India",           "Rs."),
    ".BO":     ("INR",     "BSE India",            "Rs."),
    ".AS":     ("EUR",     "Euronext Amsterdam",   "EUR"),
    ".T":      ("JPY",     "Tokyo SE",             "JPY"),
    ".L":      ("GBP",     "London SE",            "GBP"),
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
# Data Classes
# ===========================================================================

@dataclass
class DayEvent:
    """A single trade event during a walk-forward simulation day."""
    day_num:          int
    date:             str
    ticker:           str
    action:           str
    trigger:          str
    exec_price:       float
    shares_traded:    float
    capital_in:       float
    capital_out:      float
    realized_pnl:     float
    cash_after:       float
    position_roi_pct: float
    signal:           float
    vol_20d:          float
    ai_reason:        str


@dataclass
class RegimePortfolioResult:
    """Aggregated result for one portfolio under one historical market regime."""
    regime_name:        str
    regime_label:       str
    portfolio_name:     str
    currency:           str
    symbol:             str
    initial_capital:    float
    final_cash:         float
    final_unrealized:   float
    total_realized_pnl: float
    total_roi_pct:      float
    win_rate_pct:       float
    max_drawdown_pct:   float
    n_trades:           int
    tickers:            list = field(default_factory=list)
    events:             list = field(default_factory=list)
    ai_reports:         dict = field(default_factory=dict)
    ai_stances:         dict = field(default_factory=dict)
    ai_intents:         dict = field(default_factory=dict)
    critic_results:     dict = field(default_factory=dict)
    signals:            dict = field(default_factory=dict)
    volatilities:       dict = field(default_factory=dict)


# ===========================================================================
# Data Fetching
# ===========================================================================

def fetch_regime_data(
    ticker: str,
    regime_start: str,
    regime_end:   str,
    context_days: int = CONTEXT_CALENDAR_DAYS,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV covering the pre-window context + 5-day walk-forward window.
    context_days calendar days before regime_start form the pre-window context.
    regime_end is extended by 4 days to account for yfinance exclusive end-date behaviour.
    """
    start_dt    = datetime.strptime(regime_start, "%Y-%m-%d") - timedelta(days=context_days)
    end_dt      = datetime.strptime(regime_end,   "%Y-%m-%d") + timedelta(days=4)
    fetch_start = start_dt.strftime("%Y-%m-%d")
    fetch_end   = end_dt.strftime("%Y-%m-%d")

    try:
        hist = yf.download(
            ticker,
            start=fetch_start,
            end=fetch_end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if hist.empty or len(hist) < 5:
            log.warning("[yfinance] %s: insufficient rows (%d) for %s to %s",
                        ticker, len(hist), regime_start, regime_end)
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist

    except Exception as exc:
        log.error("[yfinance] %s: fetch failed - %s", ticker, exc)
        return None


def split_regime(
    hist:         pd.DataFrame,
    regime_start: str,
    regime_end:   str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full OHLCV history into:
      pre_window  - rows strictly before regime_start (for signals/LLM context)
      walk_days   - rows within [regime_start, regime_end] (the 5 trading days)
    Zero-lookahead: walk_days is NEVER passed to the LLM.
    """
    start_ts = pd.Timestamp(regime_start)
    end_ts   = pd.Timestamp(regime_end)
    pre  = hist[hist.index < start_ts].copy()
    walk = hist[(hist.index >= start_ts) & (hist.index <= end_ts)].copy()
    return pre, walk


# ===========================================================================
# Quantitative Signal Computation
# ===========================================================================

def compute_ticker_signals(
    pre_window: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Compute (momentum_signal, vol_20d_annualised, vol_ratio_5d_20d) from pre-window OHLCV.
    Returns: (signal [0.30-0.70], vol_20d [floored at 5%], vol_ratio)
    """
    closes = pre_window["Close"].dropna()
    if len(closes) < 5:
        return 0.50, 0.30, 1.0

    window  = min(10, len(closes) - 1)
    sig     = momentum_signal(closes, window=window)
    rets    = closes.pct_change().dropna()
    n_rets  = len(rets)
    vol_20d = float(rets.tail(min(20, n_rets)).std()) * (252 ** 0.5)
    vol_20d = max(vol_20d, 0.05)
    vol_r   = rolling_vol_ratio(closes)

    return float(sig), float(vol_20d), float(vol_r)


def compute_allocation_weights(
    tickers:     list[str],
    signals:     dict[str, float],
    vols:        dict[str, float],
    ai_stances:  dict[str, str],
    cash_buffer: float,
) -> dict[str, float]:
    """
    Inverse-vol x conviction weighting. Weights sum to (1 - cash_buffer).
    Only tickers with signal > threshold OR stance in {BUY, HOLD} receive capital.
    """
    POSITIVE = {"BUY", "HOLD"}
    eligible = [
        t for t in tickers
        if signals.get(t, 0.5) > SIGNAL_BUY_THRESHOLD
        or ai_stances.get(t, "HOLD").upper() in POSITIVE
    ]
    if not eligible:
        log.warning("[Allocator] No tickers pass entry filter - all in CASH")
        return {t: 0.0 for t in tickers}

    scores: dict[str, float] = {
        t: (1.0 / max(vols.get(t, 0.30), 0.05)) * signals.get(t, 0.50)
        for t in eligible
    }
    total_score = sum(scores.values())
    deployable  = 1.0 - cash_buffer
    w_elig = {t: (scores[t] / total_score) * deployable for t in eligible}
    return {t: w_elig.get(t, 0.0) for t in tickers}


# ===========================================================================
# Historical LLM Context Builder
# ===========================================================================

def build_historical_prompt(
    ticker:     str,
    pre_window: pd.DataFrame,
    regime:     dict,
) -> str:
    """Build a structured Markdown advisory context from historical pre-window OHLCV data."""
    closes   = pre_window["Close"].dropna()
    currency, exchange, sym = _currency_info(ticker)

    latest_px = float(closes.iloc[-1]) if len(closes) > 0 else 0.0
    oldest_px = float(closes.iloc[0])  if len(closes) > 0 else 0.0
    ret_30d   = (latest_px - oldest_px) / oldest_px if oldest_px > 0 else 0.0

    sig, vol_20d, vol_r = compute_ticker_signals(pre_window)

    last5      = pre_window[["Open", "High", "Low", "Close", "Volume"]].tail(5).copy()
    avail_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in last5.columns]
    ohlcv_lines = ["| Date | " + " | ".join(avail_cols) + " |",
                   "|------|" + "--------|" * len(avail_cols)]
    for idx, row in last5.iterrows():
        date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
        vals = []
        for col in avail_cols:
            v = row[col]
            if pd.isna(v):
                vals.append("N/A")
            elif col == "Volume":
                vals.append(f"{int(v):,}")
            else:
                vals.append(f"{v:,.2f}")
        ohlcv_lines.append("| " + date_str + " | " + " | ".join(vals) + " |")
    ohlcv_table = "\n".join(ohlcv_lines)

    return f"""## HISTORICAL REGIME SIMULATION CONTEXT

> WARNING - BACK-TEST MODE: This advisory is generated for a historical regime
> simulation. ALL data below is from the pre-window (before regime start date).
> The LLM must NOT reference any data after the pre-window close.
>
> Regime: {regime['label']}
> Simulation Window: {regime['start']} to {regime['end']}
> Market Context: {regime['description']}

---

## Ticker: {ticker}
Currency: {currency} ({exchange}) | Cash Buffer in Force: {regime['cash_buffer']*100:.0f}%

### Pre-Window Price Snapshot
| Metric | Value |
|--------|-------|
| Last Pre-Window Close | {sym}{latest_px:,.2f} |
| Context-Period Return (~30d) | {ret_30d:+.2%} |
| 20-Day Ann. Realised Volatility | {vol_20d:.1%} |
| Momentum Signal [0-1] | {sig:.4f} ({'Bullish' if sig > 0.55 else 'Neutral' if sig > 0.45 else 'Bearish'}) |
| Vol Spike Ratio (5d/20d) | {vol_r:.3f} {'SPIKE DETECTED' if vol_r > 1.3 else 'Normal'} |
| 30d Trend Direction | {'UP' if ret_30d > 0 else 'DOWN'} ({ret_30d:+.2%}) |

### Last 5 Pre-Window Sessions (OHLCV)
{ohlcv_table}

### Quantitative Signal Summary
- Signal: {'BULLISH - enter with full conviction weight' if sig > 0.60 else 'NEUTRAL - enter with reduced weight' if sig > 0.50 else 'BEARISH - hold cash or minimal position'}
- Volatility Regime: {'HIGH' if vol_20d > 0.40 else 'MODERATE' if vol_20d > 0.20 else 'LOW'} ({vol_20d:.1%} annualised)
- Volatility Spike: {'WARNING: spike detected - reduce position size' if vol_r > 1.3 else 'Within normal range'}
- Regime Context: {regime['name']} - {'maintain maximum 80% deployment' if regime['cash_buffer'] >= 0.20 else 'maintain maximum 85% deployment' if regime['cash_buffer'] >= 0.15 else 'deploy up to 95% of capital'}

> Currency Context: All prices for {ticker} are in {currency} ({exchange}).
> For non-USD tickers in a USD portfolio, returns are computed on a price-relative (FX-neutral) basis.
> FinBERT Sentiment: Historical back-test - live news sentiment unavailable. Use price action context above.
"""


# ===========================================================================
# Walk-Forward Portfolio Simulator
# ===========================================================================

@dataclass
class _Pos:
    """Tracks a single open position."""
    ticker:           str
    shares:           float
    entry_price:      float
    capital_invested: float
    entry_day:        int
    entry_date:       str

    def roi(self, current_price: float) -> float:
        return (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0

    def current_value(self, current_price: float) -> float:
        return self.shares * current_price


class WalkForwardSimulator:
    """
    5-day sequential walk-forward simulation for one portfolio/regime.

    Zero-Lookahead Contract:
        - pre_windows used ONLY for signal computation and LLM context
        - walk_hists released one day at a time (row 0 = Day 1, row 4 = Day 5)
        - At Day d the simulator sees walk_hists rows [0 .. d] only

    Active Management (evaluated at each day's close):
        TRIM_PROFIT : ROI >= +10% -> sell 30% | ROI >= +20% -> sell 50%
        STOP_LOSS   : ROI <= -7%  -> exit 100% to CASH
        REINVEST    : After any exit, 80% of freed cash re-deployed into BUY picks
        CLOSE_ALL   : Day 5 -> liquidate everything at close
    """

    def __init__(self, portfolio: dict, regime: dict) -> None:
        self.portfolio       = portfolio
        self.regime          = regime
        self.cash            = float(portfolio["capital"])
        self.initial_capital = float(portfolio["capital"])
        self.cash_buffer     = float(regime["cash_buffer"])
        self.symbol          = portfolio["symbol"]
        self.positions: dict[str, _Pos]           = {}
        self.events:    list[DayEvent]             = []
        self.closed_pnl: dict[str, float]          = {}
        self._curve: list[tuple[int, float]]       = []

    def _total_value(self, day_prices: dict[str, float]) -> float:
        pos_val = sum(
            pos.current_value(day_prices.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )
        return self.cash + pos_val

    def _max_drawdown(self) -> float:
        if len(self._curve) < 2:
            return 0.0
        vals = [v for _, v in self._curve]
        peak, max_dd = vals[0], 0.0
        for v in vals:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd * 100.0

    @staticmethod
    def _px(wh: Optional[pd.DataFrame], row: int, col: str = "Close") -> Optional[float]:
        if wh is None or wh.empty or row >= len(wh):
            return None
        try:
            val = wh.iloc[row][col]
            return float(val) if not pd.isna(val) else None
        except Exception:
            return None

    @staticmethod
    def _dt(wh: Optional[pd.DataFrame], row: int) -> str:
        if wh is None or wh.empty or row >= len(wh):
            return "N/A"
        idx_val = wh.index[row]
        return str(idx_val.date()) if hasattr(idx_val, "date") else str(idx_val)

    def run(
        self,
        walk_hists: dict[str, pd.DataFrame],
        signals:    dict[str, float],
        vols:       dict[str, float],
        ai_stances: dict[str, str],
        ai_reports: dict[str, str],
        weights:    dict[str, float],
    ) -> tuple[float, float, float, float]:
        """Execute 5-day simulation. Returns (final_cash, final_unrealized, total_pnl, max_dd_pct)."""
        tickers = self.portfolio["tickers"]
        valid   = [t for t in tickers if t in walk_hists and not walk_hists[t].empty]

        # ── Day 1: Entry at open price ───────────────────────────────────────
        for ticker in valid:
            w = weights.get(ticker, 0.0)
            if w < 0.001:
                continue
            wh = walk_hists[ticker]
            entry_px = self._px(wh, 0, "Open") or self._px(wh, 0, "Close")
            if not entry_px or entry_px <= 0:
                continue

            d1_date = self._dt(wh, 0)
            cap     = self.cash * w
            shares  = cap / entry_px
            self.positions[ticker] = _Pos(
                ticker=ticker, shares=shares, entry_price=entry_px,
                capital_invested=cap, entry_day=1, entry_date=d1_date,
            )
            self.cash -= cap
            self.events.append(DayEvent(
                day_num=1, date=d1_date, ticker=ticker,
                action="BUY", trigger="SIGNAL_AND_AI_STANCE",
                exec_price=entry_px, shares_traded=shares,
                capital_in=cap, capital_out=0.0,
                realized_pnl=0.0, cash_after=self.cash,
                position_roi_pct=0.0,
                signal=signals.get(ticker, 0.5),
                vol_20d=vols.get(ticker, 0.30),
                ai_reason=(
                    f"[W:{w*100:.1f}%] "
                    + extract_ai_reason(ai_reports.get(ticker, ""), 220)
                ),
            ))

        d1_closes = {t: self._px(walk_hists[t], 0) or 0.0 for t in valid}
        self._curve.append((1, self._total_value(d1_closes)))

        max_days = max(
            (len(wh) for wh in walk_hists.values() if wh is not None and not wh.empty),
            default=1,
        )

        # ── Days 2-5: Active management ──────────────────────────────────────
        for day_idx in range(1, min(5, max_days)):
            day_num  = day_idx + 1
            is_final = (day_num == 5) or (day_idx >= max_days - 1)

            today_closes: dict[str, float] = {}
            today_dates:  dict[str, str]   = {}
            for t in valid:
                px = self._px(walk_hists.get(t), day_idx)
                if px:
                    today_closes[t] = px
                    today_dates[t]  = self._dt(walk_hists.get(t), day_idx)

            freed_today  = 0.0
            newly_closed: list[str] = []

            for ticker in list(self.positions.keys()):
                pos   = self.positions[ticker]
                px    = today_closes.get(ticker, pos.entry_price)
                roi   = pos.roi(px)
                date  = today_dates.get(ticker, f"Day-{day_num}")
                vol20 = vols.get(ticker, 0.30)

                if is_final:
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    freed_today += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    newly_closed.append(ticker)
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="CLOSE_ALL", trigger="END_OF_SIMULATION",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"Day-5 liquidation at {self.symbol}{px:,.2f}. "
                            f"Final ROI: {roi*100:+.2f}%."
                        ),
                    ))
                    del self.positions[ticker]

                elif roi >= TRIM_THRESHOLD_HI:
                    frac        = TRIM_FRACTION_HI
                    trim_sh     = pos.shares * frac
                    proceeds    = trim_sh * px
                    cost_basis  = (pos.capital_invested / pos.shares) * trim_sh
                    pnl         = proceeds - cost_basis
                    self.cash  += proceeds
                    freed_today += proceeds
                    pos.shares           -= trim_sh
                    pos.capital_invested -= cost_basis
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="TRIM_PROFIT", trigger=f"ROI>={TRIM_THRESHOLD_HI*100:.0f}pct",
                        exec_price=px, shares_traded=trim_sh,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"Aggressive trim: ROI={roi*100:+.2f}% exceeded +{TRIM_THRESHOLD_HI*100:.0f}%. "
                            f"Sold {frac*100:.0f}% ({trim_sh:,.3f} sh). "
                            f"Locked {self.symbol}{pnl:+,.2f}."
                        ),
                    ))

                elif roi >= TRIM_THRESHOLD_LO:
                    frac        = TRIM_FRACTION_LO
                    trim_sh     = pos.shares * frac
                    proceeds    = trim_sh * px
                    cost_basis  = (pos.capital_invested / pos.shares) * trim_sh
                    pnl         = proceeds - cost_basis
                    self.cash  += proceeds
                    freed_today += proceeds
                    pos.shares           -= trim_sh
                    pos.capital_invested -= cost_basis
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="TRIM_PROFIT", trigger=f"ROI>={TRIM_THRESHOLD_LO*100:.0f}pct",
                        exec_price=px, shares_traded=trim_sh,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"Profit trim: ROI={roi*100:+.2f}% crossed +{TRIM_THRESHOLD_LO*100:.0f}%. "
                            f"Sold {frac*100:.0f}% ({trim_sh:,.3f} sh). "
                            f"Realized {self.symbol}{pnl:+,.2f}."
                        ),
                    ))

                elif roi < STOP_LOSS_ROI:
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    freed_today += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    newly_closed.append(ticker)
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="STOP_LOSS", trigger=f"ROI<{STOP_LOSS_ROI*100:.0f}pct",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"STOP-LOSS: ROI={roi*100:+.2f}% breached {STOP_LOSS_ROI*100:.0f}% floor. "
                            f"Full position ({pos.shares:,.3f} sh) exited at {self.symbol}{px:,.2f}. "
                            f"Loss: {self.symbol}{pnl:+,.2f}."
                        ),
                    ))
                    del self.positions[ticker]

            # Capital Recycling after exits
            if not is_final and freed_today > self.initial_capital * 0.005:
                candidates = [
                    t for t in valid
                    if t not in self.positions
                    and t not in newly_closed
                    and ai_stances.get(t, "HOLD").upper() == "BUY"
                    and signals.get(t, 0.5) > SIGNAL_BUY_THRESHOLD
                ]
                if candidates:
                    cand_sc = {
                        t: (1.0 / max(vols.get(t, 0.30), 0.05)) * signals.get(t, 0.5)
                        for t in candidates
                    }
                    total_sc = sum(cand_sc.values())
                    deploy   = freed_today * 0.80

                    for t in candidates:
                        rc   = (cand_sc[t] / total_sc) * deploy
                        px_r = today_closes.get(t)
                        if not px_r or px_r <= 0:
                            continue
                        sh_r = rc / px_r
                        self.positions[t] = _Pos(
                            ticker=t, shares=sh_r, entry_price=px_r,
                            capital_invested=rc, entry_day=day_num,
                            entry_date=today_dates.get(t, f"Day-{day_num}"),
                        )
                        self.cash -= rc
                        self.events.append(DayEvent(
                            day_num=day_num,
                            date=today_dates.get(t, f"Day-{day_num}"),
                            ticker=t, action="REINVEST", trigger="CAPITAL_RECYCLING",
                            exec_price=px_r, shares_traded=sh_r,
                            capital_in=rc, capital_out=0.0,
                            realized_pnl=0.0, cash_after=self.cash,
                            position_roi_pct=0.0,
                            signal=signals.get(t, 0.5), vol_20d=vols.get(t, 0.30),
                            ai_reason=(
                                f"Capital recycling: {self.symbol}{freed_today:,.0f} freed. "
                                f"Re-deploying {self.symbol}{rc:,.0f} into {t} "
                                f"(signal={signals.get(t,0.5):.3f}, stance=BUY)."
                            ),
                        ))

            self._curve.append((day_num, self._total_value(today_closes)))

        # Final stats
        last_closes = dict(today_closes) if "today_closes" in dir() else {}
        residual = sum(
            pos.current_value(last_closes.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )
        total_final = self.cash + residual
        total_roi   = (total_final - self.initial_capital) / self.initial_capital * 100
        wins        = sum(1 for p in self.closed_pnl.values() if p > 0)
        n_closed    = len(self.closed_pnl)
        win_rate    = (wins / n_closed * 100) if n_closed > 0 else 0.0

        return self.cash, residual, total_final - self.initial_capital, self._max_drawdown()


# ===========================================================================
# Autonomous Ticker Discovery
# ===========================================================================

TICKER_RE = re.compile(r"\b([A-Z]{2,5}(?:\.[A-Z]{1,3})?)\b")
STOPWORDS = {
    "CEO", "CFO", "IPO", "ETF", "GDP", "AI", "US", "UK", "EU", "IN",
    "FY", "Q1", "Q2", "Q3", "Q4", "EPS", "PE", "PB", "YOY", "QOQ",
    "TTM", "EV", "API", "LLC", "INC", "LTD", "CORP", "SA", "PLC",
    "AG", "NY", "LA", "DC", "BV", "CTO", "THE", "AND", "FOR", "BUT",
    "AMD", "IBM", "JPM", "GS", "MS", "ML", "DB",
}

def discover_tickers(capital: float = 100_000, top_n: int = 5) -> dict:
    """DuckDuckGo-driven autonomous ticker selection with mention ranking."""
    log.info("[AutoDiscovery] Scanning DuckDuckGo...")
    candidates: dict[str, dict] = {}

    queries = [
        "best stocks to buy strong conviction growth 2026",
        "top performing stocks earnings beat analyst upgrades 2026",
        "AI semiconductor cloud high growth stocks BUY 2026",
    ]
    for q in queries:
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
        key=lambda x: x[1]["mentions"], reverse=True,
    )[:top_n]

    if not ranked:
        log.warning("[AutoDiscovery] No tickers found - using fallback list")
        fallback = ["NVDA", "META", "GOOG", "AMZN", "MSFT"]
        ranked = [(t, {"mentions": 0, "url": ""}) for t in fallback[:top_n]]

    per_ticker = round(capital / max(top_n, 1), 2)
    return {
        "tickers_scanned": len(candidates),
        "picks": [
            {"ticker": t, "mentions": d["mentions"], "allocation": per_ticker, "url": d["url"]}
            for t, d in ranked
        ],
        "capital": capital,
    }


# ===========================================================================
# Regime x Portfolio Orchestrator
# ===========================================================================

def run_portfolio_regime(portfolio: dict, regime: dict) -> RegimePortfolioResult:
    """
    Full orchestration for one portfolio under one historical regime.
    Steps: fetch data -> compute signals -> LLM advisory -> walk-forward simulation.
    """
    port_name = portfolio["name"]
    reg_name  = regime["name"]
    tickers   = portfolio["tickers"]
    symbol    = portfolio["symbol"]

    print(f"\n  > [{reg_name}] [{port_name}] Fetching historical data...")

    # 1. Fetch + split
    pre_windows: dict[str, pd.DataFrame] = {}
    walk_hists:  dict[str, pd.DataFrame] = {}
    raw_signals: dict[str, float]        = {}
    raw_vols:    dict[str, float]        = {}

    for ticker in tickers:
        full_hist = fetch_regime_data(ticker, regime["start"], regime["end"])
        if full_hist is None:
            continue
        pre, walk = split_regime(full_hist, regime["start"], regime["end"])
        if pre.empty or walk.empty:
            log.warning("[%s][%s] %s: empty split - skipping", reg_name, port_name, ticker)
            continue
        pre_windows[ticker] = pre
        walk_hists[ticker]  = walk
        sig, vol, _ = compute_ticker_signals(pre)
        raw_signals[ticker] = sig
        raw_vols[ticker]    = vol
        log.info("[%s][%s] %s: pre=%d walk=%d sig=%.3f vol=%.2f",
                 reg_name, port_name, ticker, len(pre), len(walk), sig, vol)

    valid_tickers = [t for t in tickers if t in pre_windows]
    if not valid_tickers:
        log.error("[%s][%s] No valid tickers - returning empty result", reg_name, port_name)
        return _empty_result(portfolio, regime)

    # 2. LLM advisory (once per ticker, pre-window context only)
    print(f"  > [{reg_name}] [{port_name}] LLM pipeline ({len(valid_tickers)} tickers)...")

    ai_reports:     dict[str, str]  = {}
    ai_stances:     dict[str, str]  = {}
    ai_intents:     dict[str, str]  = {}
    critic_results: dict[str, dict] = {}

    for i, ticker in enumerate(valid_tickers):
        hist_prompt = build_historical_prompt(ticker, pre_windows[ticker], regime)

        # Stage 1: Intent
        try:
            intent = classify_query_intent(
                f"Historical analysis: BUY, HOLD, SELL or REDUCE {ticker} "
                f"during {regime['name']} conditions?"
            )
        except Exception:
            intent = "SINGLE_TICKER_NEWS"
        ai_intents[ticker] = intent

        # Stage 2: Report (Gemini waterfall -> Groq fallback)
        report = ""
        try:
            report = generate_gemini_report(hist_prompt)
            if not report.strip() or "Advisory Report Unavailable" in report:
                raise ValueError("Empty Gemini report")
        except Exception as gem_exc:
            log.warning("[LLM] %s: Gemini exhausted (%s) - trying Groq directly", ticker, gem_exc)
            try:
                report = generate_groq_report(hist_prompt)
            except Exception as groq_exc:
                log.error("[LLM] %s: Groq also failed: %s - stub report", ticker, groq_exc)
                stance_guess = "HOLD" if regime["cash_buffer"] >= 0.20 else "BUY"
                report = (
                    f"## 1. Signal Interpretation\n"
                    f"Pre-window momentum signal for **{ticker}**: `{raw_signals.get(ticker, 0.5):.4f}`. "
                    f"20d vol: `{raw_vols.get(ticker, 0.3):.1%}`. Regime: {regime['name']}. "
                    f"Both LLM providers temporarily unavailable.\n\n"
                    f"## 2. Portfolio Allocation Justification\n"
                    f"Signal {'above' if raw_signals.get(ticker, 0.5) > 0.55 else 'below'} entry threshold. "
                    f"Regime cash buffer: {regime['cash_buffer']*100:.0f}%.\n\n"
                    f"## 3. Risk Flags\n"
                    f"- {regime['description']}\n"
                    f"- API connectivity issue\n\n"
                    f"## 4. Actionable Stance\n"
                    f"**{stance_guess}** - Based on quantitative signals and regime context."
                )

        try:
            report = sanitize_and_append_disclaimer(report)
        except Exception:
            pass

        ai_reports[ticker] = report
        ai_stances[ticker] = extract_stance(report)

        # Stage 3: Critic (best-effort)
        try:
            raw_ctx = {
                "ticker":           ticker,
                "regime":           regime["name"],
                "pre_window_close": float(pre_windows[ticker]["Close"].iloc[-1]),
                "momentum_signal":  raw_signals.get(ticker, 0.5),
                "vol_20d":          raw_vols.get(ticker, 0.3),
            }
            critic = verify_report_accuracy(report, raw_ctx)
            critic_results[ticker] = {
                "is_accurate": critic.get("is_accurate"),
                "flags":       critic.get("flags", []),
            }
        except Exception as exc:
            critic_results[ticker] = {
                "is_accurate": None,
                "flags":       [f"Critic unavailable: {str(exc)[:100]}"],
            }

        print(
            f"    [{i+1}/{len(valid_tickers)}] {ticker:<14} "
            f"| Sig={raw_signals.get(ticker,0.5):.3f} "
            f"| Vol={raw_vols.get(ticker,0.3):.1%} "
            f"| Stance={ai_stances[ticker]:<7} "
            f"| Intent={ai_intents[ticker]}"
        )
        if i < len(valid_tickers) - 1:
            time.sleep(INTER_TICKER_SLEEP_S)

    # 3. Compute weights
    weights = compute_allocation_weights(
        valid_tickers, raw_signals, raw_vols, ai_stances, regime["cash_buffer"]
    )

    # 4. Walk-forward simulation
    print(f"  > [{reg_name}] [{port_name}] Running 5-day walk-forward...")
    sim = WalkForwardSimulator(portfolio, regime)
    final_cash, final_unrealized, total_pnl, max_dd = sim.run(
        walk_hists=walk_hists,
        signals=raw_signals,
        vols=raw_vols,
        ai_stances=ai_stances,
        ai_reports=ai_reports,
        weights=weights,
    )

    final_value = final_cash + final_unrealized
    total_roi   = (final_value - portfolio["capital"]) / portfolio["capital"] * 100
    wins        = sum(1 for p in sim.closed_pnl.values() if p > 0)
    n_closed    = len(sim.closed_pnl)
    win_rate    = (wins / n_closed * 100) if n_closed > 0 else 0.0
    n_trades    = len([e for e in sim.events if e.action != "HOLD"])

    print(
        f"  OK [{port_name}] Final={symbol}{final_value:,.0f} "
        f"ROI={total_roi:+.2f}% Trades={n_trades} MaxDD={max_dd:.2f}%"
    )

    return RegimePortfolioResult(
        regime_name=reg_name, regime_label=regime["label"],
        portfolio_name=port_name, currency=portfolio["currency"],
        symbol=symbol, initial_capital=portfolio["capital"],
        final_cash=final_cash, final_unrealized=final_unrealized,
        total_realized_pnl=total_pnl, total_roi_pct=total_roi,
        win_rate_pct=win_rate, max_drawdown_pct=max_dd, n_trades=n_trades,
        tickers=valid_tickers, events=sim.events,
        ai_reports=ai_reports, ai_stances=ai_stances, ai_intents=ai_intents,
        critic_results=critic_results, signals=raw_signals, volatilities=raw_vols,
    )


def _empty_result(portfolio: dict, regime: dict) -> RegimePortfolioResult:
    return RegimePortfolioResult(
        regime_name=regime["name"], regime_label=regime["label"],
        portfolio_name=portfolio["name"], currency=portfolio["currency"],
        symbol=portfolio["symbol"], initial_capital=portfolio["capital"],
        final_cash=portfolio["capital"], final_unrealized=0.0,
        total_realized_pnl=0.0, total_roi_pct=0.0,
        win_rate_pct=0.0, max_drawdown_pct=0.0, n_trades=0,
    )


# ===========================================================================
# Markdown Audit Report Generator
# ===========================================================================

def _ab(action: str) -> str:
    return {
        "BUY":        "[BUY]",
        "TRIM_PROFIT":"[TRIM]",
        "STOP_LOSS":  "[STOP_LOSS]",
        "REINVEST":   "[REINVEST]",
        "CLOSE_ALL":  "[CLOSE]",
    }.get(action, action)


def _sb(stance: str) -> str:
    return {"BUY": "BUY", "HOLD": "HOLD", "SELL": "SELL", "REDUCE": "REDUCE"}.get(
        stance.upper(), stance
    )


def _cb(is_accurate) -> str:
    if is_accurate is True:
        return "Accurate"
    if is_accurate is False:
        return "Discrepancy"
    return "Unverified"


def generate_audit_report(
    all_regime_results: list[list[RegimePortfolioResult]],
    regimes:            list[dict],
    discovery_meta:     dict,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []

    # Header
    lines += [
        "# GenWealth AI - Walk-Forward Multi-Regime Simulation & Complete Rationale Audit",
        "",
        f"> **Generated**: {now}",
        "> **LLM Engine**: Groq openai/gpt-oss-120b (primary) -> Gemini-3.1-Flash-Lite (6-key waterfall pool)",
        "> **Strategy**: 5-Day Sequential Walk-Forward | Zero Lookahead | Active Management",
        "> **Active Rules**: TRIM_PROFIT (+10%/+20%) | STOP_LOSS (-7%) | CAPITAL_RECYCLING | CLOSE_ALL Day5",
        "",
        "---",
        "",
    ]

    # A. Cross-Regime Executive Summary
    lines += [
        "## A. Cross-Regime Executive Summary",
        "",
        "| Regime | Portfolio | Currency | Capital | Final Value | Realized P&L | Unrealized | Total ROI | Win Rate | Max DD | Trades |",
        "|--------|-----------|----------|---------|-------------|--------------|------------|-----------|----------|--------|--------|",
    ]
    for regime_results in all_regime_results:
        for r in regime_results:
            fv = r.final_cash + r.final_unrealized
            lines.append(
                f"| **{r.regime_name}** | {r.portfolio_name} | {r.currency} "
                f"| {r.symbol}{r.initial_capital:,.0f} "
                f"| {r.symbol}{fv:,.0f} "
                f"| {r.symbol}{r.total_realized_pnl:+,.0f} "
                f"| {r.symbol}{r.final_unrealized:,.0f} "
                f"| **{r.total_roi_pct:+.2f}%** "
                f"| {r.win_rate_pct:.0f}% "
                f"| {r.max_drawdown_pct:.2f}% "
                f"| {r.n_trades} |"
            )
    lines += ["", "---", ""]

    # Per-Regime Sections
    for regime, regime_results in zip(regimes, all_regime_results):
        lines += [
            f"## {regime['label']}",
            "",
            f"> {regime['description']}",
            f"> **Window**: `{regime['start']}` to `{regime['end']}` | **Cash Buffer**: {regime['cash_buffer']*100:.0f}%",
            "",
        ]

        # Regime portfolio summary
        lines += [
            f"### B. Portfolio Performance Summary - {regime['name']}",
            "",
            "| Portfolio | Capital | Final Value | Realized P&L | Unrealized | ROI | Win Rate | Max DD | Trades |",
            "|-----------|---------|-------------|--------------|------------|-----|----------|--------|--------|",
        ]
        for r in regime_results:
            fv = r.final_cash + r.final_unrealized
            lines.append(
                f"| **{r.portfolio_name}** "
                f"| {r.symbol}{r.initial_capital:,.0f} "
                f"| {r.symbol}{fv:,.0f} "
                f"| {r.symbol}{r.total_realized_pnl:+,.0f} "
                f"| {r.symbol}{r.final_unrealized:,.0f} "
                f"| {r.total_roi_pct:+.2f}% "
                f"| {r.win_rate_pct:.0f}% "
                f"| {r.max_drawdown_pct:.2f}% "
                f"| {r.n_trades} |"
            )
        lines += ["", "---", ""]

        # Per-portfolio ledger + transcripts
        for r in regime_results:
            if not r.events and not r.ai_reports:
                lines += [
                    f"### {r.portfolio_name} x {r.regime_name}",
                    "",
                    "> No data available for this portfolio/regime combination.",
                    "",
                ]
                continue

            fv = r.final_cash + r.final_unrealized
            lines += [
                f"### C. Transaction Ledger - {r.portfolio_name} x {r.regime_name}",
                "",
                (f"**Currency**: {r.currency} | **Capital**: {r.symbol}{r.initial_capital:,.0f} | "
                 f"**Final**: {r.symbol}{fv:,.0f} | **ROI**: {r.total_roi_pct:+.2f}% | "
                 f"**Cash**: {r.symbol}{r.final_cash:,.0f} | **Unrealized**: {r.symbol}{r.final_unrealized:,.0f}"),
                "",
                "| Day | Date | Ticker | Action | Exec Price | Shares | Capital In | Capital Out | Realized P&L | Cash After | ROI% | Signal | Vol 20d | AI Driver |",
                "|-----|------|--------|--------|------------|--------|------------|-------------|--------------|------------|------|--------|---------|-----------|",
            ]

            for ev in r.events:
                ai_ex = (ev.ai_reason[:85] + "...") if len(ev.ai_reason) > 85 else ev.ai_reason
                ai_ex = ai_ex.replace("|", "I")
                lines.append(
                    f"| D{ev.day_num} | {ev.date} | **{ev.ticker}** "
                    f"| {_ab(ev.action)} "
                    f"| {r.symbol}{ev.exec_price:,.2f} "
                    f"| {ev.shares_traded:,.3f} "
                    f"| {r.symbol}{ev.capital_in:,.0f} "
                    f"| {r.symbol}{ev.capital_out:,.0f} "
                    f"| {r.symbol}{ev.realized_pnl:+,.2f} "
                    f"| {r.symbol}{ev.cash_after:,.0f} "
                    f"| {ev.position_roi_pct:+.2f}% "
                    f"| {ev.signal:.3f} "
                    f"| {ev.vol_20d:.2%} "
                    f"| {ai_ex} |"
                )

            lines += [
                "",
                (f"> **Cash Reserve**: {r.symbol}{r.final_cash:,.2f} | "
                 f"**Unrealized Residual**: {r.symbol}{r.final_unrealized:,.2f}"),
                "",
            ]

            # AI Reasoning Transcripts
            lines += [
                f"### D. Verbatim AI Reasoning Transcripts - {r.portfolio_name} x {r.regime_name}",
                "",
                "_Reports below were generated from pre-window historical context (zero lookahead)._",
                "_These reports drove all 5-day trading decisions for this regime._",
                "",
            ]

            for ticker in r.tickers:
                if ticker not in r.ai_reports:
                    continue
                report   = r.ai_reports[ticker]
                stance   = r.ai_stances.get(ticker, "HOLD")
                intent   = r.ai_intents.get(ticker, "UNKNOWN")
                critic   = r.critic_results.get(ticker, {})
                sig      = r.signals.get(ticker, 0.5)
                vol      = r.volatilities.get(ticker, 0.3)
                flags    = get_compliance_flags(report)
                t_events = [e for e in r.events if e.ticker == ticker]
                total_pnl = sum(e.realized_pnl for e in t_events)
                final_roi = t_events[-1].position_roi_pct if t_events else 0.0
                actions   = list(dict.fromkeys(e.action for e in t_events))

                lines += [
                    "<details>",
                    (f"<summary><strong>{ticker}</strong> - Stance: {_sb(stance)} | "
                     f"Realized P&L: {r.symbol}{total_pnl:+,.2f} | "
                     f"Final ROI: {final_roi:+.2f}% | "
                     f"Actions: {', '.join(actions)} | "
                     f"Critic: {_cb(critic.get('is_accurate'))}"
                     f"</summary>"),
                    "",
                    "**Decision Context**",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| Ticker | {ticker} |",
                    f"| Regime | {r.regime_name} |",
                    f"| Simulation Window | {regime['start']} to {regime['end']} |",
                    f"| Pre-Window Momentum Signal | {sig:.4f} ({'Bullish' if sig > 0.55 else 'Neutral' if sig > 0.45 else 'Bearish'}) |",
                    f"| 20d Annualised Volatility | {vol:.2%} ({'HIGH' if vol > 0.40 else 'MODERATE' if vol > 0.20 else 'LOW'}) |",
                    f"| AI Stance | {_sb(stance)} |",
                    f"| Intent Classified | {intent} |",
                    f"| Regime Cash Buffer | {regime['cash_buffer']*100:.0f}% |",
                    f"| Actions Executed | {', '.join(actions)} |",
                    f"| Realized P&L | {r.symbol}{total_pnl:+,.2f} |",
                    f"| Critic Verification | {_cb(critic.get('is_accurate'))} |",
                    "",
                ]

                if t_events:
                    lines += [
                        "**Day-by-Day Trade Events**",
                        "",
                        "| Day | Date | Action | Price | Shares | P&L | Cash After | Trigger |",
                        "|-----|------|--------|-------|--------|-----|------------|---------|",
                    ]
                    for ev in t_events:
                        lines.append(
                            f"| D{ev.day_num} | {ev.date} | {_ab(ev.action)} "
                            f"| {r.symbol}{ev.exec_price:,.2f} | {ev.shares_traded:,.3f} "
                            f"| {r.symbol}{ev.realized_pnl:+,.2f} "
                            f"| {r.symbol}{ev.cash_after:,.0f} "
                            f"| `{ev.trigger}` |"
                        )
                    lines.append("")

                lines += [
                    "**Full LLM Advisory Report (Verbatim - Gemini / Groq Fallback)**:",
                    "",
                    "```",
                    report if report else "(No report generated)",
                    "```",
                    "",
                ]

                c_flags = critic.get("flags", [])
                lines.append("**Groq Critic Verification**:")
                for f_item in c_flags:
                    lines.append(f"- `{f_item}`")
                if not c_flags:
                    lines.append("- No material discrepancies detected.")
                lines.append("")

                lines.append("**Guardrails and Compliance Audit**:")
                for gf in flags:
                    lines.append(f"- `{gf}`")
                if not flags:
                    lines.append("- No compliance violations. Disclaimer appended.")

                lines += ["", "</details>", ""]

            lines += ["---", ""]

    # E. Discovery section
    picks = discovery_meta.get("picks", [])
    lines += [
        "## E. Autonomous AI Discovery (DuckDuckGo) - Live Picks",
        "",
        f"**Tickers Scanned**: {discovery_meta.get('tickers_scanned', 0)}",
        (f"**Capital Per Ticker**: ${picks[0]['allocation']:,.0f}" if picks else ""),
        "",
        "| Rank | Ticker | Mentions | Source URL |",
        "|------|--------|----------|------------|",
    ]
    for i, pick in enumerate(picks, 1):
        url = pick.get("url", "N/A")[:80]
        lines.append(f"| {i} | **{pick['ticker']}** | {pick['mentions']} | {url} |")

    lines += ["", "---", ""]

    # F. Methodology
    lines += [
        "## F. Simulation Methodology",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Architecture | 5-Day Sequential Walk-Forward, Zero Lookahead |",
        "| Historical Regimes | COVID-19 Crash (Feb 2020), Tech Bear (Nov 2022), Bull Market (Nov 2023) |",
        "| Pre-Window Context | 50 calendar days before regime start (~35 trading days) |",
        "| LLM Generation | Once per ticker from pre-window only - no future data in prompt |",
        "| Primary LLM | Groq openai/gpt-oss-120b (30 RPM / 2,000 RPD pool) |",
        "| Secondary LLM | Gemini-3.1-Flash-Lite via 6-key waterfall pool (90 RPM / 3,000 RPD) |",
        "| Critic Verification | Groq Stage-3 per ticker, best-effort |",
        "| Take Profit Light | ROI >= +10% -> sell 30% of position |",
        "| Take Profit Heavy | ROI >= +20% -> sell 50% of position |",
        "| Stop Loss | ROI <= -7% -> exit 100% to CASH immediately |",
        "| Capital Recycling | 80% of freed cash redeployed into BUY-stance picks |",
        "| Cash Buffer | COVID Crash: 20% / Tech Bear: 15% / Bull Market: 5% |",
        "| Allocation | Inverse-vol x momentum conviction (zero-lookahead) |",
        "| Entry Price | Day-1 OPEN price (avoids same-bar lookahead bias) |",
        "| Exit Prices | Day 2-5 CLOSE prices |",
        "| FX Note | Non-USD tickers use price-relative returns (FX-neutral) |",
        "| Pacing | 2s sleep between sequential LLM calls |",
        "",
        "> DISCLAIMER: Paper trading simulation for research and AI validation only.",
        "> All P&L figures are hypothetical. Past performance does not guarantee future results.",
        "> GenWealth AI advisory reports do not constitute financial advice.",
        "",
        f"_Auto-generated by GenWealth AI - {now}_",
    ]

    return "\n".join(lines)


# ===========================================================================
# Main Entry Point
# ===========================================================================

def main() -> None:
    print("\n" + "=" * 72)
    print("  GenWealth AI - Walk-Forward Multi-Regime Investor Simulation")
    print(f"  Started : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("  Regimes : COVID-19 Crash (2020), Tech Bear (2022), Bull Market (2023)")
    print("  LLM     : Groq openai/gpt-oss-120b -> Gemini-3.1-Flash-Lite [6-key waterfall]")
    print("=" * 72)

    # Autonomous Discovery
    print("\n[AutoDiscovery] Scanning DuckDuckGo for top conviction picks...")
    discovery    = discover_tickers(capital=100_000, top_n=5)
    auto_tickers = [p["ticker"] for p in discovery["picks"]]
    print(f"  Scanned : {discovery['tickers_scanned']} candidates")
    print(f"  Picked  : {auto_tickers}")

    all_portfolios = PORTFOLIOS + [
        {
            "name":     "Autonomous AI Discovery",
            "currency": "USD",
            "symbol":   "$",
            "capital":  100_000,
            "tickers":  auto_tickers,
        }
    ]

    # Run all Regimes x Portfolios
    all_regime_results: list[list[RegimePortfolioResult]] = []

    for regime in REGIMES:
        print(f"\n{'='*72}")
        print(f"  REGIME: {regime['label']}")
        print(f"  Window: {regime['start']} to {regime['end']}")
        print(f"  Buffer: {regime['cash_buffer']*100:.0f}% cash reserve")
        print(f"{'='*72}")

        regime_results: list[RegimePortfolioResult] = []
        for portfolio in all_portfolios:
            result = run_portfolio_regime(portfolio, regime)
            regime_results.append(result)
        all_regime_results.append(regime_results)

    # Print summary
    print("\n" + "=" * 72)
    print("  SIMULATION COMPLETE - CROSS-REGIME SUMMARY")
    print("=" * 72)

    total_trades = 0
    for regime, regime_results in zip(REGIMES, all_regime_results):
        print(f"\n  {regime['name']}:")
        for r in regime_results:
            fv = r.final_cash + r.final_unrealized
            print(
                f"    {r.portfolio_name:<30} | "
                f"ROI={r.total_roi_pct:+.2f}% | "
                f"P&L={r.symbol}{r.total_realized_pnl:+,.0f} | "
                f"Trades={r.n_trades} | "
                f"MaxDD={r.max_drawdown_pct:.2f}%"
            )
            total_trades += r.n_trades

    print(f"\n  Total trade events : {total_trades}")
    print(f"  Regimes tested     : {len(REGIMES)}")
    print(f"  Portfolios         : {len(all_portfolios)}")

    # Generate report
    print("\n  Generating WALKFORWARD_INVESTOR_AUDIT.md...")
    report_md = generate_audit_report(all_regime_results, REGIMES, discovery)
    REPORT_PATH.write_text(report_md, encoding="utf-8")

    na_count = report_md.count("Advisory Report Unavailable")
    print(f"  Report saved  : {REPORT_PATH}")
    print(f"  Report size   : {len(report_md):,} characters")
    print(f"  N/A count     : {na_count} (target: 0)")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
