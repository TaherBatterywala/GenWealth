"""
GenWealth — Dynamic Trade Execution Simulator
=============================================
File : src/advisor/trade_execution_simulator.py

Implements multi-step portfolio rebalancing over a ~30-trading-day horizon.

Rules applied at each weekly checkpoint (every 5 trading days):
  BUY            : Signal > 0.55 AND AI stance in {BUY, HOLD}  →  enter position
  TRIM_PROFIT    : ROI > +15%  →  sell 30% of shares, book profit to CASH
  STOP_LOSS_EXIT : ROI < -8%   →  exit full position to CASH
  VOL_EXIT       : vol_ratio > 1.3 (short/long vol spike)  →  exit to CASH
  CLOSE_ALL      : Final step (day ~30)  →  liquidate all remaining positions

SmartPortfolioAllocator uses inverse-volatility × Phase1 conviction weighting.
Tickers outside Phase 1 training set receive a momentum-based signal proxy.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("genwealth.trade_exec")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Rule Constants
# ---------------------------------------------------------------------------
# NOTE: SIGNAL_BUY_THRESHOLD is now REPLACED by get_dynamic_threshold().
# The static value is retained only as a last-resort default if signal list
# is empty (e.g. single-ticker portfolios where quantile is undefined).
_SIGNAL_BUY_THRESHOLD_DEFAULT = 0.55
PROFIT_TAKE_ROI      = 0.15   # +15% → TRIM 30%
TRIM_FRACTION        = 0.30   # Fraction sold on profit-take
STOP_LOSS_ROI        = -0.08  # -8% → full EXIT
VOL_RATIO_SPIKE      = 1.30   # short/long vol ratio → EXIT
SIM_STEP_DAYS        = 5      # Trading days between checks
PRICE_FETCH_PERIOD   = "65d"  # yfinance pull window (~3 months for buffer)

POSITIVE_STANCES = {"BUY", "HOLD"}
NEGATIVE_STANCES = {"SELL", "REDUCE"}


def get_dynamic_threshold(
    signals:   list[float],
    vol_ratios: list[float] | None = None,
    regime:    str = "auto",
) -> float:
    """
    Compute a regime-adaptive signal entry threshold.

    Replaces the static ``SIGNAL_BUY_THRESHOLD = 0.55`` with a quantile-based
    threshold that responds to the current market regime:

        - **Bull regime** (vol_ratio median < 1.0):
          40th percentile of the signal distribution.
          Deploys capital more aggressively when momentum is broad.
        - **Bear regime** (vol_ratio median ≥ 1.0):
          65th percentile — tightens the entry filter to preserve capital.
        - **Auto** (default): classifies regime using vol_ratio median;
          falls back to 50th percentile if vol_ratio is not supplied.

    Args:
        signals:    List of Phase 1 composite signals for all tickers.
        vol_ratios: Optional list of Vol_20/Vol_200 ratios per ticker.
                    Required for ``regime="auto"`` classification.
        regime:     ``"auto"`` | ``"bull"`` | ``"bear"``.

    Returns:
        float: Dynamic entry threshold in the range [0.30, 0.75].
               Falls back to 0.55 if ``signals`` is empty.

    Example:
        >>> signals = [0.62, 0.58, 0.71, 0.45, 0.39]
        >>> get_dynamic_threshold(signals, regime="bull")
        0.45   # 40th percentile → lower bar → deploy more capital
        >>> get_dynamic_threshold(signals, regime="bear")
        0.62   # 65th percentile → higher bar → stay defensive
    """
    if not signals:
        return _SIGNAL_BUY_THRESHOLD_DEFAULT

    sig_arr = np.array(signals, dtype=float)

    # ── Regime classification ─────────────────────────────────────────────────
    if regime == "auto":
        if vol_ratios and len(vol_ratios) > 0:
            vol_median = float(np.median(vol_ratios))
            regime = "bear" if vol_median >= 1.0 else "bull"
        else:
            # No vol data — use 50th percentile as neutral threshold
            return float(np.clip(np.percentile(sig_arr, 50), 0.30, 0.75))

    if regime == "bull":
        threshold = float(np.percentile(sig_arr, 40))  # lower bar — deploy more
    else:  # bear
        threshold = float(np.percentile(sig_arr, 65))  # higher bar — stay defensive

    threshold = float(np.clip(threshold, 0.30, 0.75))
    logger.debug("[DynamicThreshold] regime=%s -> threshold=%.4f", regime, threshold)
    return threshold


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Tracks an open position during the simulation."""
    ticker:           str
    shares:           float
    entry_price:      float
    capital_invested: float
    entry_step:       int
    entry_date:       str

    def current_value(self, price: float) -> float:
        return self.shares * price

    def roi(self, price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        return (price - self.entry_price) / self.entry_price

    def unrealised_pnl(self, price: float) -> float:
        return self.current_value(price) - self.capital_invested


@dataclass
class LedgerEntry:
    """A single recorded trade event in the simulation ledger."""
    step:          int
    date:          str
    ticker:        str
    action:        str      # BUY | TRIM_PROFIT | STOP_LOSS_EXIT | VOL_EXIT | CLOSE_ALL
    trigger:       str      # SIGNAL | PROFIT_TAKE | STOP_LOSS | VOL_SPIKE | END_OF_SIM
    exec_price:    float
    shares_traded: float
    capital_in:    float    # Cash invested (positive for BUY)
    capital_out:   float    # Cash received (positive for exits)
    realized_pnl:  float    # Realised profit/loss from this action
    cash_balance:  float    # Running cash balance after action
    position_roi:  float    # ROI (%) at time of action
    signal:        float    # Phase1 signal (or momentum proxy)
    vol_ratio:     float    # Short/long volatility ratio at checkpoint
    ai_reason:     str      # Extracted from AI advisory report


@dataclass
class SimulationResult:
    """Aggregated result from a full DynamicTradeSimulator run."""
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
    ledger:                list = field(default_factory=list)
    portfolio_curve:       list = field(default_factory=list)  # [(step, total_value)]


# ---------------------------------------------------------------------------
# Price / Signal Helpers
# ---------------------------------------------------------------------------

def load_price_history(ticker: str, period: str = PRICE_FETCH_PERIOD) -> Optional[pd.DataFrame]:
    """Pull adjusted OHLCV history via yfinance. Returns None on failure."""
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 10:
            logger.warning("[yfinance] %s: insufficient data (%d rows)", ticker, len(hist))
            return None
        return hist
    except Exception as exc:
        logger.error("[yfinance] %s: fetch failed — %s", ticker, exc)
        return None


def momentum_signal(prices: pd.Series, window: int = 10) -> float:
    """
    Momentum-based Phase1 signal proxy for out-of-scope tickers.

    Maps 10-day price return to [0.30, 0.70] via tanh compression.
    Values > 0.55 indicate positive momentum (BUY eligible).
    """
    if len(prices) < window + 1:
        return 0.50
    ret = float((prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window])
    return float(np.clip(0.50 + 0.25 * np.tanh(ret * 8), 0.30, 0.70))


def rolling_vol_ratio(prices: pd.Series, short: int = 5, long: int = 20) -> float:
    """Ratio of short-term / long-term annualised realised volatility."""
    rets = prices.pct_change().dropna()
    if len(rets) < long:
        return 1.0
    short_vol = float(rets.iloc[-short:].std()) * (252 ** 0.5)
    long_vol  = float(rets.iloc[-long:].std())  * (252 ** 0.5)
    return short_vol / long_vol if long_vol > 1e-9 else 1.0


def extract_ai_reason(report: str, max_chars: int = 300) -> str:
    """
    Extract the Actionable Stance (Section 4) as the primary AI justification.
    Falls back to the first max_chars of the report if Section 4 is absent.
    """
    match = re.search(
        r'##\s*4[.\s]+Actionable\s+Stance(.*?)(?=##\s*[1-9]|---|\Z)',
        report, re.DOTALL | re.IGNORECASE
    )
    if match:
        text = match.group(1).strip()
        return text[:max_chars].replace("\n", " ")
    # Remove markdown headers and truncate
    clean = re.sub(r'##.*\n', '', report)
    return clean[:max_chars].strip().replace("\n", " ")


def extract_stance(report: str) -> str:
    """Extract the directional stance from a report (BUY/HOLD/SELL/REDUCE)."""
    for bold_token in ["**BUY**", "**HOLD**", "**SELL**", "**REDUCE**"]:
        if bold_token in report:
            return bold_token.replace("**", "")
    for token in ["BUY", "SELL", "REDUCE", "HOLD"]:
        if re.search(r'\b' + token + r'\b', report.upper()):
            return token
    return "HOLD"


# ---------------------------------------------------------------------------
# Smart Portfolio Allocator
# ---------------------------------------------------------------------------

class SmartPortfolioAllocator:
    """
    Inverse-volatility weighting × conviction (Phase1 signal) allocation.

    weight[i] = (1/vol[i]) × signal[i]   →   normalised to sum = 1.0

    Only tickers passing the entry filter (signal > threshold AND AI stance
    in {BUY, HOLD}) receive capital. The remainder stays in CASH.
    """

    def __init__(
        self,
        tickers:    list[str],
        price_hist: dict[str, Optional[pd.DataFrame]],
        signals:    dict[str, float],
        ai_stances: dict[str, str],
    ):
        self.tickers    = tickers
        self.price_hist = price_hist
        self.signals    = signals
        self.ai_stances = ai_stances

    def compute_weights(self) -> dict[str, float]:
        """
        Returns a dict {ticker: weight} where weights sum to 1.0 for
        eligible tickers and 0.0 for tickers kept in CASH.

        Uses ``get_dynamic_threshold()`` to compute a regime-adaptive entry
        threshold based on the full cross-sectional signal distribution and
        vol_ratio readings (40th pct bull / 65th pct bear).
        """
        # Compute vol ratios for regime detection
        vol_ratios_list: list[float] = []
        for t in self.tickers:
            hist = self.price_hist.get(t)
            if hist is not None and len(hist) >= 20:
                vr = rolling_vol_ratio(hist["Close"])
                vol_ratios_list.append(vr)

        # Dynamic threshold — regime-adaptive (replaces hardcoded 0.55)
        signal_list = [self.signals.get(t, 0.50) for t in self.tickers]
        threshold   = get_dynamic_threshold(signal_list, vol_ratios=vol_ratios_list)
        logger.info("[Allocator] Dynamic entry threshold: %.4f", threshold)

        # Filter eligible tickers
        eligible = []
        for t in self.tickers:
            sig    = self.signals.get(t, 0.50)
            stance = self.ai_stances.get(t, "HOLD").upper()
            if sig > threshold or stance in POSITIVE_STANCES:
                eligible.append(t)

        if not eligible:
            logger.warning("[Allocator] No tickers pass entry filter — holding all in CASH")
            return {t: 0.0 for t in self.tickers}

        # Compute annualised realised volatility (20-day) for eligible tickers
        vols: dict[str, float] = {}
        for t in eligible:
            hist = self.price_hist.get(t)
            if hist is not None and len(hist) >= 10:
                rets = hist["Close"].pct_change().dropna()
                v = float(rets.tail(20).std()) * (252 ** 0.5)
                vols[t] = max(v, 0.05)   # floor at 5% to avoid division by zero
            else:
                vols[t] = 0.30           # default 30% annualised vol

        # Composite score: inverse-vol × conviction signal
        scores: dict[str, float] = {}
        for t in eligible:
            scores[t] = (1.0 / vols[t]) * self.signals.get(t, 0.50)

        total_score = sum(scores.values())
        weights_eligible = {t: scores[t] / total_score for t in eligible}

        # Non-eligible → 0
        final_weights: dict[str, float] = {}
        for t in self.tickers:
            final_weights[t] = weights_eligible.get(t, 0.0)

        return final_weights


# ---------------------------------------------------------------------------
# Dynamic Trade Simulator
# ---------------------------------------------------------------------------

class DynamicTradeSimulator:
    """
    Simulates multi-step portfolio rebalancing over a ~30-trading-day horizon.

    Steps are taken every ``SIM_STEP_DAYS`` trading days (default: 5).
    For a 30-trading-day window that is approximately 6 rebalance checkpoints.

    Usage::

        sim = DynamicTradeSimulator("US Tech", 100_000, ["NVDA","AAPL"], "USD", "$")
        sim.load_histories()
        result = sim.run(signals, ai_stances, ai_reports)
    """

    def __init__(
        self,
        portfolio_name:  str,
        initial_capital: float,
        tickers:         list[str],
        currency:        str,
        symbol:          str,
    ):
        self.portfolio_name  = portfolio_name
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.currency        = currency
        self.symbol          = symbol
        self.tickers         = tickers

        # Runtime state
        self.positions:    dict[str, Position]              = {}
        self.ledger:       list[LedgerEntry]               = []
        self.closed_pnl:   dict[str, float]                = {}   # ticker → cumulative realised P&L
        self.price_hist:   dict[str, Optional[pd.DataFrame]] = {}
        self.portfolio_curve: list[tuple[int, float]]      = []

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load_histories(self) -> None:
        """Fetch yfinance price history for all portfolio tickers."""
        for t in self.tickers:
            self.price_hist[t] = load_price_history(t)
            logger.info("[Sim] %s: %s rows loaded",
                t,
                len(self.price_hist[t]) if self.price_hist[t] is not None else 0)

    # ── Price / Date Access ───────────────────────────────────────────────────

    def _price_at(self, ticker: str, step: int) -> Optional[float]:
        """
        Map simulation step (0 = 30 days ago, N = today) to a Close price.

        Step 0  → row at  max(0, total - 31)
        Step N  → row at  min(total - 1, row0 + N * SIM_STEP_DAYS)
        """
        hist = self.price_hist.get(ticker)
        if hist is None or hist.empty:
            return None
        n = len(hist)
        row0 = max(0, n - 31)
        if step == 0:
            idx = row0
        else:
            idx = min(n - 1, row0 + step * SIM_STEP_DAYS)
        try:
            return float(hist["Close"].iloc[idx])
        except Exception:
            return None

    def _date_at(self, ticker: str, step: int) -> str:
        hist = self.price_hist.get(ticker)
        if hist is None or hist.empty:
            return f"Step-{step}"
        n = len(hist)
        row0 = max(0, n - 31)
        idx = row0 if step == 0 else min(n - 1, row0 + step * SIM_STEP_DAYS)
        try:
            return str(hist.index[idx].date())
        except Exception:
            return f"Step-{step}"

    def _prices_up_to(self, ticker: str, step: int) -> Optional[pd.Series]:
        """Close prices from day 0 up to (inclusive) the current step."""
        hist = self.price_hist.get(ticker)
        if hist is None or hist.empty:
            return None
        n = len(hist)
        row0 = max(0, n - 31)
        end  = min(n, row0 + (step + 1) * SIM_STEP_DAYS)
        return hist["Close"].iloc[:end]

    # ── Portfolio Value ───────────────────────────────────────────────────────

    def _total_portfolio_value(self, step: int) -> float:
        val = self.cash
        for t, pos in self.positions.items():
            px = self._price_at(t, step)
            if px:
                val += pos.current_value(px)
        return val

    def _max_drawdown(self) -> float:
        if len(self.portfolio_curve) < 2:
            return 0.0
        values = [v for _, v in self.portfolio_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd * 100.0

    # ── Main Simulation Loop ──────────────────────────────────────────────────

    def run(
        self,
        signals:    dict[str, float],
        ai_stances: dict[str, str],
        ai_reports: dict[str, str],
    ) -> SimulationResult:
        """
        Execute the full dynamic simulation and return a ``SimulationResult``.

        Args:
            signals    : Phase1 signal (or momentum proxy) per ticker.
            ai_stances : Directional stance (BUY/HOLD/SELL/REDUCE) per ticker.
            ai_reports : Full advisory report text per ticker (for AI reason).

        Returns:
            SimulationResult with ledger, portfolio curve, and summary stats.
        """
        N_STEPS = 6   # 6 × 5 trading days ≈ 30-day simulation window

        # ── Step 0: Initial Allocation ────────────────────────────────────────
        allocator = SmartPortfolioAllocator(
            tickers=self.tickers,
            price_hist=self.price_hist,
            signals=signals,
            ai_stances=ai_stances,
        )
        weights = allocator.compute_weights()

        for ticker, weight in weights.items():
            if weight < 0.001:
                continue
            entry_px = self._price_at(ticker, 0)
            if not entry_px or entry_px <= 0:
                logger.warning("[Sim] %s: no entry price — skipping", ticker)
                continue

            cap = self.cash * weight
            shares = cap / entry_px
            self.positions[ticker] = Position(
                ticker=ticker,
                shares=shares,
                entry_price=entry_px,
                capital_invested=cap,
                entry_step=0,
                entry_date=self._date_at(ticker, 0),
            )
            self.cash -= cap

            ai_reason = extract_ai_reason(ai_reports.get(ticker, ""), 280)
            self.ledger.append(LedgerEntry(
                step=0,
                date=self._date_at(ticker, 0),
                ticker=ticker,
                action="BUY",
                trigger="SIGNAL",
                exec_price=entry_px,
                shares_traded=shares,
                capital_in=cap,
                capital_out=0.0,
                realized_pnl=0.0,
                cash_balance=self.cash,
                position_roi=0.0,
                signal=signals.get(ticker, 0.5),
                vol_ratio=1.0,
                ai_reason=f"[Weight {weight*100:.1f}%] {ai_reason}",
            ))

        self.portfolio_curve.append((0, self._total_portfolio_value(0)))

        # ── Steps 1–6: Rebalance Checkpoints ─────────────────────────────────
        for step in range(1, N_STEPS + 1):
            is_final = (step == N_STEPS)

            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                px  = self._price_at(ticker, step) or pos.entry_price
                roi = pos.roi(px)
                ser = self._prices_up_to(ticker, step)
                vol_r = rolling_vol_ratio(ser) if ser is not None and len(ser) >= 6 else 1.0
                date  = self._date_at(ticker, step)
                ai_r  = extract_ai_reason(ai_reports.get(ticker, ""), 200)

                if is_final:
                    # ── Liquidate all remaining positions ─────────────────────
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.ledger.append(LedgerEntry(
                        step=step, date=date, ticker=ticker,
                        action="CLOSE_ALL", trigger="END_OF_SIM",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_balance=self.cash,
                        position_roi=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_ratio=vol_r,
                        ai_reason=(
                            f"End-of-simulation liquidation. "
                            f"Final ROI: {roi*100:+.2f}%. {ai_r}"
                        ),
                    ))
                    del self.positions[ticker]

                elif roi > PROFIT_TAKE_ROI:
                    # ── Trim 30% on profit-take ───────────────────────────────
                    trim_shares = pos.shares * TRIM_FRACTION
                    proceeds    = trim_shares * px
                    cost_basis  = (pos.capital_invested / pos.shares) * trim_shares
                    pnl         = proceeds - cost_basis
                    self.cash  += proceeds
                    pos.shares -= trim_shares
                    pos.capital_invested -= cost_basis
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.ledger.append(LedgerEntry(
                        step=step, date=date, ticker=ticker,
                        action="TRIM_PROFIT", trigger="PROFIT_TAKE",
                        exec_price=px, shares_traded=trim_shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_balance=self.cash,
                        position_roi=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_ratio=vol_r,
                        ai_reason=(
                            f"Profit-taking: ROI={roi*100:+.2f}% exceeded +15% "
                            f"threshold. Trimmed {TRIM_FRACTION*100:.0f}% "
                            f"({trim_shares:.2f} shares). Booked {self.symbol}{pnl:+,.2f} to CASH."
                        ),
                    ))

                elif roi < STOP_LOSS_ROI:
                    # ── Full stop-loss exit ───────────────────────────────────
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.ledger.append(LedgerEntry(
                        step=step, date=date, ticker=ticker,
                        action="STOP_LOSS_EXIT", trigger="STOP_LOSS",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_balance=self.cash,
                        position_roi=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_ratio=vol_r,
                        ai_reason=(
                            f"Stop-loss triggered: ROI={roi*100:+.2f}% breached -8% floor. "
                            f"Full position ({pos.shares:.2f} shares) exited to CASH at "
                            f"{self.symbol}{px:,.2f}."
                        ),
                    ))
                    del self.positions[ticker]

                elif vol_r > VOL_RATIO_SPIKE:
                    # ── Volatility spike exit ─────────────────────────────────
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.ledger.append(LedgerEntry(
                        step=step, date=date, ticker=ticker,
                        action="VOL_EXIT", trigger="VOL_SPIKE",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_balance=self.cash,
                        position_roi=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_ratio=vol_r,
                        ai_reason=(
                            f"Volatility spike exit: vol_ratio={vol_r:.2f} exceeded 1.3 threshold. "
                            f"Risk management exit at {self.symbol}{px:,.2f}."
                        ),
                    ))
                    del self.positions[ticker]

            self.portfolio_curve.append((step, self._total_portfolio_value(step)))

        # ── Summary Statistics ────────────────────────────────────────────────
        total_realized = sum(self.closed_pnl.values())
        # Any leftover open positions (shouldn't exist after CLOSE_ALL but guard anyway)
        residual = sum(
            pos.current_value(self._price_at(pos.ticker, N_STEPS) or pos.entry_price)
            for pos in self.positions.values()
        )
        # Remove residual from cash if it wasn't properly closed
        total_final_value = self.cash + residual
        total_roi = (total_final_value - self.initial_capital) / self.initial_capital * 100
        # Trade-level win rate (evaluates all closed trades: profit takes, stop losses, liquidations)
        closed_trades = [
            e for e in self.ledger
            if e.action in ("TRIM_PROFIT", "STOP_LOSS_EXIT", "VOL_EXIT", "CLOSE_ALL")
            and abs(e.realized_pnl) > 1e-4
        ]
        if closed_trades:
            trade_wins = sum(1 for e in closed_trades if e.realized_pnl > 0)
            win_rate = (trade_wins / len(closed_trades)) * 100.0
        else:
            wins = sum(1 for p in self.closed_pnl.values() if p > 0)
            n_closed = len(self.closed_pnl)
            win_rate = (wins / n_closed * 100) if n_closed > 0 else 0.0

        return SimulationResult(
            portfolio_name=self.portfolio_name,
            currency=self.currency,
            symbol=self.symbol,
            initial_capital=self.initial_capital,
            final_cash=self.cash,
            final_positions_value=residual,
            total_realized_pnl=total_final_value - self.initial_capital,
            total_roi_pct=total_roi,
            win_rate_pct=win_rate,
            max_drawdown_pct=self._max_drawdown(),
            n_trades=len([e for e in self.ledger if e.action not in ("HOLD",)]),
            ledger=self.ledger,
            portfolio_curve=self.portfolio_curve,
        )


# ===========================================================================
# Walk-Forward Simulation Components
# ===========================================================================
# Promoted from tests/test_walkforward_investor.py into the main package so
# they are importable as production APIs.
#
# These implement a strict zero-lookahead 5-day walk-forward backtester that
# can replay the GenWealth pipeline across historical market regimes without
# any forward-looking bias.
# ===========================================================================

import time as _time
from datetime import timedelta

# ---------------------------------------------------------------------------
# Walk-Forward Constants
# ---------------------------------------------------------------------------
WF_TRIM_THRESHOLD_LO  = 0.10   # ROI > +10% → trim 30%
WF_TRIM_THRESHOLD_HI  = 0.20   # ROI > +20% → trim 50%
WF_TRIM_FRACTION_LO   = 0.30
WF_TRIM_FRACTION_HI   = 0.50
WF_STOP_LOSS_ROI      = -0.07  # ROI < -7% → exit 100%
WF_SIGNAL_BUY_FLOOR   = 0.50   # Momentum signal floor for entry
WF_REINVEST_FRACTION  = 0.80   # Fraction of freed cash recycled into BUYs

# ---------------------------------------------------------------------------
# Walk-Forward Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DayEvent:
    """A single trade event recorded during a walk-forward simulation day."""
    day_num:          int
    date:             str
    ticker:           str
    action:           str    # BUY | TRIM_PROFIT | STOP_LOSS | REINVEST | CLOSE_ALL
    trigger:          str
    exec_price:       float
    shares_traded:    float
    capital_in:       float
    capital_out:      float
    realized_pnl:     float
    cash_after:       float
    position_roi_pct: float
    signal:           float
    vol_20d:          float  # Annualised 20-day realised vol at time of event
    ai_reason:        str


@dataclass
class RegimePortfolioResult:
    """Aggregated result for one portfolio tested under one historical market regime."""
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


# ---------------------------------------------------------------------------
# Historical Data Utilities
# ---------------------------------------------------------------------------

def fetch_regime_data(
    ticker:        str,
    regime_start:  str,
    regime_end:    str,
    context_days:  int = 50,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV covering the pre-window context + 5-day walk-forward window.

    ``context_days`` calendar days before ``regime_start`` form the pre-window context
    used by signal computation and LLM advisory. The walk-forward window
    (``regime_start`` → ``regime_end``) is fetched but intentionally withheld from the
    LLM — it is only released one day at a time during ``WalkForwardSimulator.run()``.

    Args:
        ticker:       Yahoo Finance ticker symbol.
        regime_start: Regime start date in "YYYY-MM-DD" format.
        regime_end:   Regime end date in "YYYY-MM-DD" format.
        context_days: Calendar days before regime_start to include as pre-window.

    Returns:
        Combined OHLCV DataFrame or None on failure.
    """
    from datetime import datetime as _dt
    start_dt   = _dt.strptime(regime_start, "%Y-%m-%d") - timedelta(days=context_days)
    end_dt     = _dt.strptime(regime_end,   "%Y-%m-%d") + timedelta(days=4)
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
            logger.warning("[WF] %s: insufficient rows (%d) for %s->%s",
                           ticker, len(hist), regime_start, regime_end)
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as exc:
        logger.error("[WF] %s: fetch failed — %s", ticker, exc)
        return None


def split_regime(
    hist:         pd.DataFrame,
    regime_start: str,
    regime_end:   str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split full OHLCV history into pre-window and walk-forward window.

    Zero-Lookahead Contract:
        ``pre_window`` contains rows **strictly before** ``regime_start``.
        ``walk_days`` contains rows within ``[regime_start, regime_end]``.
        The walk_days slice is NEVER passed to the LLM; the simulator releases
        it one day at a time.

    Returns:
        (pre_window, walk_days) as separate DataFrames.
    """
    start_ts = pd.Timestamp(regime_start)
    end_ts   = pd.Timestamp(regime_end)
    pre  = hist[hist.index < start_ts].copy()
    walk = hist[(hist.index >= start_ts) & (hist.index <= end_ts)].copy()
    return pre, walk


# ---------------------------------------------------------------------------
# Quantitative Signal Utilities
# ---------------------------------------------------------------------------

def compute_ticker_signals(
    pre_window: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Compute (momentum_signal, vol_20d_annualised, vol_ratio_5d_20d) from pre-window OHLCV.

    Uses only the pre-window to ensure zero-lookahead compliance.

    Returns:
        (signal, vol_20d, vol_ratio) — all floats.
        signal  ∈ [0.30, 0.70]
        vol_20d annualised, floored at 5%.
        vol_ratio = 5-day vol / 20-day vol.
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
    cash_buffer: float = 0.05,
) -> dict[str, float]:
    """
    Inverse-volatility × conviction weighting with regime-specific cash buffer.

    Weights sum to ``1 - cash_buffer``. Only tickers with signal > threshold
    OR positive AI stance (BUY/HOLD) receive capital. The remainder stays in CASH.

    Args:
        tickers:     Ordered list of ticker symbols.
        signals:     Per-ticker momentum signal (0–1).
        vols:        Per-ticker 20-day annualised realised volatility.
        ai_stances:  Per-ticker AI directional stance (BUY/HOLD/SELL/REDUCE).
        cash_buffer: Fraction of capital to withhold as CASH (0.0–1.0).

    Returns:
        Dict {ticker: weight} where weights sum to ``1 - cash_buffer``.
    """
    POSITIVE = {"BUY", "HOLD"}
    eligible = [
        t for t in tickers
        if signals.get(t, 0.5) > WF_SIGNAL_BUY_FLOOR
        or ai_stances.get(t, "HOLD").upper() in POSITIVE
    ]
    if not eligible:
        logger.warning("[WF Allocator] No tickers pass entry filter — all in CASH")
        return {t: 0.0 for t in tickers}

    scores: dict[str, float] = {
        t: (1.0 / max(vols.get(t, 0.30), 0.05)) * signals.get(t, 0.50)
        for t in eligible
    }
    total_score = sum(scores.values())
    deployable  = 1.0 - cash_buffer
    w_elig = {t: (scores[t] / total_score) * deployable for t in eligible}
    return {t: w_elig.get(t, 0.0) for t in tickers}


# ---------------------------------------------------------------------------
# Autonomous Ticker Discovery
# ---------------------------------------------------------------------------

import re as _re

_TICKER_RE = _re.compile(r"\b([A-Z]{2,5}(?:\.[A-Z]{1,3})?)\b")
_DDG_STOPWORDS = {
    "CEO", "CFO", "IPO", "ETF", "GDP", "AI", "US", "UK", "EU", "IN",
    "FY", "Q1", "Q2", "Q3", "Q4", "EPS", "PE", "PB", "YOY", "QOQ",
    "TTM", "EV", "API", "LLC", "INC", "LTD", "CORP", "SA", "PLC",
    "AG", "NY", "LA", "DC", "BV", "CTO", "THE", "AND", "FOR", "BUT",
}


def discover_tickers(capital: float = 100_000, top_n: int = 5) -> dict:
    """
    Autonomously discover high-conviction tickers via DuckDuckGo news scraping.

    Scans three growth/momentum news queries, extracts uppercase ticker-like
    tokens, ranks by mention frequency, and returns the top ``top_n`` candidates
    with equal-weight capital allocation.

    Args:
        capital: Total portfolio capital for the allocation calculation.
        top_n:   Number of tickers to select.

    Returns:
        dict with keys:
            ``tickers_scanned``  — total unique candidates seen
            ``picks``            — list of {ticker, mentions, allocation, url}
            ``capital``          — input capital value

    Falls back to ``["NVDA","META","GOOG","AMZN","MSFT"]`` if no candidates found.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("[AutoDiscovery] ddgs not installed; returning fallback tickers")
        fb = ["NVDA", "META", "GOOG", "AMZN", "MSFT"][:top_n]
        per = round(capital / top_n, 2)
        return {
            "tickers_scanned": 0,
            "picks": [{"ticker": t, "mentions": 0, "allocation": per, "url": ""} for t in fb],
            "capital": capital,
        }

    logger.info("[AutoDiscovery] Scanning DuckDuckGo...")
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
                    for h in _TICKER_RE.findall(text):
                        if h not in _DDG_STOPWORDS and len(h) >= 2:
                            candidates.setdefault(h, {"mentions": 0, "url": ""})
                            candidates[h]["mentions"] += 1
                            if not candidates[h]["url"]:
                                candidates[h]["url"] = item.get("url", "")
            _time.sleep(1.2)
        except Exception as exc:
            logger.warning("[AutoDiscovery] DDG query failed: %s", exc)

    ranked = sorted(
        [(t, d) for t, d in candidates.items() if t not in _DDG_STOPWORDS],
        key=lambda x: x[1]["mentions"], reverse=True,
    )[:top_n]

    if not ranked:
        logger.warning("[AutoDiscovery] No tickers found — using fallback list")
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


# ---------------------------------------------------------------------------
# Walk-Forward Simulator
# ---------------------------------------------------------------------------

@dataclass
class _WFPos:
    """Internal: tracks a single open position inside WalkForwardSimulator."""
    ticker:           str
    shares:           float
    entry_price:      float
    capital_invested: float
    entry_day:        int
    entry_date:       str

    def roi(self, px: float) -> float:
        return (px - self.entry_price) / self.entry_price if self.entry_price > 0 else 0.0

    def current_value(self, px: float) -> float:
        return self.shares * px


class WalkForwardSimulator:
    """
    5-day sequential walk-forward portfolio simulator with zero-lookahead guarantee.

    Zero-Lookahead Contract
    -----------------------
    Pre-window OHLCV (rows before ``regime_start``) is used ONLY for signal
    computation and LLM advisory context. The walk-forward window is released
    to the simulator one trading day at a time — no future prices are ever
    visible to the decision engine.

    Active Management (evaluated at each day's close)
    --------------------------------------------------
    TRIM_PROFIT    : ROI ≥ +10% → sell 30%  |  ROI ≥ +20% → sell 50%
    STOP_LOSS      : ROI ≤ -7%  → exit 100% to CASH
    REINVEST       : Freed cash (80%) recycled into remaining BUY candidates
    CLOSE_ALL      : Day 5 — liquidate all remaining positions at close

    Usage
    -----
    ::

        sim = WalkForwardSimulator(portfolio_cfg, regime_cfg)
        final_cash, unrealized, total_pnl, max_dd = sim.run(
            walk_hists, signals, vols, ai_stances, ai_reports, weights
        )
        events = sim.events          # list[DayEvent]
        closed_pnl = sim.closed_pnl  # dict[ticker, float]
    """

    def __init__(self, portfolio: dict, regime: dict) -> None:
        self.portfolio       = portfolio
        self.regime          = regime
        self.cash            = float(portfolio["capital"])
        self.initial_capital = float(portfolio["capital"])
        self.cash_buffer     = float(regime.get("cash_buffer", 0.05))
        self.symbol          = portfolio.get("symbol", "$")
        self.positions:   dict[str, _WFPos]     = {}
        self.events:      list[DayEvent]        = []
        self.closed_pnl:  dict[str, float]      = {}
        self._curve:      list[tuple[int, float]] = []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _total_value(self, day_prices: dict[str, float]) -> float:
        return self.cash + sum(
            pos.current_value(day_prices.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )

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

    # ── Main simulation loop ──────────────────────────────────────────────────

    def run(
        self,
        walk_hists: dict[str, pd.DataFrame],
        signals:    dict[str, float],
        vols:       dict[str, float],
        ai_stances: dict[str, str],
        ai_reports: dict[str, str],
        weights:    dict[str, float],
    ) -> tuple[float, float, float, float]:
        """
        Execute the 5-day walk-forward simulation.

        Args:
            walk_hists: Per-ticker OHLCV for the walk-forward window only.
            signals:    Pre-window momentum signals per ticker.
            vols:       Pre-window 20-day annualised vol per ticker.
            ai_stances: AI directional stance per ticker (BUY/HOLD/SELL/REDUCE).
            ai_reports: Full advisory report text per ticker.
            weights:    Capital allocation weight per ticker (from compute_allocation_weights).

        Returns:
            (final_cash, final_unrealized, total_pnl, max_drawdown_pct)
        """
        tickers = self.portfolio["tickers"]
        valid   = [t for t in tickers if t in walk_hists and not walk_hists[t].empty]

        # ── Day 1: Entry at open price ────────────────────────────────────────
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
            self.positions[ticker] = _WFPos(
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
        d1_date   = self._dt(walk_hists[valid[0]], 0) if valid else "Day 1"
        self._curve.append((d1_date, self._total_value(d1_closes)))
        max_days = max(
            (len(wh) for wh in walk_hists.values() if wh is not None and not wh.empty),
            default=1,
        )

        # ── Days 2 to max_days: Active management over full custom date regime ─
        today_closes: dict[str, float] = {}
        for day_idx in range(1, max_days):
            day_num  = day_idx + 1
            is_final = (day_idx >= max_days - 1)

            today_closes = {}
            today_dates: dict[str, str] = {}
            for t in valid:
                px = self._px(walk_hists.get(t), day_idx)
                if px:
                    today_closes[t] = px
                    today_dates[t]  = self._dt(walk_hists.get(t), day_idx)

            freed_today   = 0.0
            newly_closed: list[str] = []

            for ticker in list(self.positions.keys()):
                pos  = self.positions[ticker]
                px   = today_closes.get(ticker, pos.entry_price)
                roi  = pos.roi(px)
                date = today_dates.get(ticker, f"Day-{day_num}")
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
                        ai_reason=f"Day-5 liquidation at {self.symbol}{px:,.2f}. Final ROI: {roi*100:+.2f}%.",
                    ))
                    del self.positions[ticker]

                elif roi >= WF_TRIM_THRESHOLD_HI:
                    frac       = WF_TRIM_FRACTION_HI
                    trim_sh    = pos.shares * frac
                    proceeds   = trim_sh * px
                    cost_basis = (pos.capital_invested / pos.shares) * trim_sh
                    pnl        = proceeds - cost_basis
                    self.cash += proceeds
                    freed_today += proceeds
                    pos.shares           -= trim_sh
                    pos.capital_invested -= cost_basis
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="TRIM_PROFIT", trigger=f"ROI>={WF_TRIM_THRESHOLD_HI*100:.0f}pct",
                        exec_price=px, shares_traded=trim_sh,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"Aggressive trim: ROI={roi*100:+.2f}% exceeded +{WF_TRIM_THRESHOLD_HI*100:.0f}%. "
                            f"Sold {frac*100:.0f}% ({trim_sh:,.3f} sh). Locked {self.symbol}{pnl:+,.2f}."
                        ),
                    ))

                elif roi >= WF_TRIM_THRESHOLD_LO:
                    frac       = WF_TRIM_FRACTION_LO
                    trim_sh    = pos.shares * frac
                    proceeds   = trim_sh * px
                    cost_basis = (pos.capital_invested / pos.shares) * trim_sh
                    pnl        = proceeds - cost_basis
                    self.cash += proceeds
                    freed_today += proceeds
                    pos.shares           -= trim_sh
                    pos.capital_invested -= cost_basis
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="TRIM_PROFIT", trigger=f"ROI>={WF_TRIM_THRESHOLD_LO*100:.0f}pct",
                        exec_price=px, shares_traded=trim_sh,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"Profit trim: ROI={roi*100:+.2f}% crossed +{WF_TRIM_THRESHOLD_LO*100:.0f}%. "
                            f"Sold {frac*100:.0f}% ({trim_sh:,.3f} sh). Realized {self.symbol}{pnl:+,.2f}."
                        ),
                    ))

                elif roi < WF_STOP_LOSS_ROI:
                    proceeds = pos.shares * px
                    pnl      = proceeds - pos.capital_invested
                    self.cash += proceeds
                    freed_today += proceeds
                    self.closed_pnl[ticker] = self.closed_pnl.get(ticker, 0.0) + pnl
                    newly_closed.append(ticker)
                    self.events.append(DayEvent(
                        day_num=day_num, date=date, ticker=ticker,
                        action="STOP_LOSS", trigger=f"ROI<{WF_STOP_LOSS_ROI*100:.0f}pct",
                        exec_price=px, shares_traded=pos.shares,
                        capital_in=0.0, capital_out=proceeds,
                        realized_pnl=pnl, cash_after=self.cash,
                        position_roi_pct=roi * 100,
                        signal=signals.get(ticker, 0.5), vol_20d=vol20,
                        ai_reason=(
                            f"STOP-LOSS: ROI={roi*100:+.2f}% breached {WF_STOP_LOSS_ROI*100:.0f}% floor. "
                            f"Full position ({pos.shares:,.3f} sh) exited at {self.symbol}{px:,.2f}. "
                            f"Loss: {self.symbol}{pnl:+,.2f}."
                        ),
                    ))
                    del self.positions[ticker]

            # ── Capital Recycling (REINVEST) ──────────────────────────────────
            if not is_final and freed_today > self.initial_capital * 0.005:
                candidates = [
                    t for t in valid
                    if t not in self.positions
                    and t not in newly_closed
                    and ai_stances.get(t, "HOLD").upper() == "BUY"
                    and signals.get(t, 0.5) > WF_SIGNAL_BUY_FLOOR
                ]
                if candidates:
                    cand_sc = {
                        t: (1.0 / max(vols.get(t, 0.30), 0.05)) * signals.get(t, 0.5)
                        for t in candidates
                    }
                    total_sc = sum(cand_sc.values())
                    deploy   = freed_today * WF_REINVEST_FRACTION
                    for t in candidates:
                        rc   = (cand_sc[t] / total_sc) * deploy
                        px_r = today_closes.get(t)
                        if not px_r or px_r <= 0:
                            continue
                        sh_r = rc / px_r
                        self.positions[t] = _WFPos(
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

            date_today = today_dates.get(valid[0], f"Day {day_num}") if valid else f"Day {day_num}"
            self._curve.append((date_today, self._total_value(today_closes)))

        self.portfolio_curve = self._curve

        # ── Final statistics ──────────────────────────────────────────────────
        residual = sum(
            pos.current_value(today_closes.get(t, pos.entry_price))
            for t, pos in self.positions.items()
        )
        total_final = self.cash + residual
        return self.cash, residual, total_final - self.initial_capital, self._max_drawdown()
