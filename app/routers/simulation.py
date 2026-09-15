"""
GenWealth — Simulation Router
===============================
File: app/routers/simulation.py

POST /api/v1/simulate/dynamic
    30-day DynamicTradeSimulator (6 weekly rebalance checkpoints).

POST /api/v1/simulate/walkforward
    5-day zero-lookahead WalkForwardSimulator over a historical regime.
"""

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from app.schemas.request_schemas import SimulationRequest, WalkForwardRequest
from app.schemas.response_schemas import SimResultResponse, LedgerEntryResponse

logger = logging.getLogger("genwealth.api.simulation")

router = APIRouter(prefix="/api/v1/simulate", tags=["Simulation"])


def _ledger_to_response(ledger) -> list[LedgerEntryResponse]:
    """Convert LedgerEntry / DayEvent dataclass list to Pydantic schema list."""
    out = []
    for e in ledger:
        # DynamicTradeSimulator uses LedgerEntry; WalkForwardSimulator uses DayEvent
        # Both share the same essential fields with minor naming differences.
        out.append(LedgerEntryResponse(
            step=getattr(e, "step", getattr(e, "day_num", 0)),
            date=getattr(e, "date", ""),
            ticker=getattr(e, "ticker", ""),
            action=getattr(e, "action", ""),
            trigger=getattr(e, "trigger", ""),
            exec_price=getattr(e, "exec_price", 0.0),
            shares_traded=getattr(e, "shares_traded", 0.0),
            capital_in=getattr(e, "capital_in", 0.0),
            capital_out=getattr(e, "capital_out", 0.0),
            realized_pnl=getattr(e, "realized_pnl", 0.0),
            cash_balance=getattr(e, "cash_balance", getattr(e, "cash_after", 0.0)),
            position_roi=getattr(e, "position_roi", getattr(e, "position_roi_pct", 0.0)),
            signal=getattr(e, "signal", 0.5),
            vol_ratio=getattr(e, "vol_ratio", getattr(e, "vol_20d", 1.0)),
            ai_reason=getattr(e, "ai_reason", ""),
        ))
    return out


# ===========================================================================
# Dynamic Simulator
# ===========================================================================

@router.post(
    "/dynamic",
    response_model=SimResultResponse,
    summary="30-Day Dynamic Trade Simulation",
)
async def run_dynamic_simulation(body: SimulationRequest) -> SimResultResponse:
    """
    Run the 30-trading-day DynamicTradeSimulator for a given portfolio.

    **Simulation Rules** (evaluated every 5 trading days):
    - **BUY**: Signal > threshold AND AI stance in {BUY, HOLD}
    - **TRIM_PROFIT**: ROI > +15% → sell 30% of position
    - **STOP_LOSS**: ROI < -8% → exit full position to cash
    - **VOL_EXIT**: vol_ratio > 1.3 → exit full position to cash
    - **CLOSE_ALL**: Final step (day ~30) → liquidate all

    **Capital allocation**: SmartPortfolioAllocator (inverse-vol × conviction).
    Tickers outside the Phase 1 CSV receive a live yfinance momentum signal.
    """
    t0      = time.perf_counter()
    tickers = body.tickers
    capital = body.capital

    try:
        from src.advisor.live_inference import LiveQuantEngine, _detect_currency
        from src.advisor.trade_execution_simulator import DynamicTradeSimulator
        from src.advisor.llm_engine import LLMAdvisorEngine
        from src.advisor.context_builder import ContextAggregator

        # ── Compute signals & AI stances ──────────────────────────────────────
        engine = LiveQuantEngine()
        signals:    dict[str, float] = {}
        ai_stances: dict[str, str]   = {}
        ai_reports: dict[str, str]   = {}
        currency    = "USD"
        sym         = "$"

        agg     = ContextAggregator()
        llm_eng = LLMAdvisorEngine()

        for t in tickers:
            try:
                lr = engine.compute_live_signal(t)
                signals[t] = lr.phase1_signal
                currency, sym = lr.currency, ("₹" if lr.currency == "INR" else
                                               ("€" if lr.currency == "EUR" else
                                                ("¥" if lr.currency == "JPY" else "$")))
                ctx    = agg.build_ticker_context(t, use_live_engine=False)
                prompt = agg.build_llm_prompt_context(t)
                report = llm_eng.generate_report(ticker=t, prompt_context=prompt)
                ai_reports[t] = report

                # Quick stance extraction
                from src.advisor.trade_execution_simulator import extract_stance
                ai_stances[t] = extract_stance(report)
            except Exception as exc:
                logger.warning("[Sim/Dynamic] '%s' signal/report failed: %s", t, exc)
                signals[t]    = 0.5
                ai_stances[t] = "HOLD"
                ai_reports[t] = ""

        # ── Run DynamicTradeSimulator ──────────────────────────────────────────
        sim = DynamicTradeSimulator(
            portfolio_name=body.portfolio_name,
            initial_capital=capital,
            tickers=tickers,
            currency=currency,
            symbol=sym,
        )
        sim.load_histories()
        result = sim.run(signals=signals, ai_stances=ai_stances, ai_reports=ai_reports)

        return SimResultResponse(
            portfolio_name=result.portfolio_name,
            currency=result.currency,
            symbol=result.symbol,
            initial_capital=result.initial_capital,
            final_cash=round(result.final_cash, 2),
            final_positions_value=round(result.final_positions_value, 2),
            total_realized_pnl=round(result.total_realized_pnl, 2),
            total_roi_pct=round(result.total_roi_pct, 4),
            win_rate_pct=round(result.win_rate_pct, 2),
            max_drawdown_pct=round(result.max_drawdown_pct, 4),
            n_trades=result.n_trades,
            ledger=_ledger_to_response(result.ledger),
            portfolio_curve=[[s, round(v, 2)] for s, v in result.portfolio_curve],
            simulation_type="DYNAMIC",
            latency_s=round(time.perf_counter() - t0, 2),
        )

    except Exception as exc:
        logger.error("[Sim/Dynamic] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dynamic simulation failed: {exc}")


# ===========================================================================
# Walk-Forward Simulator
# ===========================================================================

@router.post(
    "/walkforward",
    response_model=SimResultResponse,
    summary="5-Day Walk-Forward Backtest",
)
async def run_walkforward_simulation(body: WalkForwardRequest) -> SimResultResponse:
    """
    Run the 5-day zero-lookahead WalkForwardSimulator over a historical market regime.

    **Zero-Lookahead Contract**: Pre-window OHLCV (before ``regime_start``) is used
    for signal computation and LLM advisory only. The walk-forward window is released
    one trading day at a time — no future prices are ever visible to the engine.

    **Management Rules** (evaluated at each day's close):
    - TRIM_PROFIT: ROI ≥ +10% → sell 30%  |  ROI ≥ +20% → sell 50%
    - STOP_LOSS:   ROI ≤ -7%  → exit 100% to CASH
    - REINVEST:    Freed cash (80%) recycled into remaining BUY candidates
    - CLOSE_ALL:   Day 5 — liquidate all remaining positions
    """
    t0      = time.perf_counter()
    tickers = body.tickers
    capital = body.capital

    try:
        from src.advisor.trade_execution_simulator import (
            fetch_regime_data, split_regime, compute_ticker_signals,
            compute_allocation_weights, WalkForwardSimulator,
        )
        from src.advisor.context_builder import ContextAggregator
        from src.advisor.llm_engine import LLMAdvisorEngine
        from src.advisor.trade_execution_simulator import extract_stance

        agg     = ContextAggregator()
        llm_eng = LLMAdvisorEngine()

        # ── Background Hidden Constraints & Sanitization ─────────────────────
        from datetime import datetime, timedelta
        regime_start = body.regime_start
        regime_end   = body.regime_end

        try:
            start_dt = datetime.strptime(regime_start, "%Y-%m-%d")
            end_dt   = datetime.strptime(regime_end, "%Y-%m-%d")
            today    = datetime.now()

            # Constraint 1: No future dates beyond today
            if end_dt > today:
                end_dt = today
                regime_end = end_dt.strftime("%Y-%m-%d")

            # Constraint 2: End date strictly after start date (min 2 days)
            if end_dt <= start_dt:
                end_dt = start_dt + timedelta(days=5)
                regime_end = end_dt.strftime("%Y-%m-%d")

            # Constraint 3: Max historical lookback limit (5 years)
            earliest_allowed = today - timedelta(days=365 * 5)
            if start_dt < earliest_allowed:
                start_dt = earliest_allowed
                regime_start = start_dt.strftime("%Y-%m-%d")

        except Exception as date_exc:
            logger.warning("[WF] Date normalization: %s", date_exc)

        # ── Fetch historical data & split ──────────────────────────────────────
        walk_hists: dict = {}
        signals:    dict[str, float] = {}
        vols:       dict[str, float] = {}
        ai_stances: dict[str, str]   = {}
        ai_reports: dict[str, str]   = {}
        currency = "USD"
        sym      = "$"

        for t in tickers:
            hist = fetch_regime_data(t, regime_start, regime_end)
            if hist is None:
                logger.warning("[WF] '%s': no data — skipping", t)
                continue

            pre_win, walk_win = split_regime(hist, regime_start, regime_end)
            walk_hists[t] = walk_win

            sig, vol, _ = compute_ticker_signals(pre_win)
            signals[t] = sig
            vols[t]    = vol

            # AI advisory on pre-window
            try:
                prompt = agg.build_llm_prompt_context(t)
                report = llm_eng.generate_report(ticker=t, prompt_context=prompt)
                ai_reports[t] = report
                ai_stances[t] = extract_stance(report)
            except Exception as exc:
                logger.warning("[WF] LLM failed for '%s': %s", t, exc)
                ai_stances[t] = "HOLD"
                ai_reports[t] = ""

        if not walk_hists:
            raise HTTPException(
                status_code=400,
                detail="No price data available for any ticker in the specified regime.",
            )

        # ── Compute weights ────────────────────────────────────────────────────
        valid_tickers = list(walk_hists.keys())
        weights = compute_allocation_weights(
            tickers=valid_tickers,
            signals=signals,
            vols=vols,
            ai_stances=ai_stances,
            cash_buffer=0.05,
        )

        # ── Run WalkForwardSimulator ───────────────────────────────────────────
        portfolio_cfg = {
            "name":     body.portfolio_name,
            "tickers":  valid_tickers,
            "capital":  capital,
            "symbol":   sym,
        }
        regime_cfg = {
            "name":         f"{regime_start}→{regime_end}",
            "label":        "CUSTOM",
            "cash_buffer":  0.05,
        }

        sim = WalkForwardSimulator(portfolio=portfolio_cfg, regime=regime_cfg)
        final_cash, residual, total_pnl, max_dd = sim.run(
            walk_hists=walk_hists,
            signals=signals,
            vols=vols,
            ai_stances=ai_stances,
            ai_reports=ai_reports,
            weights=weights,
        )

        final_value = final_cash + residual
        roi_pct     = (total_pnl / capital * 100) if capital > 0 else 0.0
        # Trade-level win rate for walk-forward simulation
        closed_events = [
            e for e in sim.events
            if e.action in ("TRIM_PROFIT", "STOP_LOSS", "CLOSE_ALL")
            and abs(e.realized_pnl) > 1e-4
        ]
        if closed_events:
            wf_wins = sum(1 for e in closed_events if e.realized_pnl > 0)
            win_rate = (wf_wins / len(closed_events)) * 100.0
        else:
            wins     = sum(1 for p in sim.closed_pnl.values() if p > 0)
            n_closed = len(sim.closed_pnl)
            win_rate = (wins / n_closed * 100) if n_closed > 0 else 0.0

        return SimResultResponse(
            portfolio_name=body.portfolio_name,
            currency=currency,
            symbol=sym,
            initial_capital=capital,
            final_cash=round(final_cash, 2),
            final_positions_value=round(residual, 2),
            total_realized_pnl=round(total_pnl, 2),
            total_roi_pct=round(roi_pct, 4),
            win_rate_pct=round(win_rate, 2),
            max_drawdown_pct=round(max_dd, 4),
            n_trades=len(sim.events),
            ledger=_ledger_to_response(sim.events),
            portfolio_curve=[[s, round(v, 2)] for s, v in sim._curve],
            simulation_type="CUSTOM_RANGE",
            latency_s=round(time.perf_counter() - t0, 2),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[WF] Error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Walk-forward simulation failed: {exc}")
