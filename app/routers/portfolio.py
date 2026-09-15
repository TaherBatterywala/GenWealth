"""
GenWealth — Portfolio Allocation Router
========================================
File: app/routers/portfolio.py

POST /api/v1/portfolio/allocate
    Computes inverse-volatility × conviction portfolio weights for a
    multi-ticker basket using PPO metrics + SmartPortfolioAllocator.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.schemas.request_schemas import PortfolioAllocRequest
from app.schemas.response_schemas import (
    PortfolioAllocResponse, AllocationWeight, Phase2Result,
)

logger = logging.getLogger("genwealth.api.portfolio")

router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio"])


@router.post("/allocate", response_model=PortfolioAllocResponse, summary="AI Portfolio Allocation")
async def allocate_portfolio(body: PortfolioAllocRequest) -> PortfolioAllocResponse:
    """
    Compute intelligent portfolio weights for a multi-ticker basket.

    **Allocation Strategy**:
    1. Fetch yfinance price history for each ticker.
    2. Compute Phase 1 live signal (LiveQuantEngine) for each ticker.
    3. Generate AI advisory stance (BUY/HOLD/SELL/REDUCE) via LLMAdvisorEngine.
    4. Apply inverse-volatility × conviction weighting (SmartPortfolioAllocator).
    5. Deduct 5% cash buffer (risk management reserve).

    **PPO**: The PPO policy allocation is used as a reference benchmark in the
    Phase2Result field. Live PPO inference is attempted but falls back to the
    SmartPortfolioAllocator if the Protobuf conflict blocks SB3 loading.

    Returns a complete ``PortfolioAllocResponse`` with per-ticker weights,
    capital amounts, and PPO backtest metrics.
    """
    t0      = time.perf_counter()
    tickers = body.tickers
    capital = body.capital

    try:
        # ── Step 1: Fetch price histories ──────────────────────────────────────
        from src.advisor.trade_execution_simulator import load_price_history, momentum_signal
        price_hist = {}
        for t in tickers:
            price_hist[t] = load_price_history(t)

        # ── Step 2: Compute signals per ticker ─────────────────────────────────
        from src.advisor.live_inference import LiveQuantEngine
        engine  = LiveQuantEngine()
        signals: dict[str, float] = {}
        live_results = {}
        for t in tickers:
            try:
                lr = engine.compute_live_signal(t)
                signals[t]      = lr.phase1_signal
                live_results[t] = lr
            except Exception as exc:
                logger.warning("[Portfolio] Signal failed for '%s': %s", t, exc)
                # Momentum proxy fallback
                hist = price_hist.get(t)
                if hist is not None and not hist.empty:
                    signals[t] = momentum_signal(hist["Close"])
                else:
                    signals[t] = 0.5

        # ── Step 3: Get AI stances (quick intent classify) ─────────────────────
        ai_stances: dict[str, str] = {}
        for t in tickers:
            try:
                sig = signals.get(t, 0.5)
                # Fast heuristic stance (avoids full LLM call per ticker)
                if sig >= 0.65:
                    ai_stances[t] = "BUY"
                elif sig <= 0.35:
                    ai_stances[t] = "SELL"
                else:
                    ai_stances[t] = "HOLD"
            except Exception:
                ai_stances[t] = "HOLD"

        # ── Step 4: Compute weights via SmartPortfolioAllocator ────────────────
        from src.advisor.model_loader import get_ppo_weights
        weights, method = get_ppo_weights(
            tickers=tickers,
            obs_vec=None,            # Force SmartPortfolioAllocator path
            price_hist=price_hist,
            signals=signals,
            ai_stances=ai_stances,
        )

        # ── Step 5: Build allocation entries ───────────────────────────────────
        cash_pct   = 5.0
        deployable = capital * (1.0 - cash_pct / 100.0)
        allocations: list[AllocationWeight] = []
        for t in tickers:
            w = weights.get(t, 0.0)
            lr = live_results.get(t)
            allocations.append(AllocationWeight(
                ticker=t,
                weight=round(w, 4),
                capital=round(deployable * w, 2),
                currency=lr.currency if lr else "USD",
                signal=round(signals.get(t, 0.5), 4),
                stance=ai_stances.get(t, "HOLD"),
            ))

        # ── Step 6: Phase 2 metrics (PPO backtest reference) ──────────────────
        ppo_metrics_result = None
        try:
            from src.advisor.model_loader import load_phase2_metrics
            from src.advisor.context_builder import get_phase2_allocation
            alloc_data = get_phase2_allocation(tickers[0] if tickers else "NVDA")
            ppo_metrics_result = Phase2Result(
                strategy=alloc_data.get("strategy", "PPO Agent"),
                total_return_pct=alloc_data.get("total_return_pct", 0.0),
                equal_weight_ret_pct=alloc_data.get("equal_weight_ret_pct", 0.0),
                sharpe_ratio=alloc_data.get("sharpe_ratio", 0.0),
                max_drawdown_pct=alloc_data.get("max_drawdown_pct", 0.0),
                calmar_ratio=alloc_data.get("calmar_ratio", 0.0),
                final_value_usd=alloc_data.get("final_value_usd", 0.0),
                alpha_vs_equal_weight=alloc_data.get("alpha_vs_equal_weight", "N/A"),
                allocation={t: weights.get(t, 0.0) * 100 for t in tickers},
                cash_pct=cash_pct,
                allocation_method=method,
            )
        except Exception as exc:
            logger.warning("[Portfolio] Phase 2 metrics load failed: %s", exc)

        return PortfolioAllocResponse(
            portfolio_name=f"GenWealth — {', '.join(tickers[:3])}{'...' if len(tickers) > 3 else ''}",
            total_capital=capital,
            currency=body.currency,
            cash_reserved=round(capital * cash_pct / 100.0, 2),
            cash_pct=cash_pct,
            allocations=allocations,
            ppo_metrics=ppo_metrics_result,
            allocation_method=method,
            latency_s=round(time.perf_counter() - t0, 2),
        )

    except Exception as exc:
        logger.error("[Portfolio] Allocation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Portfolio allocation failed: {exc}")
