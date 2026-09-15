"""
GenWealth — Advisory Router (SSE Streaming)
============================================
File: app/routers/advisor.py

POST /api/v1/advisor/analyze
    Streams the 4-stage advisory pipeline as Server-Sent Events:
        event: phase1   → Phase 1 signal snapshot JSON
        event: phase2   → Phase 2 PPO allocation JSON
        event: rag      → Top-k RAG documents JSON
        event: report   → Full 4-stage Markdown advisory report
        event: complete → Final AdvisoryReportResponse JSON
        event: error    → Error payload
"""

import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.schemas.request_schemas import TickerQuery
from app.schemas.response_schemas import (
    AdvisoryReportResponse, Phase1Result, Phase2Result, RAGDocument, ComplianceInfo,
)

logger = logging.getLogger("genwealth.api.advisor")

router = APIRouter(prefix="/api/v1/advisor", tags=["Advisory"])


def _make_sse(event: str, data: dict | list | str) -> str:
    """Format a single SSE frame."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


def _build_phase1_result(snap: dict) -> Phase1Result:
    """Convert a phase1_quant snapshot dict to a Phase1Result schema."""
    return Phase1Result(
        ticker=snap.get("ticker", ""),
        as_of_date=snap.get("as_of_date", "N/A"),
        close_price=snap.get("close_price", 0.0),
        currency=snap.get("currency", "USD"),
        log_ret=snap.get("log_ret", 0.0),
        vol_20=snap.get("vol_20", 0.0),
        vol_ratio=snap.get("vol_ratio", 1.0),
        ret_1m=snap.get("ret_1m", 0.0),
        efficiency=snap.get("efficiency", 0.5),
        sentiment=snap.get("sentiment", 0.0),
        lstm_prob=snap.get("lstm_prob"),
        rf_prob=snap.get("rf_prob"),
        phase1_signal=snap.get("phase1_signal", 0.5),
        signal_label=snap.get("signal_label", "NEUTRAL"),
        inference_mode=snap.get("inference_mode", "CSV_CACHE"),
        news_headlines=snap.get("news_headlines", []),
    )


def _build_phase2_result(alloc: dict, method: str = "BACKTEST_METRICS") -> Phase2Result:
    """Convert a phase2_portfolio dict to a Phase2Result schema."""
    return Phase2Result(
        strategy=alloc.get("strategy", "PPO Agent"),
        total_return_pct=alloc.get("total_return_pct", 0.0),
        equal_weight_ret_pct=alloc.get("equal_weight_ret_pct", 0.0),
        sharpe_ratio=alloc.get("sharpe_ratio", 0.0),
        max_drawdown_pct=alloc.get("max_drawdown_pct", 0.0),
        calmar_ratio=alloc.get("calmar_ratio", 0.0),
        final_value_usd=alloc.get("final_value_usd", 0.0),
        alpha_vs_equal_weight=alloc.get("alpha_vs_equal_weight", "N/A"),
        allocation=alloc.get("allocation", {}),
        cash_pct=alloc.get("cash_pct", 5.0),
        allocation_method=method,
    )


def _build_rag_docs(rag: list[dict]) -> list[RAGDocument]:
    """Convert RAG documents list to RAGDocument schema list."""
    docs = []
    for d in rag:
        meta = d.get("metadata", {})
        docs.append(RAGDocument(
            title=meta.get("title", "Article"),
            source=meta.get("source", "Unknown"),
            url=meta.get("url", ""),
            snippet=d.get("text_content", "")[:300],
            similarity_score=d.get("similarity_score", 0.0),
            timestamp=str(d.get("timestamp", "")) if d.get("timestamp") else None,
        ))
    return docs


async def _advisory_stream(ticker: str, use_live_engine: bool) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the full 4-stage advisory pipeline and
    yields SSE frames for each completed stage.
    """
    t0 = time.perf_counter()

    try:
        # ── Stage 1: Build context (Phase 1 + Phase 2 + RAG) ─────────────────
        from src.advisor.context_builder import ContextAggregator
        agg = ContextAggregator()
        ctx = agg.build_ticker_context(ticker, use_live_engine=use_live_engine)

        # Phase 1 snapshot
        p1_raw = ctx.get("phase1_quant", {})
        live   = ctx.get("live_signal")  # LiveSignalResult or None

        # Enrich snapshot with live signal fields if available
        # ── Always resolve correct currency from ticker suffix ────────────────
        from src.advisor.llm_engine import get_ticker_currency
        currency_code, currency_symbol, exchange_name = get_ticker_currency(ticker)

        if live is not None:
            p1_raw.update({
                "currency":       live.currency or currency_code,
                "lstm_prob":      live.lstm_prob,
                "rf_prob":        live.rf_prob,
                "signal_label":   live.signal_label,
                "inference_mode": live.inference_mode,
                "news_headlines": live.news_headlines,
            })
        else:
            # CSV-based: infer label from signal value; always use proper currency
            sig = p1_raw.get("phase1_signal", 0.5)
            p1_raw.setdefault("signal_label", "BULLISH" if sig >= 0.65 else ("BEARISH" if sig <= 0.35 else "NEUTRAL"))
            p1_raw.setdefault("inference_mode", "CSV_CACHE")
            # Override any stale "USD" with the correct exchange currency
            p1_raw["currency"] = currency_code

        # Always inject exchange name so frontend can show it
        p1_raw["exchange"] = exchange_name

        phase1_result = _build_phase1_result(p1_raw)
        yield _make_sse("phase1", phase1_result.model_dump())

        # Phase 2 snapshot
        p2_raw = ctx.get("phase2_portfolio", {})
        if "error" not in p2_raw:
            phase2_result = _build_phase2_result(p2_raw)
            yield _make_sse("phase2", phase2_result.model_dump())
        else:
            yield _make_sse("phase2", {"error": p2_raw.get("error", "Phase 2 unavailable")})

        # RAG documents
        rag_docs = _build_rag_docs(ctx.get("phase3_rag", []))
        yield _make_sse("rag", [d.model_dump() for d in rag_docs])

        # ── Stage 2: LLM Advisory Report ──────────────────────────────────────
        from src.advisor.llm_engine import LLMAdvisorEngine
        from src.advisor.guardrails import sanitize_and_append_disclaimer, get_compliance_flags

        prompt  = agg.build_llm_prompt_context(ticker)
        engine  = LLMAdvisorEngine()
        report  = engine.generate_report(ticker=ticker, prompt_context=prompt)

        # Compliance guardrails
        flags       = get_compliance_flags(report)
        safe_report = sanitize_and_append_disclaimer(report)

        compliance = ComplianceInfo(
            flags_detected=len(flags),
            flag_details=flags,
            disclaimer_added=True,
        )
        yield _make_sse("report", {"content": safe_report})

        # ── Final payload ──────────────────────────────────────────────────────
        final = AdvisoryReportResponse(
            ticker=ticker,
            report=safe_report,
            phase1=phase1_result,
            phase2=phase2_result if "error" not in p2_raw else None,
            rag_docs=rag_docs,
            compliance=compliance,
            intent="SINGLE_TICKER_NEWS",
            latency_s=round(time.perf_counter() - t0, 2),
        )
        yield _make_sse("complete", final.model_dump())

    except Exception as exc:
        logger.error("[Advisor] Stream error for '%s': %s", ticker, exc, exc_info=True)
        yield _make_sse("error", {
            "ticker": ticker,
            "error":  f"{type(exc).__name__}: {exc}",
            "latency_s": round(time.perf_counter() - t0, 2),
        })


@router.post(
    "/analyze",
    summary="Analyze Ticker — SSE Stream",
    response_description=(
        "Server-Sent Events stream: phase1 → phase2 → rag → report → complete"
    ),
)
async def analyze_ticker(body: TickerQuery, request: Request) -> StreamingResponse:
    """
    Run the full 4-stage GenWealth advisory pipeline for a given ticker,
    streaming results as Server-Sent Events.

    **SSE Event Types**:
    | Event      | Payload                                    |
    |------------|--------------------------------------------|
    | `phase1`   | Phase1Result JSON                          |
    | `phase2`   | Phase2Result JSON (or error dict)          |
    | `rag`      | List of RAGDocument JSON objects           |
    | `report`   | `{"content": "<full markdown report>"}` |
    | `complete` | AdvisoryReportResponse JSON                |
    | `error`    | `{"error": "...", "ticker": "..."}` |

    **Consumer example (JavaScript)**:
    ```js
    const resp = await fetch('/api/v1/advisor/analyze', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ticker: 'NVDA'})
    });
    const reader = resp.body.getReader();
    ```
    """
    return StreamingResponse(
        _advisory_stream(body.ticker, body.use_live_engine),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
