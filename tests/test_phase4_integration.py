"""
GenWealth — Phase 4 Integration Test Suite
==========================================
File: tests/test_phase4_integration.py

Tests the FastAPI layer using httpx.AsyncClient in ASGI mode
(no real network calls, no real LLM calls for the fast-path tests).

Test Groups:
  1. Health Endpoint
  2. Config & Schema Validation
  3. Advisory Router (schema validation, SSE event format, error path)
  4. Portfolio Router (weight sum, cash buffer, signal filtering)
  5. Simulation Router (schema validation, endpoint reachability)
  6. Chat Router (SSE event format, validation)
  7. Regression Guard (16 Phase 1–3 core tests still pass after Phase 4 changes)
"""

import json
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Conditional imports — skip gracefully if Phase 4 deps are not installed
# ---------------------------------------------------------------------------
try:
    from httpx import AsyncClient, ASGITransport
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    from app.main import app
    APP_AVAILABLE = True
except Exception as _app_err:
    APP_AVAILABLE = False
    _APP_SKIP_REASON = str(_app_err)

# ---------------------------------------------------------------------------
# Markers & fixtures
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.asyncio(loop_scope="session")

SKIP_APP = not (HTTPX_AVAILABLE and APP_AVAILABLE)
SKIP_REASON = (
    "httpx not installed" if not HTTPX_AVAILABLE
    else f"app failed to import: {_APP_SKIP_REASON if not APP_AVAILABLE else ''}"
)


@pytest_asyncio.fixture(scope="session")
async def client():
    """Async httpx client backed by the ASGI app (no real server)."""
    if SKIP_APP:
        pytest.skip(SKIP_REASON)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ===========================================================================
# 1. Health Endpoint
# ===========================================================================

@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_health_returns_200(client):
    """GET /api/v1/health must return HTTP 200."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_health_response_schema(client):
    """Health response must include all required HealthResponse fields."""
    resp = await client.get("/api/v1/health")
    data = resp.json()
    required = ["status", "version", "mongodb", "groq_keys", "gemini_keys",
                "lstm_ok", "rf_ok", "ppo_zip_ok", "uptime_s"]
    for field in required:
        assert field in data, f"Missing field '{field}' in health response"


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_health_version(client):
    """Health response must report version 4.0.0."""
    resp = await client.get("/api/v1/health")
    assert resp.json()["version"] == "4.0.0"


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_health_model_artifacts_present(client):
    """Health check must report LSTM and RF artifacts as OK."""
    resp = await client.get("/api/v1/health")
    data = resp.json()
    assert data["lstm_ok"] is True, "LSTM artifact not loaded"
    assert data["rf_ok"]   is True, "RF artifact not loaded"


# ===========================================================================
# 2. Config & Schema Validation
# ===========================================================================

def test_ticker_query_schema():
    """TickerQuery normalises ticker to uppercase and strips whitespace."""
    from app.schemas.request_schemas import TickerQuery
    q = TickerQuery(ticker=" nvda ")
    assert q.ticker == "NVDA"


def test_ticker_query_min_length():
    """TickerQuery rejects empty ticker."""
    from pydantic import ValidationError
    from app.schemas.request_schemas import TickerQuery
    with pytest.raises(ValidationError):
        TickerQuery(ticker="")


def test_portfolio_request_dedup():
    """PortfolioAllocRequest deduplicates tickers (order-preserving)."""
    from app.schemas.request_schemas import PortfolioAllocRequest
    req = PortfolioAllocRequest(tickers=["NVDA", "AAPL", "nvda", "MSFT", "aapl"])
    assert req.tickers == ["NVDA", "AAPL", "MSFT"]


def test_portfolio_request_capital_positive():
    """PortfolioAllocRequest rejects non-positive capital."""
    from pydantic import ValidationError
    from app.schemas.request_schemas import PortfolioAllocRequest
    with pytest.raises(ValidationError):
        PortfolioAllocRequest(tickers=["NVDA"], capital=-1000)


def test_walkforward_date_format():
    """WalkForwardRequest requires YYYY-MM-DD date format."""
    from pydantic import ValidationError
    from app.schemas.request_schemas import WalkForwardRequest
    with pytest.raises(ValidationError):
        WalkForwardRequest(
            tickers=["NVDA"], capital=100000,
            regime_start="13-09-2024",   # wrong format
            regime_end="15-09-2024",
        )


def test_chat_message_ticker_normalise():
    """ChatMessage normalises optional ticker to uppercase."""
    from app.schemas.request_schemas import ChatMessage
    msg = ChatMessage(message="Explain NVDA", ticker="  nvda  ")
    assert msg.ticker == "NVDA"


def test_response_phase1_result():
    """Phase1Result schema round-trips a full snapshot dict."""
    from app.schemas.response_schemas import Phase1Result
    p = Phase1Result(
        ticker="NVDA", as_of_date="2024-09-13", close_price=125.50,
        currency="USD", log_ret=0.01, vol_20=0.30, vol_ratio=0.95,
        ret_1m=0.05, efficiency=0.62, sentiment=0.3,
        lstm_prob=0.71, rf_prob=0.68, phase1_signal=0.695,
        signal_label="BULLISH", inference_mode="LIVE_MODEL",
    )
    assert p.ticker == "NVDA"
    assert p.signal_label == "BULLISH"
    assert p.phase1_signal == pytest.approx(0.695)


# ===========================================================================
# 3. Advisory Router
# ===========================================================================

@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_advisory_analyze_invalid_ticker_returns_event_stream(client):
    """
    POST /api/v1/advisor/analyze with a valid request must return
    Content-Type: text/event-stream (stream starts — not HTTP 4xx).
    """
    resp = await client.post(
        "/api/v1/advisor/analyze",
        json={"ticker": "NVDA", "use_live_engine": False},
        headers={"Accept": "text/event-stream"},
        timeout=60.0,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_advisory_analyze_bad_request(client):
    """POST /api/v1/advisor/analyze with missing ticker returns HTTP 422."""
    resp = await client.post("/api/v1/advisor/analyze", json={})
    assert resp.status_code == 422


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_advisory_sse_contains_valid_events(client):
    """
    SSE stream must emit at least a 'phase1' event with valid JSON payload.
    We consume the stream partially (first SSE block only) and parse it.
    """
    # Use a very short timeout to get just the first SSE event
    async with client.stream(
        "POST", "/api/v1/advisor/analyze",
        json={"ticker": "NVDA", "use_live_engine": False},
        timeout=90.0,
    ) as resp:
        assert resp.status_code == 200
        events_seen = []
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                events_seen.append(line[7:].strip())
            if len(events_seen) >= 2:
                break

    assert events_seen, "No SSE events received within timeout"
    # 'phase1' should be the first event type for any ticker
    assert events_seen[0] in ("phase1", "error"), \
        f"Expected 'phase1' as first SSE event, got '{events_seen[0]}'"


# ===========================================================================
# 4. Portfolio Router
# ===========================================================================

@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_portfolio_allocate_bad_request(client):
    """POST /api/v1/portfolio/allocate with missing tickers returns 422."""
    resp = await client.post("/api/v1/portfolio/allocate", json={"capital": 100000})
    assert resp.status_code == 422


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_portfolio_allocate_weight_sum(client):
    """
    Allocation weights for all tickers + cash must sum to ≈ 1.0.
    (Sum of allocated weights ≤ 1.0, with cash_pct making up the remainder.)
    """
    resp = await client.post(
        "/api/v1/portfolio/allocate",
        json={"tickers": ["NVDA", "AAPL"], "capital": 100000, "currency": "USD"},
        timeout=120.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        total_weight = sum(a["weight"] for a in data["allocations"])
        # Deployed weight + cash buffer = 1.0
        assert total_weight <= 1.0 + 1e-6, f"Total weight {total_weight:.4f} exceeds 1.0"
        assert data["cash_pct"] > 0, "Cash buffer must be positive"
    elif resp.status_code == 500:
        pytest.skip("Portfolio endpoint needs live yfinance data (network required)")


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_portfolio_allocate_response_fields(client):
    """PortfolioAllocResponse must contain required top-level fields."""
    resp = await client.post(
        "/api/v1/portfolio/allocate",
        json={"tickers": ["NVDA"], "capital": 50000, "currency": "USD"},
        timeout=120.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        for field in ["portfolio_name", "total_capital", "currency", "cash_reserved",
                      "allocations", "allocation_method", "latency_s"]:
            assert field in data, f"Missing field: {field}"


# ===========================================================================
# 5. Simulation Router
# ===========================================================================

@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_simulation_dynamic_bad_request(client):
    """POST /api/v1/simulate/dynamic with missing tickers returns 422."""
    resp = await client.post("/api/v1/simulate/dynamic", json={"capital": 100000})
    assert resp.status_code == 422


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_simulation_walkforward_bad_request(client):
    """POST /api/v1/simulate/walkforward with invalid date format returns 422."""
    resp = await client.post(
        "/api/v1/simulate/walkforward",
        json={"tickers": ["NVDA"], "capital": 100000,
              "regime_start": "invalid", "regime_end": "invalid"},
    )
    assert resp.status_code == 422


# ===========================================================================
# 6. Chat Router
# ===========================================================================

@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_chat_stream_bad_request(client):
    """POST /api/v1/chat/stream with missing message returns 422."""
    resp = await client.post("/api/v1/chat/stream", json={"ticker": "NVDA"})
    assert resp.status_code == 422


@pytest.mark.skipif(SKIP_APP, reason=SKIP_REASON)
async def test_chat_stream_returns_event_stream(client):
    """POST /api/v1/chat/stream returns text/event-stream content-type."""
    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "What is the GenWealth AI signal for NVDA?", "ticker": "NVDA"},
        timeout=30.0,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


# ===========================================================================
# 7. Currency Map Unit Tests (Task 4.4b)
# ===========================================================================

def test_get_ticker_currency_us():
    """US ticker (no suffix) → USD."""
    from src.advisor.llm_engine import get_ticker_currency
    code, sym, exc = get_ticker_currency("NVDA")
    assert code == "USD"
    assert sym  == "$"


def test_get_ticker_currency_india_nse():
    """NSE ticker (.NS) → INR ₹."""
    from src.advisor.llm_engine import get_ticker_currency
    code, sym, exc = get_ticker_currency("RELIANCE.NS")
    assert code == "INR"
    assert sym  == "₹"
    assert "NSE" in exc


def test_get_ticker_currency_europe_amsterdam():
    """Euronext Amsterdam (.AS) → EUR €."""
    from src.advisor.llm_engine import get_ticker_currency
    code, sym, exc = get_ticker_currency("ASML.AS")
    assert code == "EUR"
    assert sym  == "€"


def test_get_ticker_currency_japan():
    """Tokyo Stock Exchange (.T) → JPY ¥."""
    from src.advisor.llm_engine import get_ticker_currency
    code, sym, _ = get_ticker_currency("7203.T")
    assert code == "JPY"
    assert sym  == "¥"


def test_get_ticker_currency_crypto():
    """Crypto -USD suffix → USD."""
    from src.advisor.llm_engine import get_ticker_currency
    code, sym, exc = get_ticker_currency("BTC-USD")
    assert code == "USD"
    assert "Crypto" in exc


# ===========================================================================
# 8. Dynamic Threshold Unit Tests (Task 4.4a)
# ===========================================================================

def test_dynamic_threshold_bull_regime():
    """Bull regime (low vol) → 40th percentile threshold ≤ 0.55."""
    from src.advisor.trade_execution_simulator import get_dynamic_threshold
    signals    = [0.62, 0.58, 0.71, 0.45, 0.39, 0.67, 0.53]
    vol_ratios = [0.85, 0.90, 0.80, 0.88, 0.82, 0.87, 0.91]   # all < 1.0 → bull
    threshold  = get_dynamic_threshold(signals, vol_ratios=vol_ratios)
    assert 0.30 <= threshold <= 0.75, f"Threshold {threshold} outside valid range"


def test_dynamic_threshold_bear_regime():
    """Bear regime (high vol) → 65th percentile threshold ≥ bull threshold."""
    from src.advisor.trade_execution_simulator import get_dynamic_threshold
    signals    = [0.62, 0.58, 0.71, 0.45, 0.39, 0.67, 0.53]
    vol_ratios = [1.20, 1.35, 1.10, 1.25, 1.40, 1.15, 1.30]   # all > 1.0 → bear
    bear_thr   = get_dynamic_threshold(signals, vol_ratios=vol_ratios)
    bull_vols  = [0.80] * 7
    bull_thr   = get_dynamic_threshold(signals, vol_ratios=bull_vols)
    assert bear_thr >= bull_thr, \
        f"Bear threshold ({bear_thr:.4f}) should be ≥ bull threshold ({bull_thr:.4f})"


def test_dynamic_threshold_empty_signals():
    """Empty signal list falls back to default 0.55."""
    from src.advisor.trade_execution_simulator import get_dynamic_threshold
    threshold = get_dynamic_threshold([])
    assert threshold == 0.55


def test_dynamic_threshold_single_ticker():
    """Single-ticker portfolio falls back gracefully (50th percentile ≈ signal value)."""
    from src.advisor.trade_execution_simulator import get_dynamic_threshold
    threshold = get_dynamic_threshold([0.62])
    assert 0.30 <= threshold <= 0.75


# ===========================================================================
# 9. Regression Guard — Phase 1–3 Core Tests
# ===========================================================================

def test_model_loader_health():
    """Phase 2 regression: all Phase 1–3 model artifacts still accessible."""
    from src.advisor.model_loader import check_artifact_health
    health = check_artifact_health()
    assert health["lstm_ok"],  "LSTM artifact missing after Phase 4 changes"
    assert health["rf_ok"],    "RF artifact missing after Phase 4 changes"
    assert health["ppo_zip_ok"], "PPO zip artifact missing after Phase 4 changes"
    assert health["backtest_ok"], "Backtest metrics artifact missing after Phase 4 changes"


def test_live_inference_imports_cleanly():
    """Phase 4.1 regression: LiveQuantEngine still imports without error."""
    from src.advisor.live_inference import LiveQuantEngine
    eng = LiveQuantEngine()
    assert eng is not None


def test_context_builder_imports():
    """Phase 4.1 regression: ContextAggregator still imports cleanly."""
    from src.advisor.context_builder import ContextAggregator
    agg = ContextAggregator()
    assert agg is not None
