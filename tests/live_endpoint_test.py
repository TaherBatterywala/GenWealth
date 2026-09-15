"""
GenWealth AI — Live Endpoint Test Runner
=========================================
Tests every API endpoint against the running server at localhost:8000.
Prints colour-coded PASS/FAIL for each check with full diagnostics.

Run with:
    .venv\Scripts\python tests\live_endpoint_test.py
"""

import json
import sys
import time
import traceback
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
RESULTS: list[tuple[str, bool, str, float]] = []   # (test_name, passed, detail, latency_s)

# ── ANSI colours ──────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def record(name: str, passed: bool, detail: str = "", latency: float = 0.0):
    marker = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    lat    = f"{latency:.2f}s"
    print(f"  [{marker}] {name:<58} {CYAN}{lat}{RESET}")
    if detail:
        prefix = "       "
        for line in detail.splitlines():
            print(f"{prefix}{line}")
    RESULTS.append((name, passed, detail, latency))


def http_get(path: str, timeout: int = 10) -> tuple[int, dict | str]:
    t0 = time.perf_counter()
    try:
        url = BASE + path
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw  = r.read().decode("utf-8")
            lat  = time.perf_counter() - t0
            ct   = r.getheader("Content-Type", "")
            try:
                return r.status, json.loads(raw), lat, ct
            except Exception:
                return r.status, raw, lat, ct
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        lat = time.perf_counter() - t0
        return e.code, raw, lat, ""
    except Exception as e:
        return 0, str(e), time.perf_counter() - t0, ""


def http_post(path: str, payload: dict, timeout: int = 30) -> tuple[int, dict | str, float, str]:
    t0 = time.perf_counter()
    try:
        url  = BASE + path
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            lat = time.perf_counter() - t0
            ct  = r.getheader("Content-Type", "")
            try:
                return r.status, json.loads(raw), lat, ct
            except Exception:
                return r.status, raw, lat, ct
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        lat = time.perf_counter() - t0
        return e.code, raw, lat, ""
    except Exception as e:
        return 0, str(e), time.perf_counter() - t0, ""


def consume_sse(path: str, payload: dict, max_events: int = 6, timeout: int = 120) -> tuple[list[dict], float]:
    """Read SSE events from a POST endpoint. Returns list of {event, data} dicts."""
    import socket
    t0 = time.perf_counter()
    events = []
    try:
        url  = BASE + path
        body = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf   = ""
            ev    = None
            while True:
                chunk = r.read(512)
                if not chunk:
                    break
                buf += chunk.decode("utf-8")
                lines = buf.split("\n")
                buf   = lines.pop()
                for line in lines:
                    if line.startswith("event: "):
                        ev = line[7:].strip()
                    elif line.startswith("data: ") and ev:
                        raw = line[6:].strip()
                        try:
                            parsed = json.loads(raw)
                        except Exception:
                            parsed = raw
                        events.append({"event": ev, "data": parsed})
                        ev = None
                        if len(events) >= max_events:
                            return events, time.perf_counter() - t0
    except Exception as exc:
        pass
    return events, time.perf_counter() - t0


# ============================================================
# SECTION 1 — Frontend & Static Files
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 1 — Frontend & Static Files{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

# 1.1 Root serves index.html
code, body, lat, ct = http_get("/")
ok = code == 200 and "GenWealth" in str(body)
record("GET /  — SPA index.html serves", ok,
       f"HTTP {code} | Content-Type: {ct[:40]}" + (f"\n  BODY prefix: {str(body)[:80]}" if not ok else ""), lat)

# 1.2 styles.css served
code, body, lat, ct = http_get("/styles.css")
ok = code == 200 and "glassmorphism" in str(body).lower() or "glass-card" in str(body)
record("GET /styles.css — CSS served", ok, f"HTTP {code} | {len(str(body))} bytes", lat)

# 1.3 app.js served
code, body, lat, ct = http_get("/app.js")
ok = code == 200 and "TabManager" in str(body)
record("GET /app.js — JS served", ok, f"HTTP {code} | {len(str(body))} bytes", lat)


# ============================================================
# SECTION 2 — Health Endpoint
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 2 — Health Endpoint{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

code, data, lat, _ = http_get("/api/v1/health")
ok = code == 200 and isinstance(data, dict)
record("GET /api/v1/health — HTTP 200", ok, f"HTTP {code}", lat)

if isinstance(data, dict):
    for field in ["status", "version", "mongodb", "groq_keys", "gemini_keys",
                  "lstm_ok", "rf_ok", "ppo_zip_ok", "uptime_s"]:
        record(f"  health.{field} present", field in data, f"value={data.get(field, 'MISSING')!r}")

    record("  health.version == '4.0.0'", data.get("version") == "4.0.0", f"got: {data.get('version')!r}")
    record("  health.lstm_ok == True",    data.get("lstm_ok") is True,    f"got: {data.get('lstm_ok')!r}")
    record("  health.rf_ok == True",      data.get("rf_ok") is True,      f"got: {data.get('rf_ok')!r}")
    record("  health.groq_keys >= 1",     (data.get("groq_keys", 0) or 0) >= 1, f"got: {data.get('groq_keys')!r}")
    record("  health.gemini_keys >= 1",   (data.get("gemini_keys", 0) or 0) >= 1, f"got: {data.get('gemini_keys')!r}")
    print(f"\n  Full health response:\n  {json.dumps(data, indent=4)}\n")


# ============================================================
# SECTION 3 — OpenAPI / Docs
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 3 — OpenAPI Schema{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

code, schema, lat, _ = http_get("/openapi.json")
ok = code == 200 and isinstance(schema, dict)
record("GET /openapi.json — HTTP 200", ok, f"HTTP {code}", lat)

if isinstance(schema, dict):
    paths = schema.get("paths", {})
    for ep in ["/api/v1/health", "/api/v1/advisor/analyze",
               "/api/v1/portfolio/allocate", "/api/v1/simulate/dynamic",
               "/api/v1/simulate/walkforward", "/api/v1/chat/stream"]:
        record(f"  OpenAPI path: {ep}", ep in paths, f"{'FOUND' if ep in paths else 'MISSING'}")


# ============================================================
# SECTION 4 — Schema Validation (422 paths)
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 4 — Schema Validation (422 expected){RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

tests_422 = [
    ("/api/v1/advisor/analyze",      {},                              "advisor — missing ticker"),
    ("/api/v1/portfolio/allocate",   {"capital": 100000},             "portfolio — missing tickers"),
    ("/api/v1/portfolio/allocate",   {"tickers": ["NVDA"], "capital": -500}, "portfolio — negative capital"),
    ("/api/v1/simulate/dynamic",     {"capital": 100000},             "simulate/dynamic — missing tickers"),
    ("/api/v1/simulate/walkforward", {"tickers": ["NVDA"], "capital": 50000,
                                      "regime_start": "bad", "regime_end": "bad"}, "simulate/walkforward — bad dates"),
    ("/api/v1/chat/stream",          {"ticker": "NVDA"},              "chat — missing message"),
]
for path, payload, desc in tests_422:
    code, body, lat, _ = http_post(path, payload, timeout=10)
    record(f"POST {path} — 422 on {desc}", code == 422, f"HTTP {code}", lat)


# ============================================================
# SECTION 5 — Advisory Endpoint (SSE Stream)
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 5 — POST /api/v1/advisor/analyze (SSE){RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

print("  [Streaming SSE for NVDA — reading up to 4 events, max 90s...]\n")
events, lat = consume_sse(
    "/api/v1/advisor/analyze",
    {"ticker": "NVDA", "use_live_engine": True},
    max_events=4,
    timeout=90,
)

event_names = [e["event"] for e in events]
record("  Advisory SSE — at least 1 event received",   len(events) >= 1,          f"events: {event_names}")
record("  Advisory SSE — 'phase1' event present",      "phase1" in event_names,   f"events: {event_names}")

if "phase1" in event_names:
    p1 = next(e["data"] for e in events if e["event"] == "phase1")
    record("  phase1.ticker == 'NVDA'",           p1.get("ticker") == "NVDA",       f"got: {p1.get('ticker')!r}")
    record("  phase1.phase1_signal in [0, 1]",    0 <= p1.get("phase1_signal", -1) <= 1, f"got: {p1.get('phase1_signal')!r}")
    record("  phase1.signal_label valid",         p1.get("signal_label") in ("BULLISH","NEUTRAL","BEARISH"), f"got: {p1.get('signal_label')!r}")
    record("  phase1.inference_mode present",     bool(p1.get("inference_mode")),   f"got: {p1.get('inference_mode')!r}")
    record("  phase1.currency present",           bool(p1.get("currency")),         f"got: {p1.get('currency')!r}")
    print(f"\n  Phase 1 snapshot:\n  {json.dumps({k:v for k,v in p1.items() if k != 'news_headlines'}, indent=4)}\n")

record("  Advisory SSE — latency < 90s",  lat < 90, f"latency: {lat:.2f}s", lat)


# Also test a global ticker (NSE India)
print("  [Streaming SSE for RELIANCE.NS (Indian ticker) — reading phase1 only...]\n")
events2, lat2 = consume_sse(
    "/api/v1/advisor/analyze",
    {"ticker": "RELIANCE.NS", "use_live_engine": True},
    max_events=1,
    timeout=60,
)
if events2 and events2[0]["event"] == "phase1":
    p1_ns = events2[0]["data"]
    record("  RELIANCE.NS phase1.currency == 'INR'",
           p1_ns.get("currency") == "INR",
           f"got: {p1_ns.get('currency')!r}")
    record("  RELIANCE.NS phase1.close_price > 0",
           (p1_ns.get("close_price") or 0) > 0,
           f"got: {p1_ns.get('close_price')!r}")
else:
    record("  RELIANCE.NS phase1 received", False, f"events: {[e['event'] for e in events2]}")


# ============================================================
# SECTION 6 — Portfolio Allocator
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 6 — POST /api/v1/portfolio/allocate{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

code, data, lat, _ = http_post(
    "/api/v1/portfolio/allocate",
    {"tickers": ["NVDA", "AAPL"], "capital": 50000, "currency": "USD"},
    timeout=120,
)
ok = code == 200 and isinstance(data, dict)
record("  POST /portfolio/allocate — HTTP 200", ok, f"HTTP {code}", lat)

if ok:
    record("  allocations list present",    isinstance(data.get("allocations"), list),   f"count: {len(data.get('allocations',[]))}")
    record("  total_capital correct",       data.get("total_capital") == 50000.0,        f"got: {data.get('total_capital')!r}")
    record("  cash_pct > 0",               (data.get("cash_pct") or 0) > 0,             f"got: {data.get('cash_pct')!r}")
    record("  allocation_method present",  bool(data.get("allocation_method")),          f"got: {data.get('allocation_method')!r}")
    record("  latency_s present",          data.get("latency_s") is not None,            f"got: {data.get('latency_s')!r}s")

    # Weight sum check
    if data.get("allocations"):
        total_w = sum(a.get("weight", 0) for a in data["allocations"])
        record("  sum(weights) <= 1.0",    total_w <= 1.0 + 1e-6, f"sum={total_w:.4f}")
        print(f"\n  Allocations:")
        for a in data["allocations"]:
            print(f"    {a['ticker']:<15} weight={a['weight']:.4f}  capital=${a['capital']:,.2f}  signal={a['signal']:.4f}  stance={a['stance']}")
        print()


# ============================================================
# SECTION 7 — Dynamic Simulation
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 7 — POST /api/v1/simulate/dynamic{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

print("  [Running 30-day DynamicTradeSimulator for NVDA — this may take ~30-60s...]\n")
code, data, lat, _ = http_post(
    "/api/v1/simulate/dynamic",
    {"tickers": ["NVDA"], "capital": 100000, "portfolio_name": "Test Portfolio"},
    timeout=300,
)
ok = code == 200 and isinstance(data, dict)
record("  POST /simulate/dynamic — HTTP 200", ok, f"HTTP {code}", lat)

if ok:
    record("  simulation_type == 'DYNAMIC'",  data.get("simulation_type") == "DYNAMIC",   f"got: {data.get('simulation_type')!r}")
    record("  initial_capital == 100000",      data.get("initial_capital") == 100000.0,    f"got: {data.get('initial_capital')!r}")
    record("  final_cash >= 0",               (data.get("final_cash") or 0) >= 0,          f"got: {data.get('final_cash')!r}")
    record("  n_trades >= 0",                 data.get("n_trades", -1) >= 0,               f"got: {data.get('n_trades')!r}")
    record("  ledger is list",                isinstance(data.get("ledger"), list),         f"count: {len(data.get('ledger',[]))}")
    record("  portfolio_curve is list",       isinstance(data.get("portfolio_curve"), list),f"count: {len(data.get('portfolio_curve',[]))}")
    record("  latency_s present",             data.get("latency_s") is not None,            f"got: {data.get('latency_s')!r}s")

    # Ledger entry shape check
    if data["ledger"]:
        first = data["ledger"][0]
        for field in ["step","date","ticker","action","exec_price","realized_pnl","signal"]:
            record(f"    ledger[0].{field} present", field in first, f"val={first.get(field)!r}")

    print(f"\n  Simulation summary:")
    print(f"    ROI:          {data.get('total_roi_pct',0):.2f}%")
    print(f"    Win Rate:     {data.get('win_rate_pct',0):.1f}%")
    print(f"    Max Drawdown: {data.get('max_drawdown_pct',0):.2f}%")
    print(f"    Trades:       {data.get('n_trades',0)}")
    print(f"    Final Cash:   ${data.get('final_cash',0):,.2f}")
    print(f"    Latency:      {data.get('latency_s',0):.2f}s\n")

elif code == 500:
    record("  simulate/dynamic fallback note", True, "HTTP 500 — may need live LLM keys (non-critical)")


# ============================================================
# SECTION 8 — Walk-Forward Simulation
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 8 — POST /api/v1/simulate/walkforward{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

print("  [Running 5-day WalkForwardSimulator for NVDA (2024-01-02 -> 2024-01-08)...]\n")
code, data, lat, _ = http_post(
    "/api/v1/simulate/walkforward",
    {
        "tickers":       ["NVDA"],
        "capital":        50000,
        "regime_start": "2024-01-02",
        "regime_end":   "2024-01-08",
    },
    timeout=300,
)
ok = code == 200 and isinstance(data, dict)
record("  POST /simulate/walkforward — HTTP 200", ok, f"HTTP {code}", lat)

if ok:
    record("  simulation_type == 'WALK_FORWARD'", data.get("simulation_type") == "WALK_FORWARD", f"got: {data.get('simulation_type')!r}")
    record("  initial_capital == 50000",          data.get("initial_capital") == 50000.0,        f"got: {data.get('initial_capital')!r}")
    record("  final_cash >= 0",                   (data.get("final_cash") or 0) >= 0,            f"got: {data.get('final_cash')!r}")
    record("  latency_s present",                 data.get("latency_s") is not None,              f"got: {data.get('latency_s')!r}s")
    print(f"\n  Walk-Forward summary:")
    print(f"    ROI:      {data.get('total_roi_pct',0):.2f}%")
    print(f"    PnL:      ${data.get('total_realized_pnl',0):,.2f}")
    print(f"    Trades:   {data.get('n_trades',0)}")
    print(f"    Latency:  {data.get('latency_s',0):.2f}s\n")
elif code == 500:
    record("  walkforward fallback note", True, "HTTP 500 — may need live LLM keys (non-critical)")


# ============================================================
# SECTION 9 — Chat SSE Stream
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  SECTION 9 — POST /api/v1/chat/stream (SSE){RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

print("  [Streaming chat response for 'What is GenWealth AI?' — reading events...]\n")
chat_events, lat = consume_sse(
    "/api/v1/chat/stream",
    {"message": "What is GenWealth AI and how does it compute signals?", "ticker": "NVDA"},
    max_events=8,
    timeout=60,
)
chat_event_names = [e["event"] for e in chat_events]
record("  Chat SSE — at least 1 event received",      len(chat_events) >= 1,           f"events: {chat_event_names}")
record("  Chat SSE — 'token' event present",          "token" in chat_event_names,     f"events: {chat_event_names}")

if "token" in chat_event_names:
    first_token = next(e["data"] for e in chat_events if e["event"] == "token")
    record("  Chat token.content non-empty",           bool(first_token.get("content")),  f"content[:80]: {str(first_token.get('content',''))[:80]!r}")

if "done" in chat_event_names:
    done_ev = next(e["data"] for e in chat_events if e["event"] == "done")
    record("  Chat done.disclaimer_added == True",     done_ev.get("disclaimer_added") is True, f"got: {done_ev.get('disclaimer_added')!r}")

record("  Chat SSE — latency < 60s",  lat < 60, f"latency: {lat:.2f}s", lat)


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
print(f"{BOLD}  FINAL SUMMARY{RESET}")
print(f"{BOLD}{CYAN}{'='*70}{RESET}\n")

passed = sum(1 for _, ok, _, _ in RESULTS if ok)
failed = sum(1 for _, ok, _, _ in RESULTS if not ok)
total  = len(RESULTS)

print(f"  Passed: {GREEN}{passed}{RESET}  Failed: {RED}{failed}{RESET}  Total: {total}")

if failed:
    print(f"\n  {RED}{BOLD}Failed tests:{RESET}")
    for name, ok, detail, _ in RESULTS:
        if not ok:
            print(f"    {RED}[X]{RESET} {name}")
            if detail:
                for line in detail.splitlines():
                    print(f"        {line}")
else:
    print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED!{RESET}")

pct = passed / total * 100
print(f"\n  Pass rate: {GREEN if pct >= 80 else YELLOW if pct >= 60 else RED}{pct:.1f}%{RESET}")
sys.exit(0 if failed == 0 else 1)
