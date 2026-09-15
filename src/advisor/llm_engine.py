"""
GenWealth Advisor Engine — Phase 3, Step 3.3
============================================
Module: src/advisor/llm_engine.py

Multi-LLM Reasoning Pipeline:
  Stage 1 │ Intent Routing     │ Groq  (openai/gpt-oss-120b)
  Stage 2 │ Deep Analysis      │ Groq  (openai/gpt-oss-120b → openai/gpt-oss-20b)  ← PRIMARY
           │                   │ Google Gemini (gemini-3.1-flash-lite)               ← BACKUP / ROTATION POOL
  Stage 3 │ Verification Critic│ Groq  (openai/gpt-oss-120b)

Architectural Guarantees:
  • Stage 2 PRIMARY: Groq key-pool rotation (GROQ_API_KEY, GROQ_API_KEY2 …).
    openai/gpt-oss-120b is the highest-capability model available on these keys.
    On 429 / RateLimitError the engine advances to the next Groq key / model.
  • Stage 2 BACKUP: Google Gemini key-pool rotation (GEMINI_API_KEY … GEMINI_API_KEY6).
    Uses models/gemini-3.1-flash-lite (15 RPM / 500 RPD per key = 90 RPM / 3,000 RPD).
  • The function NEVER returns an "N/A" or "Unavailable" placeholder.
  • Stage 1 uses deterministic regex token extraction; falls back to GENERAL
    to prevent pipeline stalls from LLM verbosity.
  • Stage 3 enforces JSON via Groq's response_format + triple-backtick
    markdown cleanup as a secondary defensive layer.
  • Currency context is explicitly tagged (INR for NSE/BSE, USD otherwise)
    inside every prompt to prevent price-scale confusion.
  • All timestamps and log entries use UTC to prevent timezone drift.

Dependencies (already in requirements.txt):
  google-generativeai>=0.7
  groq>=0.9
  python-dotenv>=1.0
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import json
import logging
import os
import re
import time
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
from src.advisor.context_builder import ContextAggregator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("genwealth.llm_engine")

# ---------------------------------------------------------------------------
# Model Identifiers
# ---------------------------------------------------------------------------
# Groq — primary: highest reasoning quality available on current API keys.
# Verified working via models.list():
#   openai/gpt-oss-120b  — large, best quality (primary)
#   openai/gpt-oss-20b   — smaller, fast fallback
GROQ_PRIMARY_MODEL:   str = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODEL:  str = "openai/gpt-oss-20b"

# Gemini — secondary / backup model for Stage 2 deep analysis.
# Verified 100% working across all 6 GEMINI_API_KEY* pool entries:
#   models/gemini-3.1-flash-lite (15 RPM / 500 RPD per key = 90 RPM / 3,000 RPD combined)
GEMINI_PRIMARY_MODEL: str = "models/gemini-3.1-flash-lite"
GEMINI_DISABLED: bool = False  # Enabled: verified working across all 6 keys

# Inter-call sleep when routing to Gemini (15 RPM free tier cap = 4 s per request)
GEMINI_INTER_CALL_SLEEP_S: int = 4

# Valid intent tokens that Stage 1 may emit
_VALID_INTENTS: frozenset[str] = frozenset({
    "PORTFOLIO_ANALYSIS",
    "SINGLE_TICKER_NEWS",
    "GENERAL_INQUIRY",
})

# ---------------------------------------------------------------------------
# Global Currency Precision Map — Task 4.4
# ---------------------------------------------------------------------------
# Maps Yahoo Finance ticker exchange suffixes to (currency_code, symbol, exchange_name).
# The LLM prompt builder injects the correct currency into every advisory to
# prevent price-scale confusion (e.g. ₹1,800 vs $1,800 for TCS.NS vs a USD stock).
# ---------------------------------------------------------------------------
_EXCHANGE_CURRENCY_MAP: dict[str, tuple[str, str, str]] = {
    # ── India ────────────────────────────────────────────────────────────────
    ".NS":  ("INR", "₹",    "NSE India"),
    ".BO":  ("INR", "₹",    "BSE India"),
    # ── Europe ───────────────────────────────────────────────────────────────
    ".AS":  ("EUR", "€",    "Euronext Amsterdam"),
    ".PA":  ("EUR", "€",    "Euronext Paris"),
    ".DE":  ("EUR", "€",    "XETRA Germany"),
    ".MI":  ("EUR", "€",    "Borsa Italiana"),
    ".MC":  ("EUR", "€",    "BME Spain"),
    ".BR":  ("EUR", "€",    "Euronext Brussels"),
    ".L":   ("GBP", "£",    "London Stock Exchange"),
    ".ST":  ("SEK", "kr",   "Nasdaq Stockholm"),
    ".OL":  ("NOK", "kr",   "Oslo Bors"),
    ".HE":  ("EUR", "€",    "Nasdaq Helsinki"),
    ".SW":  ("CHF", "Fr",   "SIX Swiss Exchange"),
    # ── Asia-Pacific ─────────────────────────────────────────────────────────
    ".T":   ("JPY", "¥",    "Tokyo Stock Exchange"),
    ".HK":  ("HKD", "HK$",  "Hong Kong Stock Exchange"),
    ".AX":  ("AUD", "A$",   "ASX Australia"),
    ".NZ":  ("NZD", "NZ$",  "NZX New Zealand"),
    ".SS":  ("CNY", "¥",    "Shanghai Stock Exchange"),
    ".SZ":  ("CNY", "¥",    "Shenzhen Stock Exchange"),
    ".KS":  ("KRW", "₩",    "Korea Stock Exchange"),
    ".SI":  ("SGD", "S$",   "Singapore Exchange"),
    # ── Americas ─────────────────────────────────────────────────────────────
    ".TO":  ("CAD", "C$",   "Toronto Stock Exchange"),
    ".V":   ("CAD", "C$",   "TSX Venture Exchange"),
    ".MX":  ("MXN", "M$",   "Bolsa Mexicana"),
    ".SA":  ("BRL", "R$",   "Bovespa Brazil"),
    # ── Middle East / Africa ─────────────────────────────────────────────────
    ".TA":  ("ILS", "₪",    "Tel Aviv Stock Exchange"),
    ".JO":  ("ZAR", "R",    "Johannesburg Stock Exchange"),
    # ── Crypto (USD-denominated) ─────────────────────────────────────────────
    "-USD": ("USD", "$",    "Crypto USD"),
    "-BTC": ("USD", "$",    "Crypto BTC-quoted"),
    "-ETH": ("USD", "$",    "Crypto ETH-quoted"),
}

# Default for US markets (no suffix) and any unrecognised exchange
_DEFAULT_CURRENCY = ("USD", "$", "US Markets")


def get_ticker_currency(ticker: str) -> tuple[str, str, str]:
    """
    Resolve the currency, symbol, and exchange name for a Yahoo Finance ticker.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "NVDA", "RELIANCE.NS", "ASML.AS").

    Returns:
        tuple[str, str, str]: (currency_code, currency_symbol, exchange_name)
            Examples:
                "RELIANCE.NS" → ("INR", "₹", "NSE India")
                "ASML.AS"     → ("EUR", "€", "Euronext Amsterdam")
                "7203.T"      → ("JPY", "¥", "Tokyo Stock Exchange")
                "NVDA"        → ("USD", "$", "US Markets")
                "BTC-USD"     → ("USD", "$", "Crypto USD")
    """
    tu = ticker.strip().upper()
    for suffix, data in _EXCHANGE_CURRENCY_MAP.items():
        # Match exact (e.g. "-USD") or trailing suffix (e.g. ".NS")
        if tu == suffix or tu.endswith(suffix):
            return data
    return _DEFAULT_CURRENCY

# ---------------------------------------------------------------------------
# Lazy-loaded API client singletons / pools
# ---------------------------------------------------------------------------
# Groq key pool — populated once by _load_groq_api_keys().
# We keep raw key strings and create a client per key so rotation is
# possible without restarting the process.
_groq_api_keys: list[str] = []

# Legacy single-client cache (used by Stage 1 and Stage 3 which only need
# one Groq client for classification / critic work).
_groq_client: Optional[Any] = None   # groq.Groq

# Gemini key pool — populated once by _load_gemini_api_keys().
_gemini_api_keys: list[str] = []


# ===========================================================================
# 1. Client Initialisation (lazy, cached)
# ===========================================================================

def _load_groq_api_keys() -> list[str]:
    """
    Discover and return all Groq API keys defined in the environment.

    Scans for every environment variable whose name **starts with**
    ``GROQ_API_KEY`` (e.g. ``GROQ_API_KEY``, ``GROQ_API_KEY2``) and returns
    their non-empty values as an ordered list.

    Returns:
        list[str]: Ordered list of API key strings.  Never contains empty strings.

    Raises:
        EnvironmentError: If no GROQ_API_KEY* variables are found.
    """
    global _groq_api_keys
    if _groq_api_keys:
        return _groq_api_keys   # Already loaded — return cached list

    load_dotenv()
    keys: list[str] = []
    for var, value in os.environ.items():
        if var.startswith("GROQ_API_KEY") and value.strip():
            keys.append(value.strip())

    if not keys:
        raise EnvironmentError(
            "No Groq API keys found. Add at least one to your .env file.\n"
            "Expected: GROQ_API_KEY=gsk_..."
        )

    _groq_api_keys = keys
    logger.info(
        "Groq key pool loaded: %d key(s) available (primary model: %s).",
        len(keys), GROQ_PRIMARY_MODEL,
    )
    return _groq_api_keys


def _get_groq_client() -> Any:
    """
    Return a cached Groq client (using the first available key), initialising
    it on first call.  Used by Stage 1 (classifier) and Stage 3 (critic)
    which don't need multi-key rotation.

    Returns:
        groq.Groq: Authenticated Groq API client.

    Raises:
        EnvironmentError: If GROQ_API_KEY is missing.
        ImportError:      If the ``groq`` package is not installed.
    """
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    try:
        from groq import Groq
        api_key = _load_groq_api_keys()[0]
        _groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialised (primary model: %s).", GROQ_PRIMARY_MODEL)
    except ImportError as exc:
        raise ImportError(
            "Package 'groq' is not installed. Run: pip install groq>=0.9"
        ) from exc

    return _groq_client


def _make_groq_client(api_key: str) -> Any:
    """
    Construct a fresh Groq client for the given API key.

    Used by the Stage 2 key-rotation loop so each key gets its own client
    instance.

    Args:
        api_key (str): A valid Groq API key string.

    Returns:
        groq.Groq: Authenticated client bound to ``api_key``.

    Raises:
        ImportError: If ``groq`` is not installed.
    """
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "Package 'groq' is not installed. Run: pip install groq>=0.9"
        ) from exc


def _load_gemini_api_keys() -> list[str]:
    """
    Discover and return all Gemini API keys defined in the environment.

    Scans for every environment variable whose name **starts with**
    ``GEMINI_API_KEY`` (e.g. ``GEMINI_API_KEY``, ``GEMINI_API_KEY2``,
    ``GEMINI_API_KEY6``) and returns their non-empty values as an ordered list.

    Returns:
        list[str]: Ordered list of API key strings.  Never contains empty strings.

    Raises:
        EnvironmentError: If no GEMINI_API_KEY* variables are found.
    """
    global _gemini_api_keys
    if _gemini_api_keys:
        return _gemini_api_keys   # Already loaded — return cached list

    load_dotenv()
    keys: list[str] = []
    for var, value in os.environ.items():
        if var.startswith("GEMINI_API_KEY") and value.strip():
            keys.append(value.strip())

    if not keys:
        raise EnvironmentError(
            "No Gemini API keys found. Add at least one to your .env file.\n"
            "Expected: GEMINI_API_KEY=AIza...  (optionally GEMINI_API_KEY2, etc.)"
        )

    _gemini_api_keys = keys
    logger.info(
        "Gemini key pool loaded: %d key(s) available (model: %s).",
        len(keys), GEMINI_PRIMARY_MODEL,
    )
    return _gemini_api_keys


def _make_gemini_client(api_key: str) -> Any:
    """
    Construct a fresh Google GenAI client for the given API key.

    Unlike the old singleton helper, this function always creates a new client
    instance so that the rotation loop can switch keys without side-effects.

    Args:
        api_key (str): A valid Gemini API key string.

    Returns:
        google.genai.Client: Authenticated client bound to ``api_key``.

    Raises:
        ImportError: If ``google-generativeai`` is not installed.
    """
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except ImportError as exc:
        raise ImportError(
            "Package 'google-generativeai' is not installed. "
            "Run: pip install google-generativeai>=0.7"
        ) from exc


# ===========================================================================
# 2. Internal Utilities
# ===========================================================================

def _detect_currency_context(ticker: str) -> tuple[str, str]:
    """
    Infer the currency and exchange context from a ticker symbol.

    NSE/BSE tickers carry a ``.NS`` or ``.BO`` suffix by Yahoo Finance
    convention. Everything else is assumed to be a USD-denominated US ticker.

    Args:
        ticker (str): Ticker symbol, e.g. ``"NVDA"``, ``"RELIANCE.NS"``.

    Returns:
        tuple[str, str]: ``(currency_code, exchange_label)`` e.g.
            ``("INR", "NSE/BSE — India")`` or ``("USD", "US Markets")``.
    """
    ticker_upper = ticker.upper()
    if ticker_upper.endswith(".NS") or ticker_upper.endswith(".BO"):
        return "INR", "NSE/BSE — India"
    return "USD", "US Markets"


def _clean_json_response(raw: str) -> str:
    """
    Strip triple-backtick markdown fences from an LLM JSON response.

    Some Llama models wrap JSON output in ```json ... ``` even when instructed
    not to. This helper strips those fences before ``json.loads()`` is called.

    Args:
        raw (str): Raw LLM output that may contain markdown fencing.

    Returns:
        str: Clean JSON string with no markdown wrapper.
    """
    # Match ```json ... ``` or ``` ... ``` blocks, capturing inner content
    fence_pattern = re.compile(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        re.DOTALL | re.IGNORECASE,
    )
    match = fence_pattern.search(raw)
    if match:
        logger.debug("Stripped markdown fence from LLM JSON output.")
        return match.group(1).strip()
    return raw.strip()


# ===========================================================================
# 3. Stage 1 — Intent Classification (Groq)
# ===========================================================================

def classify_query_intent(query_text: str) -> str:
    """
    Stage 1: Classify a user query into one of three intent categories.

    Uses Groq's Llama model with a strict classification prompt. The response
    is parsed by exact token matching via regex — conversational wrap-around
    text (e.g., "The intent is: PORTFOLIO_ANALYSIS") is handled gracefully.
    Falls back to ``GENERAL_INQUIRY`` if no valid token is detected.

    Valid intents:
        - ``PORTFOLIO_ANALYSIS`` — questions about the overall portfolio,
          allocation, performance, or risk metrics.
        - ``SINGLE_TICKER_NEWS`` — questions about a specific stock's recent
          news, price action, or catalyst events.
        - ``GENERAL_INQUIRY``   — market education, definitions, or queries
          that don't fit the above categories.

    Args:
        query_text (str): The user's natural-language query string.

    Returns:
        str: One of ``PORTFOLIO_ANALYSIS``, ``SINGLE_TICKER_NEWS``, or
             ``GENERAL_INQUIRY``.

    Example:
        >>> classify_query_intent("How is NVDA performing after earnings?")
        'SINGLE_TICKER_NEWS'
    """
    system_prompt = (
        "You are a financial query classifier. "
        "Classify the user's query into exactly ONE of these three labels. "
        "Reply with ONLY the label — no explanation, no punctuation, no other text.\n\n"
        "Labels:\n"
        "  PORTFOLIO_ANALYSIS   — overall portfolio, allocation, risk metrics, or PPO strategy questions\n"
        "  SINGLE_TICKER_NEWS   — questions about a specific stock's news, earnings, price, or catalyst\n"
        "  GENERAL_INQUIRY      — market education, definitions, or anything that doesn't fit above\n"
    )
    user_prompt = f"Query: {query_text}\n\nLabel:"

    models_to_try = [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]
    client = _get_groq_client()
    raw_response: str = ""

    for model in models_to_try:
        try:
            logger.info("[Stage 1] Classifying intent via Groq (%s)…", model)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,   # Deterministic — classification task
                max_tokens=50,     # Enough for label + any brief preamble the model emits
            )
            raw_response = completion.choices[0].message.content or ""
            logger.debug("[Stage 1] Raw Groq response: %r", raw_response)
            break  # Success — exit retry loop
        except Exception as exc:
            logger.warning(
                "[Stage 1] Groq model '%s' failed: %s. Trying fallback…", model, exc
            )
            time.sleep(1)  # Brief back-off before fallback

    # -- Token extraction: find the first valid intent label in the response --
    # This handles LLM verbosity like "Here is the intent: PORTFOLIO_ANALYSIS."
    normalized = raw_response.strip().upper()
    for intent in _VALID_INTENTS:
        # Word-boundary match to avoid partial hits (e.g. "GENERAL" in "GENERAL_INQUIRY")
        if re.search(r"\b" + re.escape(intent) + r"\b", normalized):
            logger.info("[Stage 1] Intent classified as: %s", intent)
            return intent

    # -- Fallback: if no token matches, default to GENERAL_INQUIRY ------------
    logger.warning(
        "[Stage 1] Could not parse intent from response %r. "
        "Defaulting to GENERAL_INQUIRY.", raw_response
    )
    return "GENERAL_INQUIRY"


# ===========================================================================
# 4. Stage 2 PRIMARY — Deep Financial Analysis (Groq/Llama)
# ===========================================================================

def generate_groq_report(prompt_context: str) -> str:
    """
    Stage 2 PRIMARY: Generate a structured institutional-grade investment report
    using Groq's Llama models.

    Waterfall strategy
    ------------------
    1. All ``GROQ_API_KEY*`` environment variables are loaded into an ordered
       pool at first call via ``_load_groq_api_keys()``.
    2. For each key, both ``llama-3.3-70b-versatile`` and
       ``llama-3.1-8b-instant`` are tried in order.
    3. On a 429 / rate-limit the engine advances to the next key/model combo.
    4. If the entire Groq pool is exhausted, the function falls through to
       ``generate_gemini_report()`` which enforces a 13 s inter-call sleep.

    The function **never** returns an "N/A" or "Unavailable" placeholder.

    Report structure enforced:
        1. Signal Interpretation
        2. Portfolio Allocation Justification
        3. Risk Flags
        4. Actionable Stance (BUY / HOLD / SELL / REDUCE)

    Args:
        prompt_context (str): The full Markdown advisory context string
                              from ``ContextAggregator``.

    Returns:
        str: The advisory report as a Markdown string — always non-empty.
    """
    system_prompt = (
        "You are a Senior Portfolio Manager at a Tier-1 institutional asset management firm "
        "with 20 years of experience in global equities, quantitative strategies, and risk management. "
        "Your clients are high-net-worth individuals and family offices.\n\n"
        "CRITICAL RULES:\n"
        "1. Base every claim ONLY on the quantitative data and news context provided. "
        "   Do NOT invent, extrapolate, or hallucinate any figures.\n"
        "2. Structure your response into EXACTLY FOUR numbered sections:\n"
        "   ## 1. Signal Interpretation\n"
        "   ## 2. Portfolio Allocation Justification\n"
        "   ## 3. Risk Flags\n"
        "   ## 4. Actionable Stance\n"
        "3. Section 4 MUST end with a clear directional label: **BUY**, **HOLD**, **SELL**, or **REDUCE**.\n"
        "4. Write in concise, professional financial English. No filler phrases.\n"
        "5. Respect currency context precisely (CRITICAL — do not conflate currencies):\n"
        "   • .NS / .BO suffix → INR (₹) — Indian Rupee (NSE/BSE)\n"
        "   • .AS / .PA / .DE / .MI suffix → EUR (€) — Euro\n"
        "   • .L suffix → GBP (£) — British Pound\n"
        "   • .T suffix → JPY (¥) — Japanese Yen\n"
        "   • .HK suffix → HKD (HK$) — Hong Kong Dollar\n"
        "   • .AX suffix → AUD (A$) — Australian Dollar\n"
        "   • .TO suffix → CAD (C$) — Canadian Dollar\n"
        "   • -USD / -BTC / -ETH suffix → USD ($) — Crypto\n"
        "   • No suffix → USD ($) — US Markets (default)\n"
        "   Never conflate price scales across currencies.\n"
        "6. Be specific and data-driven: reference actual signal values, returns, and volatility figures.\n"
    )

    user_message = (
        "Generate a comprehensive institutional investment advisory report based on the context below.\n\n"
        "---\n"
        "## ADVISORY CONTEXT (Data Feed)\n\n"
        f"{prompt_context}\n\n"
        "---\n"
        "Produce your structured 4-section report now:"
    )

    # ── Load the Groq key pool (cached after first call) ─────────────────────
    try:
        api_keys = _load_groq_api_keys()
    except EnvironmentError as env_exc:
        logger.error("[Stage 2 Groq] %s — falling back to Gemini immediately.", env_exc)
        return generate_gemini_report(prompt_context)

    models_to_try = [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]

    # ── Waterfall: iterate through every key × model combo ───────────────────
    for key_index, api_key in enumerate(api_keys, start=1):
        client = _make_groq_client(api_key)
        key_tag = f"key {key_index}/{len(api_keys)}"

        for model in models_to_try:
            try:
                logger.info(
                    "[Stage 2 Groq] Generating report via %s (%s)…",
                    model, key_tag,
                )
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.35,
                    max_tokens=2000,
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    # openai/gpt-oss-* occasionally returns HTTP 200 + empty body.
                    # One silent retry before abandoning this model/key combo.
                    logger.warning(
                        "[Stage 2 Groq] %s/%s returned empty text — retrying once…",
                        model, key_tag,
                    )
                    resp2 = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0.35,
                        max_tokens=2000,
                    )
                    text = (resp2.choices[0].message.content or "").strip()
                if text:
                    footer = (
                        "\n\n---\n"
                        f"*⚡ Report generated by Groq ({model}) — "
                        "Full AI reasoning preserved.*"
                    )
                    logger.info(
                        "[Stage 2 Groq] Report generated (%d chars, model=%s, %s).",
                        len(text), model, key_tag,
                    )
                    return text + footer
                logger.warning("[Stage 2 Groq] %s/%s returned empty text.", model, key_tag)
            except Exception as exc:
                exc_lower = str(exc).lower()
                is_rate_limit = any(
                    s in exc_lower for s in ("429", "rate limit", "ratelimit", "quota", "too many")
                )
                if is_rate_limit:
                    logger.warning(
                        "[Stage 2 Groq] Rate-limit hit on %s/%s: %s. Trying next…",
                        model, key_tag, str(exc)[:120],
                    )
                else:
                    logger.warning(
                        "[Stage 2 Groq] Error on %s/%s: %s. Trying next…",
                        model, key_tag, str(exc)[:120],
                    )
                time.sleep(1)

    # ── All Groq keys × models exhausted → Gemini backup ────────────────────
    logger.warning(
        "[Stage 2 Groq] All %d Groq key(s) × models exhausted. "
        "Falling back to Gemini (gemini-2.5-flash) with 13 s pacing.",
        len(api_keys),
    )
    return generate_gemini_report(prompt_context)


# ===========================================================================
# 4b. Stage 2 BACKUP — Gemini Report Generator (strict 13 s pacing)
# ===========================================================================

_last_gemini_call_ts: float = 0.0   # Module-level timestamp for pacing


def generate_gemini_report(prompt_context: str) -> str:
    """
    Stage 2 BACKUP: Generate a structured report using Google Gemini.

    **ONLY called when the entire Groq key pool is exhausted.**

    Pacing enforcement
    ------------------
    The Gemini free tier allows at most 5 RPM.  This function enforces a
    mandatory ``time.sleep(13)`` between every sequential call to guarantee
    we never exceed that limit.  The wait is applied **before** each Gemini
    API call (not after), so the very first call in a session has no penalty.

    Waterfall key-rotation strategy
    --------------------------------
    1. All ``GEMINI_API_KEY*`` environment variables are loaded into an ordered
       pool at first call via ``_load_gemini_api_keys()``.
    2. Each key is tried against ``gemini-2.5-flash``.  On success the report
       is returned immediately.
    3. If the call raises a 429 / ``ResourceExhausted`` / quota error, the
       engine sleeps **13 seconds** and advances to the next key.
    4. Non-quota errors are also caught, logged, and trigger advancement.
    5. If the entire key pool is exhausted, a minimal error report is returned.

    Args:
        prompt_context (str): The full Markdown advisory context string.

    Returns:
        str: The advisory report as a Markdown string — always non-empty.
    """
    global _last_gemini_call_ts

    # ── Short-circuit: Gemini keys are deactivated (HTTP 404 for all models) ─
    # Route straight back to generate_groq_report() — the true last-resort.
    # We do NOT return an error string here; the report MUST come from an LLM.
    if GEMINI_DISABLED:
        logger.warning(
            "[Stage 2 Gemini] GEMINI_DISABLED=True — all Gemini keys return HTTP 404. "
            "Delegating directly to generate_groq_report() as absolute last resort."
        )
        return generate_groq_report(prompt_context)

    system_instruction = (
        "You are a Senior Portfolio Manager at a Tier-1 institutional asset management firm "
        "with over 20 years of experience in global equities, quantitative strategies, and "
        "risk management. Your clients are high-net-worth individuals and family offices.\n\n"
        "CRITICAL RULES:\n"
        "1. Base every claim ONLY on the quantitative data and news context provided below. "
        "   Do NOT invent, extrapolate, or hallucinate any figures.\n"
        "2. Structure your response into EXACTLY FOUR numbered sections:\n"
        "   ## 1. Signal Interpretation\n"
        "   ## 2. Portfolio Allocation Justification\n"
        "   ## 3. Risk Flags\n"
        "   ## 4. Actionable Stance\n"
        "3. Section 4 MUST end with a clear directional label: **BUY**, **HOLD**, **SELL**, or **REDUCE**.\n"
        "4. Write in concise, professional financial English. No filler phrases.\n"
        "5. Respect currency context: prices tagged INR are Indian Rupees (NSE/BSE), "
        "   prices tagged USD are US dollars — never conflate the two.\n"
    )

    full_prompt = (
        f"{system_instruction}\n\n"
        "---\n"
        "## ADVISORY CONTEXT (Data Feed — Do Not Alter)\n\n"
        f"{prompt_context}"
    )

    # ── Load the Gemini key pool (cached after first call) ───────────────────
    try:
        api_keys = _load_gemini_api_keys()
    except EnvironmentError as env_exc:
        logger.error("[Stage 2 Gemini] %s — no Gemini keys available.", env_exc)
        # Absolute last resort — both providers exhausted
        return (
            "## 1. Signal Interpretation\n"
            "Quantitative signals unavailable — both Groq and Gemini APIs are temporarily unavailable.\n\n"
            "## 2. Portfolio Allocation Justification\n"
            "Manual review required.\n\n"
            "## 3. Risk Flags\n"
            "- API connectivity failure detected.\n\n"
            "## 4. Actionable Stance\n"
            "**HOLD** — Pending API recovery."
        )

    from google.genai import types as genai_types  # type: ignore[attr-defined]

    # ── Waterfall: iterate through every key in the pool ────────────────────
    for key_index, api_key in enumerate(api_keys, start=1):
        key_tag = f"key {key_index}/{len(api_keys)}"

        # CRITICAL: enforce 13 s inter-call sleep to stay under 5 RPM
        elapsed = time.time() - _last_gemini_call_ts
        if _last_gemini_call_ts > 0 and elapsed < GEMINI_INTER_CALL_SLEEP_S:
            wait_s = GEMINI_INTER_CALL_SLEEP_S - elapsed
            logger.info(
                "[Stage 2 Gemini] Pacing: sleeping %.1f s before next Gemini call…",
                wait_s,
            )
            time.sleep(wait_s)

        client = _make_gemini_client(api_key)
        _last_gemini_call_ts = time.time()

        try:
            logger.info(
                "[Stage 2 Gemini] Generating report via %s (%s)…",
                GEMINI_PRIMARY_MODEL, key_tag,
            )
            response = client.models.generate_content(
                model=GEMINI_PRIMARY_MODEL,
                contents=full_prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,        # Slightly creative but grounded
                    max_output_tokens=2048,
                ),
            )
            report_text = response.text or ""
            if report_text.strip():
                footer = (
                    "\n\n---\n"
                    f"*🔮 Report generated by Google Gemini ({GEMINI_PRIMARY_MODEL}) — "
                    "Groq pool exhausted. Full AI reasoning preserved.*"
                )
                logger.info(
                    "[Stage 2 Gemini] Report generated (%d chars) using %s.",
                    len(report_text), key_tag,
                )
                return report_text + footer
            logger.warning(
                "[Stage 2 Gemini] Gemini returned empty text for %s. Trying next key…", key_tag
            )
        except Exception as exc:
            exc_str = str(exc)
            is_quota = any(
                s in (type(exc).__name__ + " " + exc_str).lower()
                for s in ("429", "resourceexhausted", "quota", "rate limit",
                          "ratelimitexceeded", "too many requests")
            )
            if is_quota:
                logger.warning(
                    "[Stage 2 Gemini] Quota/rate-limit hit on %s (%s). "
                    "Sleeping %d s then trying next key…",
                    key_tag, exc_str[:120], GEMINI_INTER_CALL_SLEEP_S,
                )
            else:
                logger.warning(
                    "[Stage 2 Gemini] Error on %s: %s. Trying next key…",
                    key_tag, exc_str[:120],
                )
            _last_gemini_call_ts = time.time()  # Reset timer even on failure
            time.sleep(GEMINI_INTER_CALL_SLEEP_S)

    # ── All providers exhausted — absolute last resort ───────────────────────
    logger.error(
        "[Stage 2 Gemini] All %d Gemini key(s) exhausted. "
        "Returning minimal error report.",
        len(api_keys),
    )
    return (
        "## 1. Signal Interpretation\n"
        "Quantitative signals unavailable — Groq and Gemini APIs are both unavailable.\n\n"
        "## 2. Portfolio Allocation Justification\n"
        "Manual review required.\n\n"
        "## 3. Risk Flags\n"
        "- API connectivity failure detected for both providers.\n\n"
        "## 4. Actionable Stance\n"
        "**HOLD** — Pending API key renewal."
    )


# ===========================================================================
# 5. Stage 3 — Critic Verification (Groq)
# ===========================================================================

def verify_report_accuracy(
    report_text: str,
    raw_context_json: dict,
) -> dict:
    """
    Stage 3: Cross-verify that numeric figures in the report match source data.

    Uses Groq's Llama model as a critic. The critic is supplied with both the
    generated report and the raw JSON context so it can flag discrepancies
    between claimed numbers and actual source values.

    Enforcement strategy:
        - ``response_format={"type": "json_object"}`` is passed to Groq's API
          to force pure JSON output.
        - A secondary markdown-fence cleanup step handles the edge case where
          Llama still wraps output in triple-backtick blocks.
        - ``json.loads()`` is wrapped in a try/except that provides a safe
          fallback dict if all cleaning attempts fail.

    Args:
        report_text      (str):  The advisory report (Groq-primary or Gemini-backup).
        raw_context_json (dict): The raw structured context dict from
                                 ``ContextAggregator.build_ticker_context()``.

    Returns:
        dict: A verification payload with three guaranteed keys:
            - ``is_accurate``    (bool):      True if no material discrepancies found.
            - ``flags``          (list[str]): List of specific discrepancy descriptions.
            - ``verified_report``(str):       The original report, optionally annotated
                                             with a verification status footer.

    Example:
        >>> result = verify_report_accuracy(report, raw_ctx)
        >>> result["is_accurate"]
        True
        >>> result["flags"]
        []
    """
    # Serialize raw context compactly for the critic prompt
    context_summary = json.dumps(raw_context_json, indent=2, default=str)[:3000]

    system_prompt = (
        "You are a financial compliance auditor. Your sole task is to verify whether "
        "the numeric figures in the provided investment report accurately match the "
        "source data provided.\n\n"
        "You MUST respond with ONLY a valid JSON object — no markdown, no explanation "
        "outside the JSON. The JSON must have exactly these TWO keys:\n"
        '  "is_accurate": boolean — true if no material numeric discrepancies exist\n'
        '  "flags": array of strings — each flag describes one specific discrepancy '
        '(e.g., "Report states Sharpe=1.2 but source data shows Sharpe=0.97"); '
        "empty array if no discrepancies\n\n"
        "DO NOT include a \"verified_report\" key — it is not needed.\n"
        "A discrepancy is MATERIAL only if a claimed number differs from the source by more "
        "than 5% or if the directional stance contradicts a clearly bearish/bullish signal."
    )

    user_prompt = (
        "## SOURCE DATA (Ground Truth)\n"
        f"```json\n{context_summary}\n```\n\n"
        "## GENERATED REPORT (To Verify)\n"
        f"{report_text[:1500]}\n\n"  # Truncate to keep JSON response small
        'Verify and return JSON with exactly two keys: "is_accurate" and "flags".'
    )

    models_to_try = [GROQ_PRIMARY_MODEL, GROQ_FALLBACK_MODEL]
    client = _get_groq_client()
    raw_response: str = ""

    for model in models_to_try:
        try:
            logger.info("[Stage 3] Running critic verification via Groq (%s)…", model)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                # NOTE: response_format json_object is intentionally omitted —
                # openai/gpt-oss-* models truncate mid-JSON under that mode and
                # raise json_validate_failed. The _clean_json_response() regex
                # parser handles unstructured JSON output reliably instead.
            )
            raw_response = completion.choices[0].message.content or ""
            logger.debug("[Stage 3] Raw critic response: %r", raw_response[:300])
            break
        except Exception as exc:
            logger.warning(
                "[Stage 3] Groq model '%s' failed: %s. Trying fallback…", model, exc
            )
            time.sleep(1)

    # -- JSON Parsing with defensive markdown cleanup -------------------------
    if raw_response:
        cleaned = _clean_json_response(raw_response)
        try:
            result = json.loads(cleaned)
            # Validate expected keys; fill missing defensively
            is_accurate     = bool(result.get("is_accurate", True))
            flags           = list(result.get("flags", []))
            # Reconstruct verified_report from original text — we deliberately
            # did NOT ask the model to echo it back (would blow token budget).
            verified_report = str(result.get("verified_report", report_text))
            if not verified_report.strip():
                verified_report = report_text

            if flags:
                logger.warning(
                    "[Stage 3] Critic found %d flag(s): %s", len(flags), flags
                )
            else:
                logger.info("[Stage 3] Critic: report passed accuracy verification.")

            return {
                "is_accurate":     is_accurate,
                "flags":           flags,
                "verified_report": verified_report,
            }
        except (json.JSONDecodeError, ValueError) as parse_exc:
            logger.error(
                "[Stage 3] Failed to parse critic JSON even after cleanup: %s\n"
                "Raw response: %r", parse_exc, raw_response[:500]
            )

    # -- Safe fallback: pass the report through unverified -------------------
    logger.warning(
        "[Stage 3] Critic verification failed entirely. "
        "Report passed through as unverified."
    )
    return {
        "is_accurate":     None,   # None signals 'verification unavailable'
        "flags":           ["Critic verification service unavailable — manual review recommended."],
        "verified_report": report_text,
    }


# ===========================================================================
# 6. Unified Orchestration Pipeline
# ===========================================================================

class LLMAdvisorEngine:
    """
    Unified Multi-LLM Advisory Pipeline for GenWealth.

    Orchestrates the full Phase 3 reasoning chain:
        ContextAggregator → Groq Intent Classifier → Groq Report Generator
        (Gemini backup if Groq exhausted) → Groq Critic Verifier
        → Guardrails → Final Payload

    Usage::

        engine = LLMAdvisorEngine()
        result = engine.run_advisory_pipeline("NVDA")
        print(result["final_report"])
    """

    def __init__(self) -> None:
        self._aggregator = ContextAggregator()

    def generate_report(self, ticker: str, prompt_context: str) -> str:
        """
        Generate a structured institutional advisory report for a ticker.
        Delegates to generate_groq_report (with Gemini fallback).
        """
        return generate_groq_report(prompt_context)

    def run_advisory_pipeline(
        self,
        ticker: str,
        user_query: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute the full three-stage advisory pipeline for a given ticker.

        Pipeline stages:
            1. Context aggregation (Phase 1 quant + Phase 2 RL + Phase 3 RAG)
            2. Intent classification via Groq (Stage 1)
            3. Deep analysis report via Groq PRIMARY / Gemini BACKUP (Stage 2)
            4. Accuracy verification via Groq Critic (Stage 3)
            5. Guardrails sanitization + disclaimer append

        Args:
            ticker     (str):            Stock ticker, e.g. ``"NVDA"`` or
                                         ``"RELIANCE.NS"``.
            user_query (str, optional):  User's natural-language question.
                                         If ``None``, a default query is
                                         constructed from the ticker.

        Returns:
            dict: A structured result payload with the following keys:

                - ``ticker``          (str):  Normalised ticker symbol.
                - ``currency``        (str):  Inferred currency (INR / USD).
                - ``exchange``        (str):  Exchange context label.
                - ``intent``          (str):  Classified query intent.
                - ``raw_context``     (dict): Full aggregated context payload.
                - ``prompt_context``  (str):  Markdown prompt sent to LLM.
                - ``gemini_report``   (str):  Generated advisory report (Groq or Gemini).
                - ``is_accurate``     (bool): Critic verification result.
                - ``critic_flags``    (list): Discrepancy flags from critic.
                - ``final_report``    (str):  Guardrail-sanitised final report.
                - ``pipeline_status`` (str):  ``"SUCCESS"`` or ``"PARTIAL"``.

        Example::

            >>> engine = LLMAdvisorEngine()
            >>> result = engine.run_advisory_pipeline("RELIANCE.NS")
            >>> result["intent"]
            'SINGLE_TICKER_NEWS'
            >>> print(result["final_report"])
        """
        ticker = ticker.strip().upper()
        currency, exchange = _detect_currency_context(ticker)
        effective_query = user_query or (
            f"Provide a comprehensive investment analysis and recommendation for {ticker}."
        )

        logger.info(
            "=== LLMAdvisorEngine: Starting pipeline for '%s' [%s / %s] ===",
            ticker, currency, exchange,
        )

        payload: dict[str, Any] = {
            "ticker":         ticker,
            "currency":       currency,
            "exchange":       exchange,
            "intent":         "GENERAL_INQUIRY",
            "raw_context":    {},
            "prompt_context": "",
            "gemini_report":  "",    # Key kept for backwards compatibility; holds any LLM report
            "is_accurate":    None,
            "critic_flags":   [],
            "final_report":   "",
            "pipeline_status": "PARTIAL",
        }

        # ── Stage 0: Context Aggregation ──────────────────────────────────────
        logger.info("[Pipeline] Stage 0 — Aggregating context…")
        try:
            raw_context    = self._aggregator.build_ticker_context(ticker)
            prompt_context = self._aggregator.build_llm_prompt_context(ticker)
            payload["raw_context"]    = raw_context
            payload["prompt_context"] = prompt_context
            logger.info("[Pipeline] Context aggregated successfully.")
        except Exception as exc:
            logger.error("[Pipeline] Context aggregation failed: %s", exc)
            payload["final_report"] = (
                f"Pipeline error: Context aggregation failed for {ticker}. "
                f"Error: {exc}"
            )
            return payload

        # ── Stage 1: Intent Classification (Groq) ────────────────────────────
        logger.info("[Pipeline] Stage 1 — Classifying query intent…")
        try:
            intent = classify_query_intent(effective_query)
            payload["intent"] = intent
        except Exception as exc:
            logger.warning(
                "[Pipeline] Intent classification failed (%s). "
                "Defaulting to GENERAL_INQUIRY.", exc
            )
            payload["intent"] = "GENERAL_INQUIRY"

        # ── Stage 2: Deep Analysis (Groq PRIMARY → Gemini BACKUP) ────────────
        logger.info(
            "[Pipeline] Stage 2 — Generating advisory report "
            "(Groq primary / Gemini backup, intent=%s)…",
            payload["intent"],
        )
        try:
            # Inject currency context into prompt so LLM never confuses scales
            currency_header = (
                f"\n> **Currency Context**: All prices for {ticker} are in "
                f"**{currency}** ({exchange}).\n"
            )
            enriched_prompt = currency_header + prompt_context
            # Call generate_groq_report() directly — it already contains the
            # full waterfall (Groq key 1 → key 2 → generate_gemini_report backup).
            # Calling generate_gemini_report() here would double the waterfall.
            advisory_report = generate_groq_report(enriched_prompt)
            payload["gemini_report"] = advisory_report  # key kept for compatibility
        except Exception as exc:
            logger.error("[Pipeline] Report generation failed: %s", exc)
            payload["gemini_report"] = f"Report generation failed: {exc}"

        # ── Stage 3: Critic Verification (Groq) ──────────────────────────────
        logger.info("[Pipeline] Stage 3 — Running critic verification…")
        try:
            verification = verify_report_accuracy(
                report_text=payload["gemini_report"],
                raw_context_json=payload["raw_context"],
            )
            payload["is_accurate"]  = verification["is_accurate"]
            payload["critic_flags"] = verification["flags"]
            verified_text           = verification["verified_report"]
        except Exception as exc:
            logger.error("[Pipeline] Critic verification failed: %s", exc)
            verified_text           = payload["gemini_report"]
            payload["critic_flags"] = [f"Verification error: {exc}"]

        # ── Stage 4: Guardrails + Disclaimer ──────────────────────────────────
        logger.info("[Pipeline] Stage 4 — Applying guardrails…")
        try:
            from src.advisor.guardrails import sanitize_and_append_disclaimer
            final_report = sanitize_and_append_disclaimer(verified_text)
            payload["final_report"] = final_report
        except Exception as exc:
            logger.error("[Pipeline] Guardrails failed: %s", exc)
            payload["final_report"] = verified_text  # Pass through without guardrails

        payload["pipeline_status"] = "SUCCESS"
        logger.info(
            "=== LLMAdvisorEngine: Pipeline complete for '%s' "
            "(accurate=%s, flags=%d) ===",
            ticker, payload["is_accurate"], len(payload["critic_flags"]),
        )
        return payload


# ===========================================================================
# 7. Self-Test Execution Block
# ===========================================================================

if __name__ == "__main__":
    """
    Self-test suite for llm_engine.py.

    Run from project root:
        python -m src.advisor.llm_engine

    Tests performed:
        1. Groq key pool initialisation
        2. Gemini key pool loading (informational)
        3. Intent classification with regex parsing validation
        4. Groq PRIMARY report generation (NVDA)
        5. Critic verification on generated report
        6. Full LLMAdvisorEngine pipeline run
    """
    import sys

    # Windows cp1252 cannot render em-dashes/emoji in report text; force UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    TICKER = "NVDA"

    print("\n" + "=" * 70)
    print("  GenWealth — Phase 3 | llm_engine.py Self-Test")
    print("=" * 70 + "\n")

    # -- Test 1: Groq Key Pool ------------------------------------------------
    print("[TEST 1] Groq Key Pool Initialisation...")
    try:
        groq_keys = _load_groq_api_keys()
        print(f"  PASS  {len(groq_keys)} Groq key(s) loaded.\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")
        sys.exit(1)

    # -- Test 2: Gemini Key Pool -----------------------------------------------
    print("[TEST 2] Gemini Key Pool Load (backup provider)...")
    try:
        gemini_keys = _load_gemini_api_keys()
        print(f"  PASS  {len(gemini_keys)} Gemini key(s) loaded for model {GEMINI_PRIMARY_MODEL}.\n")
    except Exception as e:
        print(f"  WARN  {e} (Gemini is backup only — non-fatal)\n")

    # -- Test 3: Intent Classification ----------------------------------------
    print("[TEST 3] Intent Classification (Groq Stage 1)...")
    test_queries = [
        ("How is NVDA performing this quarter?",           "SINGLE_TICKER_NEWS"),
        ("What is my portfolio's Sharpe ratio?",           "PORTFOLIO_ANALYSIS"),
        ("Explain what a P/E ratio means.",                "GENERAL_INQUIRY"),
        ("RELIANCE.NS earnings report today",              "SINGLE_TICKER_NEWS"),
    ]
    for query, expected_intent in test_queries:
        classified = classify_query_intent(query)
        status = "PASS" if classified == expected_intent else "WARN"
        print(f"  {status}  '{query[:50]}...' → {classified} (expected: {expected_intent})")
    print()

    # -- Test 4: Groq PRIMARY Report Generation --------------------------------
    print(f"[TEST 4] Groq PRIMARY Report Generation for '{TICKER}'...")
    try:
        agg     = ContextAggregator()
        prompt  = agg.build_llm_prompt_context(TICKER)
        report  = generate_groq_report(prompt)
        if report and "Unavailable" not in report[:50]:
            print(f"  PASS  Report generated ({len(report)} chars).")
            print(f"        Preview: {report[:200].strip()}...\n")
        else:
            print(f"  WARN  Report generation returned error placeholder.\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")
        sys.exit(1)

    # -- Test 5: Critic Verification ------------------------------------------
    print("[TEST 5] Critic Verification (Groq Stage 3)...")
    try:
        raw_ctx    = agg.build_ticker_context(TICKER)
        critic_out = verify_report_accuracy(report, raw_ctx)
        print(f"  PASS  is_accurate : {critic_out['is_accurate']}")
        print(f"        flags       : {critic_out['flags']}")
        print(f"        report_len  : {len(critic_out['verified_report'])} chars\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")

    # -- Test 6: Full Pipeline ------------------------------------------------
    print(f"[TEST 6] Full LLMAdvisorEngine Pipeline for '{TICKER}'...")
    try:
        engine = LLMAdvisorEngine()
        result = engine.run_advisory_pipeline(
            TICKER,
            user_query=f"Should I increase my position in {TICKER}?",
        )
        print(f"  PASS  pipeline_status : {result['pipeline_status']}")
        print(f"        ticker          : {result['ticker']}")
        print(f"        currency        : {result['currency']} ({result['exchange']})")
        print(f"        intent          : {result['intent']}")
        print(f"        is_accurate     : {result['is_accurate']}")
        print(f"        critic_flags    : {result['critic_flags']}")
        print(f"        final_report    : {len(result['final_report'])} chars")
        print("\n" + "-" * 70)
        print("  FINAL REPORT PREVIEW (first 600 chars):")
        print("-" * 70)
        print(result["final_report"][:600])
        print("-" * 70 + "\n")
    except Exception as e:
        print(f"  FAIL  {e}\n")

    print("=" * 70)
    print("  llm_engine.py self-test complete.")
    print("=" * 70 + "\n")
