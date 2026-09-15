"""
GenWealth — Interactive RAG & Global Equity Chatbot Router (SSE)
=================================================================
File: app/routers/chat.py

POST /api/v1/chat/stream
    Conversational financial Q&A grounded in live market intelligence,
    quantitative models, and the MongoDB RAG knowledge base.
    Streams LLM response tokens as Server-Sent Events.

    Features:
    - Automatically extracts stock tickers and company names from user queries
      (e.g., Paytm -> PAYTM.NS, Toyota -> 7203.T, Alibaba -> BABA, Shell -> SHEL.L).
    - Fetches real-time price & quantitative signals (LSTM + RF ensemble) for any global stock.
    - Grounded in high-relevance news and institutional analysis.
    - Provides articulate, professional analysis on any company or stock without refusing.
    - Compact regulatory compliance footer.
"""

import json
import logging
import re
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas.request_schemas import ChatMessage
from app.schemas.response_schemas import ChatStreamChunk

logger = logging.getLogger("genwealth.api.chat")

router = APIRouter(prefix="/api/v1/chat", tags=["Chatbot"])

# ── Comprehensive Company Name to Ticker Map ──────────────────────────────────
_COMPANY_TICKER_MAP = {
    # Indian stocks (NSE/BSE)
    "paytm": "PAYTM.NS",
    "one97": "PAYTM.NS",
    "one 97": "PAYTM.NS",
    "reliance": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "tata motors": "TATAMOTORS.NS",
    "tatamotors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "tatasteel": "TATASTEEL.NS",
    "hdfc": "HDFCBANK.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "infosys": "INFY.NS",
    "infy": "INFY.NS",
    "wipro": "WIPRO.NS",
    "zomato": "ZOMATO.NS",
    "itc": "ITC.NS",
    "sbi": "SBIN.NS",
    "state bank of india": "SBIN.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "maruti": "MARUTI.NS",
    "sun pharma": "SUNPHARMA.NS",

    # Chinese / Asian Equities & ADRs
    "alibaba": "BABA",
    "baba": "BABA",
    "tencent": "TCEHY",
    "pdd": "PDD",
    "pinduoduo": "PDD",
    "baidu": "BIDU",
    "nio": "NIO",
    "byd": "BYDDY",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "jd": "JD",
    "jd.com": "JD",

    # Japanese Equities (Tokyo SE)
    "toyota": "7203.T",
    "sony": "6758.T",
    "softbank": "9984.T",
    "nintendo": "7974.T",
    "honda": "7267.T",
    "mitsubishi": "8058.T",
    "tokyo electron": "8035.T",
    "fast retailing": "9983.T",

    # UK & European Equities
    "shell": "SHEL.L",
    "astrazeneca": "AZN.L",
    "hsbc": "HSBA.L",
    "bp": "BP.L",
    "asml": "ASML",
    "sap": "SAP",
    "novo nordisk": "NVO",
    "rio tinto": "RIO.L",
    "glencore": "GLEN.L",
    "unilever": "ULVR.L",

    # US Equities
    "apple": "AAPL",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "palantir": "PLTR",
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "coinbase": "COIN",
    "disney": "DIS",
    "netflix": "NFLX",
    "boeing": "BA",
    "nike": "NKE",
    "broadcom": "AVGO",
    "berkshire": "BRK-B",
}


def extract_ticker_from_text(text: str) -> str | None:
    """Extract stock ticker or company name from user message."""
    t_lower = text.lower()
    # Check multi-word company aliases first, then single-word
    for name, ticker in sorted(_COMPANY_TICKER_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(r'\b' + re.escape(name) + r'\b', t_lower):
            return ticker

    # Check for direct ticker pattern (e.g. 7203.T, SHEL.L, PAYTM.NS, PLTR, BABA)
    tokens = re.findall(r'\b[A-Za-z0-9\.\-]{1,10}\b', text)
    ignored = {
        "I", "A", "AN", "THE", "IS", "ARE", "WAS", "WERE", "DO", "DOES", "DID",
        "CAN", "COULD", "SHOULD", "WOULD", "WILL", "WHAT", "HOW", "WHY", "WHEN",
        "AND", "OR", "NOT", "FOR", "ON", "IN", "AT", "TO", "IT", "ITS", "OF",
        "AI", "US", "UK", "ALL", "ABOUT", "BUY", "SELL", "HOLD", "STOCK", "STOCKS",
        "COMPANY", "PORTFOLIO", "TELL", "ME", "GIVE", "CHECK", "SEE", "LOOK",
    }
    for tok in tokens:
        cand = tok.upper()
        if cand not in ignored and ('.' in cand or (cand.isalpha() and 2 <= len(cand) <= 5)):
            # Verify if it looks like a valid symbol
            return cand
    return None


# ── Chat System Prompt ─────────────────────────────────────────────────────────
_CHAT_SYSTEM_PROMPT = """You are GenWealth AI, an elite institutional financial research and investment advisory intelligence assistant.
You provide deep, actionable, balanced, and articulate market intelligence across global equities:
- Indian Equities (NSE/BSE, e.g. Paytm, Reliance, TCS, HDFC Bank, Tata Motors)
- US Markets (NYSE/NASDAQ, e.g. Palantir, Nvidia, Apple, Microsoft, Tesla)
- Asian & Chinese Equities / ADRs (Alibaba, Tencent, PDD, TSMC)
- Japanese Equities (Tokyo Stock Exchange, e.g. Toyota, Sony, SoftBank)
- European & UK Equities (LSE, Euronext, e.g. Shell, AstraZeneca, ASML)

CORE INSTRUCTIONS FOR STOCK & COMPANY QUERIES:
1. When asked about ANY company or stock (such as Paytm, Alibaba, Toyota, Shell, Palantir, etc.):
   - Provide insightful, high-value financial analysis covering: business model and core revenue segments, recent market catalysts and earnings trajectory, competitive moat vs peers, financial health, and key risks (regulatory, competitive, macro).
   - Conclude with a clear, objective strategic stance (e.g., Bullish, Neutral/Hold, Watchlist, or Cautious Accumulation) with quantitative rationale.
2. Grounding Data Integration:
   - If quantitative data or live signals are provided in the Grounding Context below, seamlessly cite those exact figures (Phase 1 signal, volatility, 1-month return, FinBERT sentiment).
   - If no specific documents or signals are available for a ticker, DO NOT say "I can't answer" or "I don't have enough information". Instead, draw upon your extensive financial, market, and corporate knowledge to give a comprehensive, intelligent breakdown of the company, its sector dynamics, and strategic investment considerations.
3. Response Format:
   - Structure with clean paragraphs and bullet points for readability.
   - Bold key metrics, directional stances, and company milestones.
   - Maintain institutional analytical rigor—fairly present both the bull case and bear risks.
4. Compliance: Never guarantee profits or use prohibited certainty claims. Conclude your analysis constructively."""


def _make_sse(event: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_chat_context(ticker: str | None, message: str) -> str:
    """
    Retrieve RAG context for the active ticker and format it for the LLM prompt.
    Returns an empty string if no ticker is provided or RAG fails.
    """
    if not ticker:
        return ""
    try:
        from src.advisor.vector_store import query_knowledge_base
        docs = query_knowledge_base(
            ticker=ticker,
            query_text=message,
            top_k=2,
        )
        if not docs:
            return ""
        lines = [f"## Knowledge Base Context for {ticker}"]
        for i, d in enumerate(docs, 1):
            meta    = d.get("metadata", {})
            title   = meta.get("title", "Article")
            snippet = d.get("text_content", "")[:400]
            score   = d.get("similarity_score", 0.0)
            lines.append(f"\n**[{i}] {title}** (relevance: {score:.3f})\n{snippet}…")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("[Chat] RAG context failed for '%s': %s", ticker, exc)
        return ""


def _build_phase1_context(ticker: str | None) -> str:
    """Fetch the latest Phase 1 snapshot (CSV or live) for grounding."""
    if not ticker:
        return ""
    try:
        from src.advisor.live_inference import LiveQuantEngine
        engine = LiveQuantEngine()
        lr = engine.compute_live_signal(ticker)
        if lr.error:
            return ""
        label = lr.signal_label
        lstm_str = f"{lr.lstm_prob:.4f}" if lr.lstm_prob is not None else "N/A"
        rf_str   = f"{lr.rf_prob:.4f}" if lr.rf_prob is not None else "N/A"
        return (
            f"\n## Live Quant Signal for {ticker}\n"
            f"- Phase 1 Signal: **{lr.phase1_signal:.4f}** ({label})\n"
            f"- Sentiment (FinBERT/Market): {lr.sentiment:+.4f}\n"
            f"- 20-Day Annualised Volatility: {lr.vol_20:.2%}\n"
            f"- Vol Ratio: {lr.vol_ratio:.3f}\n"
            f"- 1-Month Return: {lr.ret_1m:+.2%}\n"
            f"- Closing Price: {lr.currency} {lr.close_price:,.2f}\n"
            f"- Inference Engine: {lr.inference_mode} (LSTM={lstm_str}, RF={rf_str})\n"
        )
    except Exception as exc:
        logger.warning("[Chat] Phase 1 context failed for '%s': %s", ticker, exc)
        return ""


async def _chat_stream(message: str, ticker: str | None, history: list[dict]) -> AsyncGenerator[str, None]:
    """
    Async generator that:
    1. Identifies stock/company in the query or active ticker.
    2. Retrieves RAG + Phase 1 live quantitative metrics.
    3. Builds the conversational prompt.
    4. Streams LLM response tokens via Groq (with Gemini fallback).
    5. Appends a clean regulatory compliance note.
    """
    t0 = time.perf_counter()

    try:
        # ── 1. Resolve Target Ticker ──────────────────────────────────────────
        detected_ticker = extract_ticker_from_text(message)
        active_ticker   = detected_ticker or ticker

        # ── 2. Build Grounding Context ────────────────────────────────────────
        rag_context    = _build_chat_context(active_ticker, message) if active_ticker else ""
        quant_context  = _build_phase1_context(active_ticker) if active_ticker else ""
        grounding_block = ""
        if rag_context or quant_context:
            grounding_block = (
                f"\n---\n# Live Market & Quant Grounding Context\n{quant_context}\n{rag_context}\n---\n"
            )

        # ── 3. Build Conversation Messages ────────────────────────────────────
        recent_history = history[-20:] if len(history) > 20 else history

        messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
        if grounding_block:
            messages.append({
                "role": "system",
                "content": f"Use the following grounding data for active ticker {active_ticker}:{grounding_block}",
            })
        messages.extend(recent_history)
        messages.append({"role": "user", "content": message})

        # ── 4. Call Groq (Streaming) ──────────────────────────────────────────
        from src.advisor.llm_engine import _get_groq_client, GROQ_PRIMARY_MODEL

        client     = _get_groq_client()
        full_reply = ""
        token_buf  = ""

        try:
            stream = client.chat.completions.create(
                model=GROQ_PRIMARY_MODEL,
                messages=messages,
                max_tokens=900,
                temperature=0.3,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    full_reply += delta
                    token_buf  += delta
                    if len(token_buf) >= 30 or delta.endswith((".", "!", "?", "\n")):
                        yield _make_sse("token", {"type": "token", "content": token_buf})
                        token_buf = ""

            if token_buf:
                yield _make_sse("token", {"type": "token", "content": token_buf})

        except Exception as groq_exc:
            logger.warning("[Chat] Groq streaming failed (%s), trying Gemini…", groq_exc)
            from src.advisor.llm_engine import _load_gemini_api_keys, _make_gemini_client, GEMINI_PRIMARY_MODEL
            gemini_keys = _load_gemini_api_keys()
            if gemini_keys:
                gclient = _make_gemini_client(gemini_keys[0])
                full_prompt = _CHAT_SYSTEM_PROMPT
                if grounding_block:
                    full_prompt += grounding_block
                for m in recent_history:
                    role = "User" if m["role"] == "user" else "Assistant"
                    full_prompt += f"\n{role}: {m['content']}"
                full_prompt += f"\nUser: {message}\nAssistant:"
                resp = gclient.models.generate_content(
                    model=GEMINI_PRIMARY_MODEL,
                    contents=full_prompt,
                )
                full_reply = resp.text or ""
                yield _make_sse("token", {"type": "token", "content": full_reply})
            else:
                raise RuntimeError("Both Groq and Gemini unavailable for chat.")

        # ── 5. Append Compact Regulatory Footer ───────────────────────────────
        chat_footer = (
            "\n\n---\n*⚠️ **Disclaimer**: GenWealth AI is an automated financial research tool for informational "
            "and educational purposes only. Not SEBI (India) or SEC (US) registered investment advice.*"
        )
        yield _make_sse("token", {"type": "token", "content": chat_footer})

        # ── 6. Done Frame ─────────────────────────────────────────────────────
        yield _make_sse("done", {
            "type":             "done",
            "content":          "",
            "disclaimer_added": True,
            "latency_s":        round(time.perf_counter() - t0, 2),
        })

    except Exception as exc:
        logger.error("[Chat] Stream error: %s", exc, exc_info=True)
        yield _make_sse("error", {
            "type":    "error",
            "content": f"Chat error: {type(exc).__name__}: {exc}",
        })


@router.post(
    "/stream",
    summary="Interactive Financial AI Chatbot (SSE Stream)",
    response_description="Server-Sent Events stream: token (repeating) -> done | error",
)
async def chat_stream(body: ChatMessage) -> StreamingResponse:
    """
    Stream a conversational AI response grounded in real-time signals and research.
    """
    return StreamingResponse(
        _chat_stream(body.message, body.ticker, body.conversation_history),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
