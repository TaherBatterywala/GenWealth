"""
GenWealth — Market Data Router
================================
File: app/routers/market.py

GET /api/v1/market/history
    Fetches historical OHLCV candlestick and line price series for any ticker.
    Eliminates client-side CORS issues with Yahoo Finance.
"""

import logging
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query

from src.advisor.llm_engine import get_ticker_currency

logger = logging.getLogger("genwealth.api.market")

router = APIRouter(prefix="/api/v1/market", tags=["Market Data"])


@router.get(
    "/history",
    summary="Get OHLCV Candlestick & Line Chart Data",
)
async def get_price_history(
    ticker: str = Query(..., description="Stock ticker symbol (e.g. NVDA, PAYTM.NS, 7203.T)"),
    period: str = Query("30d", description="Lookback period: 7d, 30d, 90d, 1y, 2y"),
):
    """
    Fetch OHLCV price history for TradingView / Chart.js candle and line charts.
    """
    ticker = ticker.strip().upper()
    valid_periods = {"7d", "30d", "90d", "1y", "2y", "5y"}
    if period not in valid_periods:
        period = "30d"

    currency_code, currency_symbol, exchange_name = get_ticker_currency(ticker)

    try:
        # Pull adjusted OHLCV
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)

        if hist is None or hist.empty:
            raise HTTPException(status_code=404, detail=f"No price history found for ticker '{ticker}'.")

        # Normalise MultiIndex columns if present
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Drop timezone info
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        candles = []
        for dt, row in hist.iterrows():
            if pd.isna(row.get("Close")):
                continue
            date_str = str(dt.date())
            candles.append({
                "time":   date_str,
                "open":   round(float(row.get("Open", row["Close"])), 2),
                "high":   round(float(row.get("High", row["Close"])), 2),
                "low":    round(float(row.get("Low", row["Close"])), 2),
                "close":  round(float(row["Close"]), 2),
                "volume": int(row.get("Volume", 0)),
            })

        if not candles:
            raise HTTPException(status_code=404, detail=f"Insufficient candle data for ticker '{ticker}'.")

        current_price = candles[-1]["close"]
        prev_close    = candles[-2]["close"] if len(candles) >= 2 else candles[0]["open"]
        delta         = round(current_price - prev_close, 2)
        delta_pct     = round((delta / prev_close * 100) if prev_close > 0 else 0.0, 2)

        return {
            "ticker":         ticker,
            "currency":       currency_code,
            "symbol":         currency_symbol,
            "exchange":       exchange_name,
            "period":         period,
            "current_price":  current_price,
            "prev_close":     prev_close,
            "change":         delta,
            "change_pct":     delta_pct,
            "n_candles":      len(candles),
            "candles":        candles,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Market/History] Failed for '%s': %s", ticker, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch market history for {ticker}: {exc}")
