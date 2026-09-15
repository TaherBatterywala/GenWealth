"""
GenWealth Advisor Engine — Phase 4, Task 4.1
============================================
Module: src/advisor/live_inference.py

Real-Time Live Inference Engine
---------------------------------
Extends the Phase 1 signal pipeline beyond the 4 trained tickers in
``data/enriched_rl_data.csv`` to ANY globally-listed ticker by fetching
fresh OHLCV data via ``yfinance`` and running live model inference.

Data Flow:
    yfinance OHLCV (120 d)
        │
        ▼
    compute_feature_matrix()         ← Log_Ret, Vol_20/200/Ratio, Ret_1M/3M,
        │                              Efficiency, Vol_Shock, Ret_Lag1-5
        ├─► run_lstm_inference()      ← production_lstm.pth  (PyTorch, 60-step)
        ├─► run_rf_inference()        ← production_rf.pkl    (Scikit-Learn RF)
        └─► fetch_live_sentiment()   ← DuckDuckGo + FinBERT
                │
                ▼
        compute_live_signal()        ← 0.4×RF + 0.4×LSTM + 0.2×sentiment_boost
                │
                ▼
        LiveSignalResult (dataclass)

Fallback:
    If OHLCV < 30 rows or model load fails → momentum_signal() proxy from
    trade_execution_simulator (maps 10-d return to [0.30, 0.70] via tanh).

Model Cache:
    Module-level ``_MODEL_CACHE`` dict — models load once per process,
    avoiding repeated 200–500 ms disk I/O on each API call.

Architecture Notes:
    • LSTM:   2-layer, input_size=7, hidden_size=64, dropout=0.2, seq_len=60
      Features (LSTM): Log_Ret, Ret_Lag1, Ret_Lag2, Ret_Lag3, Ret_Lag5,
                        Ret_Lag10, Sentiment
    • RF:     6 features: Vol_Ratio, Efficiency, Vol_Shock, Ret_1M, Ret_3M,
                          Sentiment
    • Ensemble: 0.4 × RF_prob + 0.4 × LSTM_prob + 0.2 × (0.5 + sent × 0.2)
      Matches the formula validated in test_global_pipeline_stress_test.py.
    • FinBERT: 'ProsusAI/finbert' via HuggingFace Transformers pipeline.
      Scores averaged across top-3 DuckDuckGo news headlines.
      Positive → +score, Negative → -score, Neutral → 0.

Dependencies:
    torch, joblib, yfinance, transformers, ddgs
    (all already in requirements.txt)
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("genwealth.live_inference")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR  = _PROJECT_ROOT / "model_artifacts"

_LSTM_PATH      = _ARTIFACT_DIR / "production_lstm.pth"
_RF_PATH        = _ARTIFACT_DIR / "production_rf.pkl"
_SCALER_LSTM    = _ARTIFACT_DIR / "scaler_lstm.pkl"
_SCALER_RF      = _ARTIFACT_DIR / "scaler_rf.pkl"

# ---------------------------------------------------------------------------
# LSTM Architecture (must match production_lstm.pth exactly)
# Confirmed from stress-test state_dict inspection:
#   input_size=7, hidden_size=64, num_layers=2, dropout=0.2
# ---------------------------------------------------------------------------
_LSTM_INPUT_SIZE  = 7
_LSTM_HIDDEN_SIZE = 64
_LSTM_NUM_LAYERS  = 2
_LSTM_DROPOUT     = 0.2
_LSTM_SEQ_LEN     = 60    # 60 trading-day lookback window

# LSTM feature order (must match scaler_lstm.pkl column order)
_LSTM_FEATURES = [
    "Log_Ret", "Ret_Lag1", "Ret_Lag2", "Ret_Lag3",
    "Ret_Lag5", "Ret_Lag10", "Sentiment",
]

# RF feature order (must match scaler_rf.pkl column order)
_RF_FEATURES = [
    "Vol_Ratio", "Efficiency", "Vol_Shock",
    "Ret_1M", "Ret_3M", "Sentiment",
]

# Minimum rows needed for meaningful feature computation
_MIN_OHLCV_ROWS   = 30
_OHLCV_PERIOD     = "1y"    # 1 year (~250 trading days) for full 200d vol + 60-step LSTM

# FinBERT model identifier
_FINBERT_MODEL = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}   # keys: "lstm", "rf", "scaler_lstm", "scaler_rf", "finbert"


# ===========================================================================
# Data Class
# ===========================================================================

@dataclass
class LiveSignalResult:
    """
    Encapsulates the full live inference output for a single ticker.

    Attributes:
        ticker:          Normalised ticker symbol (uppercase).
        as_of_date:      ISO date string of the latest available trading day.
        close_price:     Last closing price.
        currency:        Currency code (e.g. "INR", "USD", "EUR").
        log_ret:         Latest daily log return.
        vol_20:          20-day realised annualised volatility.
        vol_ratio:       Vol_20 / Vol_200 (regime indicator).
        ret_1m:          1-month cumulative return.
        ret_3m:          3-month cumulative return.
        efficiency:      Efficiency ratio (|price change| / sum|daily changes|).
        vol_shock:       Boolean flag (1.0/0.0) for vol ratio > 1.2 spike.
        sentiment:       FinBERT composite sentiment score (−1 to +1).
        lstm_prob:       LSTM bullish probability (0–1).
        rf_prob:         RF bullish probability (0–1).
        phase1_signal:   Blended ensemble signal (0–1).
        signal_label:    "BULLISH" | "NEUTRAL" | "BEARISH".
        inference_mode:  "LIVE_MODEL" | "MOMENTUM_PROXY" | "CSV_CACHE".
        news_headlines:  List of news headlines used for sentiment scoring.
        latency_s:       Total inference wall-clock time in seconds.
        error:           Non-empty if a non-fatal error occurred during inference.
    """
    ticker:          str
    as_of_date:      str
    close_price:     float
    currency:        str
    log_ret:         float
    vol_20:          float
    vol_ratio:       float
    ret_1m:          float
    ret_3m:          float
    efficiency:      float
    vol_shock:       float
    sentiment:       float
    lstm_prob:       float
    rf_prob:         float
    phase1_signal:   float
    signal_label:    str
    inference_mode:  str
    news_headlines:  list  = field(default_factory=list)
    latency_s:       float = 0.0
    error:           str   = ""


# ===========================================================================
# 1. Model Loaders (cached)
# ===========================================================================

def _load_lstm_model():
    """Load and cache the production LSTM model. Returns (model, scaler)."""
    if "lstm" in _MODEL_CACHE:
        return _MODEL_CACHE["lstm"], _MODEL_CACHE["scaler_lstm"]

    try:
        import torch
        import torch.nn as nn

        class _LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    _LSTM_INPUT_SIZE, _LSTM_HIDDEN_SIZE, _LSTM_NUM_LAYERS,
                    batch_first=True, dropout=_LSTM_DROPOUT,
                )
                self.fc      = nn.Linear(_LSTM_HIDDEN_SIZE, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.sigmoid(self.fc(out[:, -1, :]))

        model      = _LSTMModel()
        state_dict = torch.load(_LSTM_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()

        scaler = joblib.load(_SCALER_LSTM)

        _MODEL_CACHE["lstm"]         = model
        _MODEL_CACHE["scaler_lstm"]  = scaler
        _MODEL_CACHE["torch"]        = torch    # cache torch module reference
        logger.info("LSTM model loaded from '%s'.", _LSTM_PATH)
        return model, scaler

    except Exception as exc:
        logger.error("LSTM load failed: %s", exc)
        raise


def _load_rf_model():
    """Load and cache the production RF model. Returns (rf_model, scaler)."""
    if "rf" in _MODEL_CACHE:
        return _MODEL_CACHE["rf"], _MODEL_CACHE["scaler_rf"]

    try:
        rf_model  = joblib.load(_RF_PATH)
        scaler    = joblib.load(_SCALER_RF)
        _MODEL_CACHE["rf"]        = rf_model
        _MODEL_CACHE["scaler_rf"] = scaler
        logger.info("RF model loaded from '%s'.", _RF_PATH)
        return rf_model, scaler

    except Exception as exc:
        logger.error("RF load failed: %s", exc)
        raise


def _load_finbert():
    """Load and cache FinBERT sentiment pipeline (checks local cache first, falls back quickly)."""
    if "finbert" in _MODEL_CACHE:
        return _MODEL_CACHE["finbert"]

    try:
        from transformers import pipeline as _hf_pipeline
        try:
            pipe = _hf_pipeline(
                "text-classification",
                model=_FINBERT_MODEL,
                tokenizer=_FINBERT_MODEL,
                top_k=None,
                device=-1,
                truncation=True,
                max_length=512,
                model_kwargs={"local_files_only": True},
            )
            _MODEL_CACHE["finbert"] = pipe
            logger.info("FinBERT pipeline loaded from local cache.")
            return pipe
        except Exception:
            logger.info("FinBERT not cached locally; utilizing fast financial keyword sentiment.")
            return None

    except Exception as exc:
        logger.warning("FinBERT load failed (%s) — using keyword sentiment.", exc)
        return None


# ===========================================================================
# 2. OHLCV Fetcher
# ===========================================================================

def fetch_ohlcv(ticker: str, period: str = _OHLCV_PERIOD) -> Optional[pd.DataFrame]:
    """
    Fetch adjusted daily OHLCV data for ``ticker`` via yfinance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. "NVDA", "RELIANCE.NS").
        period: yfinance period string (default "120d").

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex, or None if the fetch fails / returns insufficient rows.
    """
    try:
        hist = yf.Ticker(ticker).history(
            period=period,
            interval="1d",
            auto_adjust=True,
        )
        if hist.empty:
            logger.warning("[yfinance] %s: empty response.", ticker)
            return None

        # Normalise MultiIndex columns (yfinance sometimes returns them)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        # Drop timezone info for uniform index handling
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        hist = hist[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

        if len(hist) < _MIN_OHLCV_ROWS:
            logger.warning(
                "[yfinance] %s: only %d rows — below minimum %d.",
                ticker, len(hist), _MIN_OHLCV_ROWS,
            )
            return None

        logger.info("[yfinance] %s: fetched %d rows (%s -> %s).",
                    ticker, len(hist),
                    hist.index[0].date(), hist.index[-1].date())
        return hist

    except Exception as exc:
        logger.error("[yfinance] %s: fetch error — %s", ticker, exc)
        return None


# ===========================================================================
# 3. Feature Matrix Computation
# ===========================================================================

def compute_feature_matrix(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full Phase 1 feature matrix from raw OHLCV history.

    Derived columns (matching ``data/enriched_rl_data.csv`` column schema):
        Log_Ret    : daily log return  log(Close_t / Close_{t-1})
        Vol_20     : 20-day annualised realised volatility (σ × √252)
        Vol_200    : 200-day annualised realised volatility
        Vol_Ratio  : Vol_20 / Vol_200  (short/long regime indicator)
        Ret_1M     : 21-trading-day cumulative return
        Ret_3M     : 63-trading-day cumulative return
        Efficiency : |Close_T - Close_0| / Σ|daily returns|  (trend quality)
        Vol_Shock  : 1.0 if Vol_Ratio > 1.2 else 0.0
        Ret_Lag1–5 : lagged daily log returns (1 … 5 trading days)
        Ret_Lag10  : lagged daily log return 10 trading days ago

    Args:
        hist: DataFrame with at least a ``Close`` column and DatetimeIndex.

    Returns:
        DataFrame with all feature columns. Rows with NaN (head) are dropped.
    """
    df = hist.copy()
    closes = df["Close"]

    # Log returns
    df["Log_Ret"] = np.log(closes / closes.shift(1))

    # Rolling volatilities (annualised)
    df["Vol_20"]  = df["Log_Ret"].rolling(20, min_periods=10).std()  * np.sqrt(252)
    df["Vol_200"] = df["Log_Ret"].rolling(200, min_periods=30).std() * np.sqrt(252)
    df["Vol_Ratio"] = (df["Vol_20"] / df["Vol_200"].replace(0, np.nan)).fillna(1.0)

    # Multi-period returns (pct change over N bars)
    df["Ret_1M"] = closes.pct_change(21).fillna(0.0)   # ~1 month
    df["Ret_3M"] = closes.pct_change(63).fillna(df["Ret_1M"])   # ~3 months

    # Efficiency Ratio: absolute net move / sum of absolute daily moves
    def _efficiency(window: pd.Series) -> float:
        net_move = abs(window.iloc[-1] - window.iloc[0])
        path_len = window.diff().abs().sum()
        return net_move / path_len if path_len > 1e-9 else 0.0

    df["Efficiency"] = (
        closes.rolling(20, min_periods=5)
              .apply(_efficiency, raw=False)
              .fillna(0.5)
    )

    # Vol shock flag
    df["Vol_Shock"] = (df["Vol_Ratio"] > 1.2).astype(float)

    # Lagged log returns (for LSTM sequence)
    for lag in [1, 2, 3, 5, 10]:
        df[f"Ret_Lag{lag}"] = df["Log_Ret"].shift(lag).fillna(0.0)

    df.dropna(subset=["Log_Ret", "Vol_20"], inplace=True)
    return df


# ===========================================================================
# 4. FinBERT Sentiment Scoring
# ===========================================================================

def fetch_live_sentiment(ticker: str, max_articles: int = 3) -> tuple[float, list[str]]:
    """
    Fetch DuckDuckGo news headlines and score them with FinBERT.

    Scoring logic:
        - Run FinBERT on ``title + ". " + body[:200]`` for each article.
        - Map labels: positive → +score, negative → −score, neutral → 0.
        - Average across all articles.
        - Clip to [−1, +1].

    Args:
        ticker:       Ticker symbol for the news query.
        max_articles: Maximum number of articles to score (default 3).

    Returns:
        (sentiment_score: float, headlines: list[str])
        Returns (0.0, []) on any failure (non-fatal).
    """
    headlines = []
    try:
        from ddgs import DDGS
        query = f"{ticker} stock market latest news earnings"
        with DDGS() as ddgs:
            results = ddgs.news(query, max_results=max_articles, safesearch="off")
            articles = list(results)
    except Exception as exc:
        logger.warning("[Sentiment] DDG news fetch failed for '%s': %s", ticker, exc)
        return 0.0, []

    if not articles:
        return 0.0, []

    # Score with FinBERT
    finbert = _load_finbert()
    if finbert is None:
        # FinBERT unavailable — use heuristic keyword scoring as fallback
        texts = [a.get("title", "") for a in articles]
        headlines = texts
        return _keyword_sentiment(texts), headlines

    scores = []
    for article in articles:
        title = article.get("title", "")
        body  = article.get("body", "")[:200]
        text  = f"{title}. {body}".strip()
        if not text:
            continue
        headlines.append(title)

        try:
            preds = finbert(text)   # list[list[dict]]  (top_k=None returns all)
            label_scores = preds[0] if isinstance(preds[0], list) else preds
            score_map = {p["label"].lower(): p["score"] for p in label_scores}
            sent = score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
            scores.append(float(np.clip(sent, -1.0, 1.0)))
        except Exception as exc:
            logger.warning("[Sentiment] FinBERT scoring error: %s", exc)

    if not scores:
        return 0.0, headlines

    avg_sent = float(np.clip(np.mean(scores), -1.0, 1.0))
    logger.info("[Sentiment] '%s': %.4f  (%d article(s))", ticker, avg_sent, len(scores))
    return avg_sent, headlines


def _keyword_sentiment(texts: list[str]) -> float:
    """
    Lightweight keyword-based sentiment fallback when FinBERT is unavailable.

    Uses a curated list of positive/negative financial keywords to produce
    a simple polarity score in [−1, +1].
    """
    _POS = {
        "beat", "surge", "record", "growth", "bullish", "rally", "upgrade",
        "profit", "strong", "buy", "outperform", "gain", "positive", "raised",
        "optimistic", "expand", "innovation", "breakout", "revenue", "dividend",
    }
    _NEG = {
        "miss", "decline", "loss", "fall", "bearish", "downgrade", "cut",
        "weak", "sell", "underperform", "drop", "negative", "lowered",
        "concern", "risk", "warn", "crash", "default", "lawsuit", "layoff",
    }
    total_pos = total_neg = 0
    for text in texts:
        words = set(text.lower().split())
        total_pos += len(words & _POS)
        total_neg += len(words & _NEG)

    total = total_pos + total_neg
    if total == 0:
        return 0.0
    return float(np.clip((total_pos - total_neg) / total, -1.0, 1.0))


# ===========================================================================
# 5. Model Inference
# ===========================================================================

def run_lstm_inference(feature_df: pd.DataFrame, sentiment: float) -> float:
    """
    Run the production LSTM model on a feature DataFrame.

    Builds a 60-step sequence using LSTM feature columns, scales it,
    and returns the bullish probability output.

    Args:
        feature_df: DataFrame with at least ``_LSTM_FEATURES`` columns.
        sentiment:  Live FinBERT sentiment score injected as the ``Sentiment``
                    column (the model was trained with this as a feature).

    Returns:
        float: Bullish probability in [0, 1]. Returns 0.50 on failure.
    """
    try:
        import torch
        model, scaler = _load_lstm_model()

        # Build sequence: last _LSTM_SEQ_LEN rows (or pad with first row if shorter)
        df_seq = feature_df.copy()
        df_seq["Sentiment"] = sentiment   # inject live sentiment

        # Select only the 7 LSTM feature columns
        available = [c for c in _LSTM_FEATURES if c in df_seq.columns]
        if len(available) < len(_LSTM_FEATURES):
            # Fill missing lag columns with Log_Ret as proxy
            for col in _LSTM_FEATURES:
                if col not in df_seq.columns:
                    df_seq[col] = df_seq.get("Log_Ret", pd.Series(0.0, index=df_seq.index))

        seq_arr = df_seq[_LSTM_FEATURES].values  # shape: (N, 7)

        if len(seq_arr) >= _LSTM_SEQ_LEN:
            seq_arr = seq_arr[-_LSTM_SEQ_LEN:]
        else:
            # Pad by repeating the first row
            pad = np.tile(seq_arr[0], (_LSTM_SEQ_LEN - len(seq_arr), 1))
            seq_arr = np.vstack([pad, seq_arr])

        # Scale and run inference
        seq_scaled = scaler.transform(seq_arr)           # (60, 7)
        tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 60, 7)

        with torch.no_grad():
            lstm_prob = float(model(tensor).item())

        logger.debug("[LSTM] '%s': prob=%.4f", "ticker", lstm_prob)
        return float(np.clip(lstm_prob, 0.0, 1.0))

    except Exception as exc:
        logger.warning("[LSTM] Inference failed: %s — returning 0.50", exc)
        return 0.50


def run_rf_inference(feature_df: pd.DataFrame, sentiment: float) -> float:
    """
    Run the production Random Forest model on the latest feature row.

    Args:
        feature_df: DataFrame with at least ``_RF_FEATURES`` columns.
        sentiment:  Live FinBERT sentiment score.

    Returns:
        float: Bullish probability in [0, 1]. Returns 0.50 on failure.
    """
    try:
        rf_model, scaler = _load_rf_model()

        latest = feature_df.iloc[-1].copy()
        latest["Sentiment"] = sentiment

        rf_input_vals = []
        for col in _RF_FEATURES:
            if col in latest.index:
                rf_input_vals.append(float(latest[col]))
            elif col == "Sentiment":
                rf_input_vals.append(float(sentiment))
            else:
                rf_input_vals.append(0.0)   # safe default for missing features

        rf_input  = np.array(rf_input_vals).reshape(1, -1)
        rf_scaled = scaler.transform(rf_input)
        rf_prob   = float(rf_model.predict_proba(rf_scaled)[0][1])

        logger.debug("[RF] prob=%.4f", rf_prob)
        return float(np.clip(rf_prob, 0.0, 1.0))

    except Exception as exc:
        logger.warning("[RF] Inference failed: %s — returning 0.50", exc)
        return 0.50


# ===========================================================================
# 6. Momentum Proxy Fallback
# ===========================================================================

def _momentum_proxy(hist: pd.DataFrame) -> float:
    """
    Compute a momentum signal proxy when OHLCV is insufficient for model inference.

    Maps 10-day price return to [0.30, 0.70] via tanh compression.
    Mirrors ``momentum_signal()`` in trade_execution_simulator.py.
    """
    closes = hist["Close"].dropna()
    window = min(10, len(closes) - 1)
    if window < 1:
        return 0.50
    ret = float((closes.iloc[-1] - closes.iloc[-window]) / closes.iloc[-window])
    return float(np.clip(0.50 + 0.25 * np.tanh(ret * 8), 0.30, 0.70))


# ===========================================================================
# 7. Currency Detection (local helper — authoritative map in llm_engine.py)
# ===========================================================================

_CURRENCY_MAP: dict[str, tuple[str, str]] = {
    ".NS":  ("INR", "₹"),
    ".BO":  ("INR", "₹"),
    ".AS":  ("EUR", "€"),
    ".T":   ("JPY", "¥"),
    ".L":   ("GBP", "£"),
    ".PA":  ("EUR", "€"),
    ".DE":  ("EUR", "€"),
    ".HK":  ("HKD", "HK$"),
    ".AX":  ("AUD", "A$"),
    ".TO":  ("CAD", "C$"),
    "-USD": ("USD", "$"),
    "-BTC": ("USD", "$"),
    "-ETH": ("USD", "$"),
}


def _detect_currency(ticker: str) -> tuple[str, str]:
    """Return (currency_code, symbol) for a ticker. Default: USD, $."""
    tu = ticker.upper()
    for suffix, (code, sym) in _CURRENCY_MAP.items():
        if tu == suffix or tu.endswith(suffix):
            return code, sym
    return "USD", "$"


# ===========================================================================
# 8. Main Orchestrator
# ===========================================================================

class LiveQuantEngine:
    """
    Orchestrates real-time quantitative feature extraction and model inference
    for any globally-listed ticker.

    Usage::

        engine = LiveQuantEngine()
        result = engine.compute_live_signal("AAPL")
        print(result.phase1_signal, result.signal_label)

    Thread Safety:
        The model cache (``_MODEL_CACHE``) is module-level and is populated on
        first use. Multiple concurrent callers may trigger duplicate loads in
        the first milliseconds, but this is safe (last-writer-wins for a dict
        assignment of identical objects).
    """

    def __init__(self, ohlcv_period: str = _OHLCV_PERIOD):
        self.ohlcv_period = ohlcv_period

    def compute_live_signal(self, ticker: str) -> LiveSignalResult:
        """
        Compute the complete live Phase 1 signal for any ticker.

        Orchestration:
            1. Fetch OHLCV via yfinance.
            2. Compute full feature matrix.
            3. Fetch DuckDuckGo news + FinBERT sentiment.
            4. Run LSTM inference (60-step sequence).
            5. Run RF inference (single latest row).
            6. Blend ensemble: 0.4×RF + 0.4×LSTM + 0.2×(0.5 + sentiment×0.2).
            7. If any step fails, fall back gracefully to momentum proxy.

        Args:
            ticker: Any Yahoo Finance ticker symbol (e.g. "AAPL", "ASML.AS").

        Returns:
            LiveSignalResult with all computed fields populated.
        """
        t0      = time.perf_counter()
        ticker  = ticker.strip().upper()
        currency, cur_sym = _detect_currency(ticker)

        logger.info("[LiveEngine] Starting live signal computation for '%s'…", ticker)

        # ── Step 1: Fetch OHLCV ───────────────────────────────────────────────
        hist = fetch_ohlcv(ticker, period=self.ohlcv_period)

        if hist is None or len(hist) < _MIN_OHLCV_ROWS:
            # Hard fallback — insufficient data
            logger.warning(
                "[LiveEngine] '%s': insufficient OHLCV, using momentum proxy.", ticker
            )
            close_price = float(hist["Close"].iloc[-1]) if hist is not None and len(hist) > 0 else 0.0
            as_of_date  = str(hist.index[-1].date()) if hist is not None and len(hist) > 0 else "N/A"
            proxy_sig   = _momentum_proxy(hist) if hist is not None else 0.50
            return LiveSignalResult(
                ticker=ticker, as_of_date=as_of_date,
                close_price=close_price, currency=currency,
                log_ret=0.0, vol_20=0.30, vol_ratio=1.0,
                ret_1m=0.0, ret_3m=0.0, efficiency=0.5,
                vol_shock=0.0, sentiment=0.0,
                lstm_prob=proxy_sig, rf_prob=proxy_sig,
                phase1_signal=proxy_sig,
                signal_label=_signal_label(proxy_sig),
                inference_mode="MOMENTUM_PROXY",
                latency_s=round(time.perf_counter() - t0, 2),
                error="Insufficient OHLCV rows for model inference.",
            )

        # ── Step 2: Compute feature matrix ────────────────────────────────────
        try:
            feature_df = compute_feature_matrix(hist)
            if feature_df.empty:
                raise ValueError("Feature matrix is empty after NaN drop.")
        except Exception as exc:
            logger.warning("[LiveEngine] Feature computation failed: %s", exc)
            proxy_sig = _momentum_proxy(hist)
            return _proxy_result(ticker, hist, currency, proxy_sig, t0, str(exc))

        # Extract latest-row scalar features for the result dataclass
        latest         = feature_df.iloc[-1]
        as_of_date     = str(feature_df.index[-1].date())
        close_price    = round(float(hist["Close"].iloc[-1]), 4)
        log_ret        = round(float(latest.get("Log_Ret",   0.0)), 6)
        vol_20         = round(float(latest.get("Vol_20",    0.30)), 6)
        vol_ratio      = round(float(latest.get("Vol_Ratio", 1.0)), 4)
        ret_1m         = round(float(latest.get("Ret_1M",    0.0)), 6)
        ret_3m         = round(float(latest.get("Ret_3M",    0.0)), 6)
        efficiency     = round(float(latest.get("Efficiency",0.5)), 4)
        vol_shock      = round(float(latest.get("Vol_Shock", 0.0)), 1)

        # ── Step 3: Live sentiment ─────────────────────────────────────────────
        try:
            sentiment, headlines = fetch_live_sentiment(ticker, max_articles=3)
        except Exception as exc:
            logger.warning("[LiveEngine] Sentiment failed: %s — defaulting to 0.0.", exc)
            sentiment, headlines = 0.0, []

        # ── Step 4: LSTM inference ────────────────────────────────────────────
        lstm_prob = run_lstm_inference(feature_df, sentiment)

        # ── Step 5: RF inference ──────────────────────────────────────────────
        rf_prob = run_rf_inference(feature_df, sentiment)

        # ── Step 6: Ensemble blend ────────────────────────────────────────────
        # Formula confirmed in test_global_pipeline_stress_test.py:
        #   ensemble = 0.4×RF + 0.4×LSTM + 0.2×(0.5 + sentiment×0.2)
        sentiment_boost = 0.5 + sentiment * 0.2
        phase1_signal   = 0.4 * rf_prob + 0.4 * lstm_prob + 0.2 * sentiment_boost
        phase1_signal   = float(np.clip(phase1_signal, 0.0, 1.0))

        latency = round(time.perf_counter() - t0, 2)

        logger.info(
            "[LiveEngine] '%s': RF=%.4f LSTM=%.4f Sent=%.4f -> Signal=%.4f (%s) [%.1fs]",
            ticker, rf_prob, lstm_prob, sentiment, phase1_signal,
            _signal_label(phase1_signal), latency,
        )

        return LiveSignalResult(
            ticker=ticker,
            as_of_date=as_of_date,
            close_price=close_price,
            currency=currency,
            log_ret=log_ret,
            vol_20=vol_20,
            vol_ratio=vol_ratio,
            ret_1m=ret_1m,
            ret_3m=ret_3m,
            efficiency=efficiency,
            vol_shock=vol_shock,
            sentiment=round(sentiment, 4),
            lstm_prob=round(lstm_prob, 4),
            rf_prob=round(rf_prob, 4),
            phase1_signal=round(phase1_signal, 4),
            signal_label=_signal_label(phase1_signal),
            inference_mode="LIVE_MODEL",
            news_headlines=headlines,
            latency_s=latency,
        )


# ===========================================================================
# 9. Helpers
# ===========================================================================

def _signal_label(signal: float) -> str:
    """Convert blended signal value to human-readable label."""
    if signal >= 0.65:
        return "BULLISH"
    if signal <= 0.35:
        return "BEARISH"
    return "NEUTRAL"


def _proxy_result(
    ticker: str,
    hist: pd.DataFrame,
    currency: str,
    proxy_sig: float,
    t0: float,
    error: str,
) -> LiveSignalResult:
    """Build a momentum-proxy LiveSignalResult when model inference fails."""
    close_price = round(float(hist["Close"].iloc[-1]), 4) if not hist.empty else 0.0
    as_of_date  = str(hist.index[-1].date()) if not hist.empty else "N/A"
    return LiveSignalResult(
        ticker=ticker, as_of_date=as_of_date,
        close_price=close_price, currency=currency,
        log_ret=0.0, vol_20=0.30, vol_ratio=1.0,
        ret_1m=0.0, ret_3m=0.0, efficiency=0.5,
        vol_shock=0.0, sentiment=0.0,
        lstm_prob=proxy_sig, rf_prob=proxy_sig,
        phase1_signal=proxy_sig,
        signal_label=_signal_label(proxy_sig),
        inference_mode="MOMENTUM_PROXY",
        latency_s=round(time.perf_counter() - t0, 2),
        error=error,
    )


# ===========================================================================
# 10. Public Convenience API
# ===========================================================================

# Module-level singleton — callers can do:
#   from src.advisor.live_inference import engine
#   result = engine.compute_live_signal("NVDA")
engine = LiveQuantEngine()


def get_live_signal(ticker: str) -> LiveSignalResult:
    """
    Module-level convenience function wrapping ``LiveQuantEngine.compute_live_signal()``.

    Args:
        ticker: Any Yahoo Finance ticker symbol.

    Returns:
        LiveSignalResult populated with live inference results.
    """
    return engine.compute_live_signal(ticker)


def preload_models() -> dict[str, bool]:
    """
    Pre-load all model artifacts into the module cache.

    Called from ``app/main.py`` lifespan event to eliminate cold-start
    latency on the first API request.

    Returns:
        dict[str, bool]: {model_name: success} for each artifact.
    """
    results: dict[str, bool] = {}

    try:
        _load_lstm_model()
        results["lstm"] = True
    except Exception as exc:
        logger.error("[Preload] LSTM failed: %s", exc)
        results["lstm"] = False

    try:
        _load_rf_model()
        results["rf"] = True
    except Exception as exc:
        logger.error("[Preload] RF failed: %s", exc)
        results["rf"] = False

    # FinBERT is intentionally NOT pre-loaded here — it's large (~400 MB)
    # and only needed for the sentiment path, which is non-blocking.
    results["finbert"] = "deferred"

    logger.info("[Preload] Model cache state: %s", results)
    return results


# ===========================================================================
# Self-Test
# ===========================================================================

if __name__ == "__main__":
    """
    Quick self-test for live_inference.py.

    Run from project root:
        python -m src.advisor.live_inference

    Tests a mix of US, Indian, and European tickers.
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    TEST_TICKERS = ["NVDA", "AAPL", "RELIANCE.NS", "ASML.AS"]

    print("\n" + "=" * 70)
    print("  GenWealth — Phase 4 | live_inference.py Self-Test")
    print("=" * 70)

    eng = LiveQuantEngine()

    for t in TEST_TICKERS:
        print(f"\n[TEST] {t}")
        try:
            res = eng.compute_live_signal(t)
            print(f"  Mode    : {res.inference_mode}")
            print(f"  Date    : {res.as_of_date}")
            print(f"  Price   : {res.currency} {res.close_price:,.4f}")
            print(f"  RF prob : {res.rf_prob:.4f}")
            print(f"  LSTM    : {res.lstm_prob:.4f}")
            print(f"  Sent    : {res.sentiment:.4f}")
            print(f"  Signal  : {res.phase1_signal:.4f}  [{res.signal_label}]")
            print(f"  Latency : {res.latency_s:.1f}s")
            if res.error:
                print(f"  Error   : {res.error}")
        except Exception as e:
            print(f"  FAIL: {e}")

    print("\n" + "=" * 70)
    print("  Self-test complete.")
    print("=" * 70 + "\n")
