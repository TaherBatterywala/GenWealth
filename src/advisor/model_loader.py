"""
GenWealth Advisor Engine — Phase 4, Task 4.2
============================================
Module: src/advisor/model_loader.py

Centralised Model & Artifact Loader — Protobuf-Safe
----------------------------------------------------
Provides a single, process-wide registry of all ML model artifacts used
by the GenWealth pipeline. Designed to solve two problems:

1.  **Protobuf conflict**: ``stable-baselines3`` (SB3) imports Proto-3
    generated code that conflicts with the Proto-4 schema expected by
    ``google-generativeai``. Loading SB3 after google-genai raises
    ``TypeError: Descriptors cannot be created directly`` or
    ``RuntimeError: protobuf version mismatch``.
    Resolution: SB3 is imported inside a defensive ``try/except`` that
    catches both error types. If it fails, ``PPOUnavailableError`` is
    raised and the caller falls back to ``SmartPortfolioAllocator``.

2.  **Cold-start latency**: Loading 500 KB of sklearn/torch files on
    every API request is wasteful. This module provides a shared
    ``_MODEL_CACHE`` dict populated once per process (via the FastAPI
    lifespan startup hook) so every subsequent request hits memory only.

Public API::

    from src.advisor.model_loader import (
        load_lstm_model,        # → (model, scaler)
        load_rf_model,          # → (model, scaler)
        load_phase2_metrics,    # → dict
        get_ppo_weights,        # → dict[str, float]  (with SPA fallback)
        warm_cache,             # → dict[str, bool]   (call on app startup)
        PPOUnavailableError,    # raised when SB3/Protobuf conflict blocks PPO
        MODEL_CACHE,            # read-only view of the cache
    )

Architecture:
    ┌───────────────────────────┐
    │       FastAPI Startup     │  lifespan → warm_cache()
    └──────────────┬────────────┘
                   │  populates _MODEL_CACHE
    ┌──────────────▼────────────┐
    │      _MODEL_CACHE dict     │  { "lstm", "scaler_lstm", "rf", "scaler_rf",
    └──────┬──────────┬──────────┘    "phase2_metrics" }
           │          │
    ┌──────▼──┐ ┌────▼──────────────────────────────────┐
    │  LSTM   │ │  RF    │  Phase 2 metrics (backtest pkl) │
    └─────────┘ └────────────────────────────────────────┘
                          │
              ┌───────────▼───────────────────────────────┐
              │  get_ppo_weights()                         │
              │  → try: SB3 PPO live inference             │
              │  → except PPOUnavailableError:             │
              │       SmartPortfolioAllocator fallback      │
              └────────────────────────────────────────────┘
"""

# ---------------------------------------------------------------------------
# Standard Library
# ---------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Third-Party
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("genwealth.model_loader")

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACT_DIR = _PROJECT_ROOT / "model_artifacts"

_LSTM_PATH      = _ARTIFACT_DIR / "production_lstm.pth"
_RF_PATH        = _ARTIFACT_DIR / "production_rf.pkl"
_SCALER_LSTM    = _ARTIFACT_DIR / "scaler_lstm.pkl"
_SCALER_RF      = _ARTIFACT_DIR / "scaler_rf.pkl"
_PPO_ZIP        = _ARTIFACT_DIR / "ppo_multi_asset_v2.zip"
_VEC_NORM       = _ARTIFACT_DIR / "vec_normalize.pkl"
_BACKTEST_PKL   = _ARTIFACT_DIR / "rl_backtest_metrics_v2.pkl"

# ---------------------------------------------------------------------------
# LSTM Architecture Constants (must match production_lstm.pth exactly)
# ---------------------------------------------------------------------------
_LSTM_INPUT_SIZE  = 7
_LSTM_HIDDEN_SIZE = 64
_LSTM_NUM_LAYERS  = 2
_LSTM_DROPOUT     = 0.2

# ---------------------------------------------------------------------------
# Module-level model cache (populated on first use or via warm_cache())
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}

# Read-only alias exported to callers
MODEL_CACHE = _MODEL_CACHE


# ===========================================================================
# Custom Exceptions
# ===========================================================================

class PPOUnavailableError(RuntimeError):
    """
    Raised when SB3 PPO live inference is blocked by a Protobuf version
    conflict or an import error.

    Callers should catch this exception and fall back to
    ``SmartPortfolioAllocator`` (inverse-volatility × conviction weighting).

    Attributes:
        reason (str): Human-readable description of why PPO is unavailable.
    """
    def __init__(self, reason: str = ""):
        self.reason = reason
        super().__init__(
            f"PPO live inference unavailable: {reason}. "
            "Falling back to SmartPortfolioAllocator."
        )


class ModelLoadError(RuntimeError):
    """Raised when a required model artifact cannot be loaded."""


# ===========================================================================
# 1. LSTM Model Loader
# ===========================================================================

def load_lstm_model(force_reload: bool = False) -> tuple[Any, Any]:
    """
    Load and cache the production PyTorch LSTM model and its scaler.

    Architecture:
        - 2-layer LSTM, input_size=7, hidden_size=64, dropout=0.2
        - Followed by Linear(64, 1) + Sigmoid()
        - Loaded with ``weights_only=False`` to handle legacy checkpoint format

    Args:
        force_reload: If True, bypass the cache and reload from disk.

    Returns:
        tuple[nn.Module, StandardScaler]: (lstm_model, scaler_lstm)

    Raises:
        ModelLoadError: If the .pth file or scaler cannot be loaded.
    """
    if not force_reload and "lstm" in _MODEL_CACHE:
        return _MODEL_CACHE["lstm"], _MODEL_CACHE["scaler_lstm"]

    try:
        import torch
        import torch.nn as nn

        # ── Build architecture ────────────────────────────────────────────────
        class _LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    _LSTM_INPUT_SIZE, _LSTM_HIDDEN_SIZE, _LSTM_NUM_LAYERS,
                    batch_first=True, dropout=_LSTM_DROPOUT,
                )
                self.fc      = nn.Linear(_LSTM_HIDDEN_SIZE, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.lstm(x)
                return self.sigmoid(self.fc(out[:, -1, :]))

        if not _LSTM_PATH.exists():
            raise ModelLoadError(f"LSTM checkpoint not found: {_LSTM_PATH}")

        model = _LSTMModel()
        state_dict = torch.load(
            _LSTM_PATH, map_location="cpu", weights_only=False
        )
        model.load_state_dict(state_dict)
        model.eval()

        # ── Load scaler ───────────────────────────────────────────────────────
        if not _SCALER_LSTM.exists():
            raise ModelLoadError(f"LSTM scaler not found: {_SCALER_LSTM}")
        scaler = joblib.load(_SCALER_LSTM)

        # ── Cache ─────────────────────────────────────────────────────────────
        _MODEL_CACHE["lstm"]        = model
        _MODEL_CACHE["scaler_lstm"] = scaler
        _MODEL_CACHE["_torch"]      = torch   # cache for callers that need it

        logger.info(
            "[ModelLoader] LSTM loaded from '%s'. Parameters: %d.",
            _LSTM_PATH.name,
            sum(p.numel() for p in model.parameters()),
        )
        return model, scaler

    except (ModelLoadError, Exception) as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"LSTM load failed: {exc}") from exc


# ===========================================================================
# 2. Random Forest Model Loader
# ===========================================================================

def load_rf_model(force_reload: bool = False) -> tuple[Any, Any]:
    """
    Load and cache the production Random Forest classifier and its scaler.

    Args:
        force_reload: If True, bypass the cache and reload from disk.

    Returns:
        tuple[RandomForestClassifier, StandardScaler]: (rf_model, scaler_rf)

    Raises:
        ModelLoadError: If the .pkl files cannot be loaded.
    """
    if not force_reload and "rf" in _MODEL_CACHE:
        return _MODEL_CACHE["rf"], _MODEL_CACHE["scaler_rf"]

    try:
        if not _RF_PATH.exists():
            raise ModelLoadError(f"RF model not found: {_RF_PATH}")
        if not _SCALER_RF.exists():
            raise ModelLoadError(f"RF scaler not found: {_SCALER_RF}")

        rf_model = joblib.load(_RF_PATH)
        scaler   = joblib.load(_SCALER_RF)

        _MODEL_CACHE["rf"]        = rf_model
        _MODEL_CACHE["scaler_rf"] = scaler

        logger.info(
            "[ModelLoader] RF loaded from '%s'. Estimators: %d.",
            _RF_PATH.name,
            getattr(rf_model, "n_estimators", "?"),
        )
        return rf_model, scaler

    except (ModelLoadError, Exception) as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"RF load failed: {exc}") from exc


# ===========================================================================
# 3. Phase 2 Backtest Metrics Loader
# ===========================================================================

def load_phase2_metrics(force_reload: bool = False) -> dict:
    """
    Load and cache the Phase 2 PPO backtest metrics pickle.

    The pickle maps strategy names (``"PPO Agent"``, ``"Equal-Weight"``,
    ``"Buy-and-Hold"``) to dicts containing:
        total_return, sharpe, max_drawdown, calmar, portfolio (list[float])

    Args:
        force_reload: If True, bypass the cache and reload from disk.

    Returns:
        dict: Full backtest metrics dictionary.

    Raises:
        ModelLoadError: If the .pkl file cannot be found or loaded.
    """
    if not force_reload and "phase2_metrics" in _MODEL_CACHE:
        return _MODEL_CACHE["phase2_metrics"]

    try:
        if not _BACKTEST_PKL.exists():
            raise ModelLoadError(f"Backtest metrics not found: {_BACKTEST_PKL}")

        metrics = joblib.load(_BACKTEST_PKL)
        _MODEL_CACHE["phase2_metrics"] = metrics

        strategies = list(metrics.keys())
        logger.info(
            "[ModelLoader] Phase 2 metrics loaded. Strategies: %s", strategies
        )
        return metrics

    except (ModelLoadError, Exception) as exc:
        if isinstance(exc, ModelLoadError):
            raise
        raise ModelLoadError(f"Phase 2 metrics load failed: {exc}") from exc


# ===========================================================================
# 4. PPO Policy Loader (Protobuf-Safe)
# ===========================================================================

def _try_load_sb3_ppo() -> Any:
    """
    Attempt to load the SB3 PPO policy with full Protobuf-safe isolation.

    This function isolates SB3 imports inside a try/except so that the
    google-generativeai Protobuf environment is not corrupted if SB3's
    descriptor cache is already populated.

    Returns:
        Loaded PPO model object.

    Raises:
        PPOUnavailableError: On any import/deserialization failure.
    """
    try:
        # SB3 must be imported FIRST before google-generativeai in the same
        # process. If google-generativeai was imported first (the normal API
        # server case), this will raise a TypeError.
        import stable_baselines3 as sb3_module      # noqa: F401
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize

        if not _PPO_ZIP.exists():
            raise PPOUnavailableError(f"PPO zip not found: {_PPO_ZIP}")

        model = PPO.load(str(_PPO_ZIP), device="cpu")
        logger.info("[ModelLoader] SB3 PPO loaded successfully.")
        return model

    except (TypeError, RuntimeError) as exc:
        # Protobuf version conflict
        raise PPOUnavailableError(
            f"Protobuf conflict with google-generativeai: {exc}"
        ) from exc
    except ImportError as exc:
        raise PPOUnavailableError(
            f"stable-baselines3 not importable: {exc}"
        ) from exc
    except PPOUnavailableError:
        raise
    except Exception as exc:
        raise PPOUnavailableError(f"Unexpected SB3 error: {exc}") from exc


def get_ppo_model(force_reload: bool = False) -> Any:
    """
    Return the PPO model, loading from cache or attempting SB3 deserialization.

    Raises:
        PPOUnavailableError: If loading fails due to the Protobuf conflict.
    """
    if not force_reload and "ppo" in _MODEL_CACHE:
        return _MODEL_CACHE["ppo"]

    model = _try_load_sb3_ppo()
    _MODEL_CACHE["ppo"] = model
    return model


# ===========================================================================
# 5. Safe PPO Action Prediction
# ===========================================================================

def safe_ppo_predict(obs: np.ndarray) -> np.ndarray:
    """
    Run a forward pass through the PPO policy for a given observation.

    This is the primary inference point for the RL allocation layer. It wraps
    ``PPO.predict()`` with full exception isolation so that the API server
    never crashes on a Protobuf conflict.

    Args:
        obs: Observation vector as a numpy array matching the training env's
             observation space (shape depends on the specific PPO training config).

    Returns:
        numpy.ndarray: Predicted action (target weights per asset).

    Raises:
        PPOUnavailableError: If PPO cannot be loaded or inference fails.
    """
    try:
        model = get_ppo_model()
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action)
    except PPOUnavailableError:
        raise
    except Exception as exc:
        raise PPOUnavailableError(f"PPO.predict() failed: {exc}") from exc


# ===========================================================================
# 6. Smart Portfolio Allocation Fallback
# ===========================================================================

def get_smart_allocator_weights(
    tickers:    list[str],
    price_hist: dict[str, Optional[pd.DataFrame]],
    signals:    dict[str, float],
    ai_stances: dict[str, str],
) -> dict[str, float]:
    """
    Compute inverse-volatility × conviction portfolio weights.

    This is the guaranteed-working fallback when PPO live inference is blocked
    by the Protobuf conflict. Uses the same ``SmartPortfolioAllocator`` class
    that the DynamicTradeSimulator uses, which is already stress-tested across
    16/16 global pipeline tests.

    Args:
        tickers:    Ordered list of ticker symbols.
        price_hist: Dict of {ticker: OHLCV DataFrame or None}.
        signals:    Phase 1 signal (or momentum proxy) per ticker.
        ai_stances: AI directional stance (BUY/HOLD/SELL/REDUCE) per ticker.

    Returns:
        Dict {ticker: weight} where weights sum to ≤ 1.0.
    """
    try:
        from src.advisor.trade_execution_simulator import SmartPortfolioAllocator
        allocator = SmartPortfolioAllocator(
            tickers=tickers,
            price_hist=price_hist,
            signals=signals,
            ai_stances=ai_stances,
        )
        weights = allocator.compute_weights()
        logger.info(
            "[ModelLoader] SmartPortfolioAllocator weights: %s",
            {k: f"{v:.3f}" for k, v in weights.items()},
        )
        return weights

    except Exception as exc:
        logger.error(
            "[ModelLoader] SmartPortfolioAllocator failed: %s — equal-weight fallback.", exc
        )
        n = len(tickers)
        return {t: (1.0 / n) if n > 0 else 0.0 for t in tickers}


# ===========================================================================
# 7. Unified PPO-or-Fallback Allocation
# ===========================================================================

def get_ppo_weights(
    tickers:    list[str],
    obs_vec:    Optional[np.ndarray] = None,
    price_hist: Optional[dict] = None,
    signals:    Optional[dict] = None,
    ai_stances: Optional[dict] = None,
) -> tuple[dict[str, float], str]:
    """
    Return portfolio weights using PPO live inference if available, otherwise
    fall back to ``SmartPortfolioAllocator``.

    Decision tree:
        1. ``obs_vec`` provided + PPO loads cleanly → run ``safe_ppo_predict()``
        2. PPO blocked (``PPOUnavailableError``) → ``get_smart_allocator_weights()``
        3. ``obs_vec`` is None → skip to step 2 immediately

    Args:
        tickers:    List of ticker symbols.
        obs_vec:    PPO observation vector (numpy array). If None, skips PPO.
        price_hist: Required for the SmartPortfolioAllocator fallback.
        signals:    Phase 1 signals per ticker.
        ai_stances: AI stances per ticker.

    Returns:
        tuple[dict[str, float], str]: (weights, method_used)
            where method_used is ``"PPO_LIVE"`` or ``"SMART_ALLOCATOR"``.
    """
    price_hist  = price_hist  or {}
    signals     = signals     or {t: 0.5 for t in tickers}
    ai_stances  = ai_stances  or {t: "HOLD" for t in tickers}

    # ── Attempt PPO live inference ────────────────────────────────────────────
    if obs_vec is not None:
        try:
            raw_action = safe_ppo_predict(obs_vec)
            # Normalise action vector to sum = 1.0 (PPO may output unbounded values)
            raw = np.asarray(raw_action, dtype=float)
            # Softmax normalisation to get proper probability weights
            exp_raw = np.exp(raw - raw.max())
            norm    = exp_raw / exp_raw.sum()
            n_tickers = min(len(tickers), len(norm))
            weights = {tickers[i]: float(norm[i]) for i in range(n_tickers)}
            # Zero-fill any extra tickers
            for t in tickers[n_tickers:]:
                weights[t] = 0.0
            logger.info("[ModelLoader] PPO live inference succeeded.")
            return weights, "PPO_LIVE"

        except PPOUnavailableError as exc:
            logger.warning(
                "[ModelLoader] PPO unavailable (%s). Using SmartPortfolioAllocator.", exc.reason
            )

    # ── SmartPortfolioAllocator fallback ──────────────────────────────────────
    weights = get_smart_allocator_weights(tickers, price_hist, signals, ai_stances)
    return weights, "SMART_ALLOCATOR"


# ===========================================================================
# 8. Artifact Health Check
# ===========================================================================

def check_artifact_health() -> dict[str, Any]:
    """
    Check the existence and basic integrity of all model artifact files.

    Used by the ``GET /api/v1/health`` endpoint to report model readiness
    without attempting expensive model deserialization on every health ping.

    Returns:
        dict with keys:
            - ``lstm_ok``         (bool): .pth file exists and is > 100 KB
            - ``rf_ok``           (bool): .pkl file exists and is > 100 KB
            - ``scaler_lstm_ok``  (bool): scaler exists
            - ``scaler_rf_ok``    (bool): scaler exists
            - ``ppo_zip_ok``      (bool): .zip exists and is > 1 MB
            - ``backtest_ok``     (bool): backtest .pkl exists
            - ``cache_loaded``    (list): list of currently cached model names
            - ``ppo_conflict``    (str):  "UNKNOWN" | "DETECTED" | "OK"
    """
    def _file_ok(path: Path, min_bytes: int = 100_000) -> bool:
        return path.exists() and os.path.getsize(path) >= min_bytes

    # Quick SB3 compatibility probe (no deserialization — import only)
    ppo_conflict = "UNKNOWN"
    try:
        import stable_baselines3  # noqa: F401
        ppo_conflict = "OK"
    except (ImportError, TypeError, RuntimeError) as exc:
        ppo_conflict = f"DETECTED: {type(exc).__name__}"
    except Exception:
        ppo_conflict = "UNKNOWN"

    return {
        "lstm_ok":        _file_ok(_LSTM_PATH,    min_bytes=100_000),
        "rf_ok":          _file_ok(_RF_PATH,       min_bytes=100_000),
        "scaler_lstm_ok": _file_ok(_SCALER_LSTM,   min_bytes=500),
        "scaler_rf_ok":   _file_ok(_SCALER_RF,     min_bytes=500),
        "ppo_zip_ok":     _file_ok(_PPO_ZIP,       min_bytes=1_000_000),
        "backtest_ok":    _BACKTEST_PKL.exists(),
        "cache_loaded":   list(_MODEL_CACHE.keys()),
        "ppo_conflict":   ppo_conflict,
    }


# ===========================================================================
# 9. Warm Cache (called on API server startup)
# ===========================================================================

def warm_cache() -> dict[str, Any]:
    """
    Pre-load all non-SB3 model artifacts into ``_MODEL_CACHE``.

    Designed to be called from the FastAPI lifespan startup event to
    eliminate cold-start latency on the first API request.

    SB3 PPO is intentionally NOT pre-loaded here because:
        1. It may fail due to the Protobuf conflict.
        2. The ``get_ppo_weights()`` function already falls back gracefully.
        3. Attempting SB3 import at startup could crash the server if the
           conflict is severe.

    Returns:
        dict[str, Any]: {model_name: True/False/error_message}
    """
    results: dict[str, Any] = {}

    # LSTM
    try:
        load_lstm_model()
        results["lstm"] = True
    except Exception as exc:
        logger.error("[WarmCache] LSTM failed: %s", exc)
        results["lstm"] = str(exc)

    # Random Forest
    try:
        load_rf_model()
        results["rf"] = True
    except Exception as exc:
        logger.error("[WarmCache] RF failed: %s", exc)
        results["rf"] = str(exc)

    # Phase 2 metrics
    try:
        load_phase2_metrics()
        results["phase2_metrics"] = True
    except Exception as exc:
        logger.warning("[WarmCache] Phase 2 metrics failed: %s", exc)
        results["phase2_metrics"] = str(exc)

    # SB3 PPO — probed but not loaded
    try:
        import stable_baselines3  # noqa: F401
        results["ppo_sb3_available"] = True
    except Exception as exc:
        results["ppo_sb3_available"] = f"Unavailable: {exc}"

    logger.info("[WarmCache] Model cache warm: %s", results)
    return results


# ===========================================================================
# Self-Test
# ===========================================================================

if __name__ == "__main__":
    """
    Quick self-test for model_loader.py.

    Run from project root:
        python -m src.advisor.model_loader
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 65)
    print("  GenWealth — Phase 4 | model_loader.py Self-Test")
    print("=" * 65 + "\n")

    # --- Artifact Health Check -----------------------------------------------
    print("[TEST 1] Artifact Health Check")
    health = check_artifact_health()
    for k, v in health.items():
        status = "✓" if v is True else ("⚠" if isinstance(v, str) else "✗")
        print(f"  {status}  {k}: {v}")
    print()

    # --- LSTM Load -----------------------------------------------------------
    print("[TEST 2] LSTM Model Load")
    try:
        m, s = load_lstm_model()
        params = sum(p.numel() for p in m.parameters())
        print(f"  PASS  params={params:,}  scaler={type(s).__name__}")
    except Exception as e:
        print(f"  FAIL  {e}")
    print()

    # --- RF Load -------------------------------------------------------------
    print("[TEST 3] RF Model Load")
    try:
        rf, s = load_rf_model()
        print(f"  PASS  estimators={getattr(rf, 'n_estimators', '?')}  scaler={type(s).__name__}")
    except Exception as e:
        print(f"  FAIL  {e}")
    print()

    # --- Phase 2 Metrics -----------------------------------------------------
    print("[TEST 4] Phase 2 Backtest Metrics")
    try:
        metrics = load_phase2_metrics()
        ppo = metrics.get("PPO Agent", {})
        print(f"  PASS  strategies={list(metrics.keys())}")
        print(f"        PPO return={ppo.get('total_return', 'N/A'):.2%}  sharpe={ppo.get('sharpe', 'N/A'):.3f}")
    except Exception as e:
        print(f"  FAIL  {e}")
    print()

    # --- PPO / Fallback ------------------------------------------------------
    print("[TEST 5] PPO Weights (with SmartPortfolioAllocator fallback)")
    try:
        test_tickers = ["NVDA", "AAPL", "MSFT"]
        test_signals = {t: 0.60 for t in test_tickers}
        test_stances = {t: "BUY" for t in test_tickers}
        weights, method = get_ppo_weights(
            tickers=test_tickers,
            obs_vec=None,     # force SmartPortfolioAllocator path
            signals=test_signals,
            ai_stances=test_stances,
        )
        print(f"  PASS  method={method}")
        for t, w in weights.items():
            print(f"        {t}: {w:.3f}")
    except Exception as e:
        print(f"  FAIL  {e}")
    print()

    # --- Warm Cache ----------------------------------------------------------
    print("[TEST 6] warm_cache()")
    try:
        result = warm_cache()
        for k, v in result.items():
            status = "✓" if v is True else "⚠"
            print(f"  {status}  {k}: {v}")
    except Exception as e:
        print(f"  FAIL  {e}")

    print("\n" + "=" * 65)
    print("  model_loader.py self-test complete.")
    print("=" * 65 + "\n")
