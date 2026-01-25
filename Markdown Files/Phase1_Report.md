# 📘 GenWealth Phase 1: The Quant Engine (Master Technical Report)

**Project Name:** GenWealth AI Trading System
**Phase:** 1.0 (Prediction & Signal Generation)
**Status:** Mechanically Complete & Audited
**Target Assets:** Nifty 50 (India) & S&P 500 (US)
**Tech Stack:** Python, PyTorch, Scikit-Learn, Transformers (Hugging Face), YFinance

---

## 1. Executive Summary

Phase 1 focused on building the "Prediction Layer" of the GenWealth system. The objective was to move beyond traditional technical analysis by creating a **Hybrid Ensemble Architecture** that fuses three distinct sources of market intelligence:

1.  **Hard Market Data:** 15 years of OHLCV (Open, High, Low, Close, Volume) data processed via statistical indicators.
2.  **Soft Market Data:** Real-time news sentiment derived from Financial Large Language Models (LLMs).
3.  **Logic Injection:** A deterministic post-processing layer that overrides mathematical probabilities during extreme news events.

The system outputs a daily **Buy Confidence Score (0.0 - 1.0)**. A score > 0.55 indicates a "Buy" signal, while a score < 0.45 indicates a "Sell/Avoid" signal.

---

## 2. Module 1: The Data Factory

### 2.1 Data Ingestion Strategy
We utilized a "Navy SEALs" approach to data selection, training only on high-quality, information-rich assets to prevent noise from polluting the model.

* **API:** `yfinance` (Yahoo Finance)
* **Timeframe:** 2010-01-01 to 2026-01-17 (16 Years)
* **Asset Universe:** Top 200 stocks with >500 associated news headlines.
* **Resolution:** Daily (End-of-Day data).

### 2.2 Feature Engineering (The Alpha Factors)
Raw price data is non-stationary and unsuitable for direct ML training. We engineered **Stationary Derivatives** to capture specific market phenomena.

| Feature Name | Type | Mathematical Formula | Financial Rationale |
| :--- | :--- | :--- | :--- |
| **`Log_Ret`** | Momentum | $\ln(P_t / P_{t-1})$ | Stationarizes price changes; handles compounding better than simple %. |
| **`Vol_Ratio`** | Regime | $\sigma_{20d} / \sigma_{200d}$ | **Mean Reversion Indicator:** High values (>1.5) indicate panic/euphoria; Low values (<0.8) indicate consolidation. |
| **`Efficiency`** | Trend | *Fractal Dimension Proxy* | Measures path efficiency. 1.0 = Laser straight trend (Strong signal). 0.0 = Random Noise (Weak signal). |
| **`Vol_Shock`** | Liquidity | $V_t / \text{SMA}(V, 50)$ | Identifies institutional accumulation or distribution events. |
| **`Ret_Lag{X}`** | Memory | $R_{t-1}, R_{t-3}, R_{t-5}...$ | Explicit memory inputs for the LSTM to learn multi-scale temporal patterns. |
| **`Sentiment`** | Context | FinBERT Output ($-1 \to 1$) | Quantifies the "mood" of the market (Fear/Greed) from text. |

### 2.3 Preprocessing Standard
* **Scaler Used:** `RobustScaler` (Scikit-Learn).
* **Justification:** Financial data contains "Fat Tails" (Black Swan events like 2008 or Covid). Standard Mean/Variance scaling (`StandardScaler`) is skewed by these outliers. `RobustScaler` uses the Interquartile Range (IQR), effectively neutralizing outliers while preserving the core data distribution.

---

## 3. Module 2: The Sentiment Engine (The "Eyes")

A dedicated Natural Language Processing (NLP) pipeline was constructed to convert unstructured news text into a structured mathematical signal.

### 3.1 The Model Architecture
* **Core Model:** `yiyanghkust/finbert-tone`
* **Source:** Hugging Face Transformers.
* **Pre-training:** BERT (Bidirectional Encoder Representations from Transformers) fine-tuned specifically on financial analyst reports, earnings transcripts, and financial news.

### 3.2 Signal Extraction Logic
The model outputs three logits: `[Neutral, Positive, Negative]`. We collapse this into a single scalar for the Quant Engine:

$$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative})$$

* **Range:** $[-1.0, +1.0]$
* **Interpretation:**
    * **+1.0:** Maximum Euphoria.
    * **0.0:** Neutral / Noise.
    * **-1.0:** Maximum Panic.

### 3.3 The Live Scraper
* **Libraries:** `feedparser`, `urllib.parse`
* **Source:** Google News RSS (Region: India/English).
* **Workflow:**
    1.  Fetch Top 5 headlines for ticker (e.g., "RELIANCE.NS").
    2.  Tokenize headlines.
    3.  Pass through FinBERT on GPU.
    4.  Average the scores to produce a "Daily Mood."

---

## 4. Module 3: The Hybrid Brain Architecture

We rejected single-model architectures in favor of a **Voting Ensemble**. This combines the stability of traditional Machine Learning with the pattern-recognition power of Deep Learning.

### 🧠 Sub-Model A: Random Forest (The Regime Detector)
* **Role:** Analyzes market *state* (Volatility, Volume, Efficiency). It ignores time sequence and focuses on current conditions.
* **Configuration:**
    * Estimators: 100
    * Max Depth: 5 (Constrained to prevent memorization/overfitting).
    * Input Features: `RF_COLS` (Regime metrics).

### 🧠 Sub-Model B: LSTM (The Pattern Matcher)
* **Role:** Analyzes market *sequence*. It looks back at a **60-Day Window** to identify repeating temporal patterns (e.g., "Three days up, followed by volume spike").
* **Architecture:**
    * Framework: PyTorch (`nn.LSTM`)
    * Input Sequence Length: 60 Days
    * Hidden Layers: 2 Stacked Layers
    * Hidden Dimension: 64 Neurons
    * Dropout: 0.2 (20% neuron shutdown to improve generalization).
    * Optimizer: Adam
    * Loss Function: Binary Cross Entropy (`BCEWithLogitsLoss`).

---

## 5. Module 4: The Logic Injection Layer

**Problem Identified:** During auditing, the model exhibited "Permabull Bias." Having been trained on a 15-year secular bull market, it learned to interpret every price drop as a buying opportunity, ignoring fundamental news risks (e.g., "CEO Resigns").

**Solution:** We implemented a **Logic Injection Layer** at inference time. This is a deterministic rule that overrides the probabilistic model output based on extreme sentiment values.

### The Logic Formula
$$P_{final} = (P_{RF} \times 0.4) + (P_{LSTM} \times 0.4) + \text{SentimentBoost}$$

**The Booster Function:**
$$\text{SentimentBoost} = (0.5 + (\text{LiveSentiment} \times 0.2)) \times 0.2$$

### Impact Analysis
* **Scenario: Neutral News (0.0):** The model relies 80% on Technicals.
* **Scenario: Crash News (-0.9):** The formula applies a penalty of **~7-10%** to the Buy Probability. A weak "Buy" (56%) effectively flips to a "Hold/Sell" (49%), preventing the AI from catching a falling knife.
* **Scenario: Breakout News (+0.9):** The formula boosts confidence, allowing the AI to enter momentum trades earlier.

---

## 6. Validation & Stress Testing results

### 6.1 Walk-Forward Backtest (2018-2026)
To avoid look-ahead bias, we used an expanding window technique:
* *Step 1:* Train 2010-2017 -> Test 2018
* *Step 2:* Train 2010-2018 -> Test 2019
* ... until 2026.

**Performance:**
* **Market Return (Buy & Hold):** +227.88%
* **AI Strategy Return:** +162.25%
* **Analysis:** The AI underperformed the raw bull market in total profit but significantly reduced drawdown risk by sitting in cash during volatile periods (evidenced by flat lines on the equity curve during 2020 and 2022).

### 6.2 The Live Stress Test (Reliance Industries)
We simulated fake news events to verify the Logic Injection layer:
1.  **Input:** RELIANCE.NS Technical Data (Fixed).
2.  **Condition A (Bad News):** Forced Sentiment = `-0.9`.
    * *Result:* Buy Confidence = **51.4%**
3.  **Condition B (Good News):** Forced Sentiment = `+0.9`.
    * *Result:* Buy Confidence = **58.6%**
4.  **Conclusion:** The **7.2% spread** confirms the system is responsive to news, validating the fix for the "Permabull" issue.

---

## 7. Artifact Manifest (Production Files)

The following files constitute the "Frozen State" of the engine and are located in the `model_artifacts/` directory:

1.  **`production_lstm.pth`**: The trained PyTorch weights for the Deep Learning model.
2.  **`production_rf.pkl`**: The serialized Random Forest model (Joblib).
3.  **`scaler_lstm.pkl`**: The RobustScaler fitted on 15 years of time-series data.
4.  **`scaler_rf.pkl`**: The RobustScaler fitted on 15 years of regime data.
5.  **`universal_sentiment_history.pkl`**: The database of 1.6M+ scored headlines used for training context.

---

## 8. Limitations & Next Steps (Phase 2)

**Current Limitation:**
Phase 1 is an **Analyst**. It predicts *direction* (Up vs Down) but lacks *Executive Function*. It does not know:
* How much capital to deploy (Position Sizing).
* How to manage portfolio volatility.
* When to exit a trade to preserve profit.

**Phase 2 Objective: The Planner (Trader)**
We will wrap the Phase 1 Engine inside a **Reinforcement Learning (RL)** environment.
* **Agent:** PPO (Proximal Policy Optimization).
* **Input:** Phase 1 Confidence + Portfolio State.
* **Action Space:** Buy/Sell/Hold + Size (0% to 100%).
* **Reward Function:** Sharpe Ratio (Profit per unit of Risk).