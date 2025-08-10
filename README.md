# 🚀 Quantitative Trading Strategy with Machine Learning

<br>

<p align="center">
  <img src="https://github.com/aryadoshii/Return-Prediction-Model/blob/main/dashboard_screenshot.png" alt="Dashboard Screenshot" width="800"/>
</p>

<p align="center">
  <strong>An end-to-end Python pipeline for forecasting stock returns and backtesting a market-neutral trading strategy.</strong>
</p>

---

## 🎯 Project Overview

This project develops and evaluates a complete machine learning system designed to identify profitable trading opportunities in the S&P 100. It moves beyond simple prediction by implementing a full quantitative strategy, from feature engineering and model training to rigorous, out-of-sample backtesting.

The core of the project is a **market-neutral, long-short portfolio**. The strategy uses a Random Forest model to rank stocks based on their predicted 5-day returns, going **Long** the top 20% (the winners) and **Short** the bottom 20% (the losers). This approach aims to deliver consistent returns (alpha) regardless of the overall market's direction (beta).

The entire analysis is packaged into a user-friendly, interactive dashboard built with **Streamlit** and **Plotly**.

---

## 📈 Final Strategy Performance & Key Metrics

The backtest was conducted on out-of-sample data from **January 2023 to December 2024**. The results demonstrate a statistically significant and profitable edge.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Sharpe Ratio** | **1.19** | Measures risk-adjusted return. A value > 1 is considered excellent. |
| **Maximum Drawdown** | **-2.03%** | The largest peak-to-trough drop in portfolio value. A low value indicates robust risk management. |
| **Hit Rate** | **51.61%** | The percentage of profitable trading days, proving a consistent predictive edge over a 50/50 chance. |

---

## 🛠️ Technical Architecture & Workflow

The project is structured as a modular pipeline, ensuring reproducibility and scalability.

**1. Data Collection (`yfinance`):**
   - Fetches daily OHLCV data for the top 100 S&P 500 components, plus SPY (benchmark) and ^VIX (market volatility index).

**2. Feature Engineering (`pandas`, `pandas_ta`):**
   - Constructs a library of over 30 predictive features for each stock, grounded in established quantitative finance principles.

**3. Model Training (`scikit-learn`, `XGBoost`):**
   - Systematically evaluates three core models using time-series aware splitting (training on 2021-2022, testing on 2023-2024).
     - **Linear Regression:** A simple, interpretable baseline.
     - **Random Forest:** A robust ensemble model for capturing non-linearities and providing feature importance.
     - **XGBoost:** A high-performance gradient boosting model, an industry standard for structured data.

**4. Backtesting & Analysis:**
   - Implements a vectorized backtest for a quintile-based, long-short portfolio strategy.
   - Calculates key performance and risk metrics to evaluate the strategy's real-world viability.

**5. Interactive Dashboard (`Streamlit`):**
   - Deploys a user-facing dashboard to visualize the cumulative returns, drawdown periods, and model feature importances, turning static analysis into an interactive tool.

---

## 🔍 Factor Library & Feature Importance

The model's predictive power is derived from a diverse set of quantitative factors. Feature importance analysis from the Random Forest model revealed that market context and risk factors were the most critical signals.

<p align="center">
  <img src="URL_TO_YOUR_FEATURE_IMPORTANCE_PLOT.png" alt="Feature Importance" width="600"/>
</p>

| Factor Category | Key Features Implemented |
| :--- | :--- |
| **Market Regime** | **VIX Level (Most Important Feature)**, Beta to SPY |
| **Volatility** | Historical Volatility (20-day), Relative Volatility, ATR |
| **Volume & Flow**| On-Balance Volume (OBV), Volume Momentum |
| **Momentum** | RSI (14-day), MACD Signals, Price Returns (5, 10, 20-day) |
| **Mean-Reversion**| Bollinger Band Position, Z-Score (20-day) |

---

## 💡 Key Interview Discussion Points & Learnings

1.  **Regression vs. Ranking:** Initial attempts to predict the raw return value resulted in a negative R-squared, a common and expected outcome for noisy financial data. The key insight was to pivot from a regression problem to a **ranking problem**, where the model's ability to simply rank stocks proved highly profitable.
2.  **The Importance of a Market-Neutral Strategy:** A simple "long-only" strategy would be highly correlated with the market. By constructing a long-short portfolio, the strategy isolates **alpha** (skill-based returns) and hedges against general market risk (**beta**).
3.  **Feature Insights:** The dominance of `vix_level` as the top feature underscores the principle that **market context is often more important than any single stock-specific indicator.** The strategy's success depends on understanding the prevailing risk environment.
4.  **Debugging Real-World Systems:** A significant portion of the project involved solving practical software engineering challenges, such as dependency conflicts (`numpy`, `XGBoost`), security certificate issues on macOS, and resolving file path inconsistencies between Jupyter and Streamlit environments by creating a centralized `config.py` file.


## 💡 Key Discussion Points & Learnings

This project provides a rich foundation for discussing practical challenges in quantitative finance and machine learning.

#### 1. Why Time-Series Validation is Non-Negotiable
*   **Preventing Data Leakage:** Traditional K-fold cross-validation shuffles data, allowing the model to "see the future." This leads to overly optimistic performance that fails in live trading.
*   **Simulating Reality:** A fixed time-series split (training on 2021-2022, testing on 2023-2024) mimics the real-world scenario where a model, trained on past data, must make predictions on unseen future data.
*   **Walk-Forward Validation as a Next Step:** While a single split was used for this project, a more robust approach (and a potential future enhancement) would be walk-forward validation, where the model is periodically retrained to adapt to changing market conditions.

#### 2. The "Regression vs. Ranking" Problem
*   **The Futility of Raw Return Prediction:** Initial models (Linear Regression, RF, XGBoost) all yielded a negative R-squared. This is a critical finding: predicting the *exact* future return value is nearly impossible due to market noise.
*   **Pivoting to a More Solvable Problem:** The project's success came from reframing the objective. Instead of asking "What will the return be?", we asked "Which stocks will likely perform better than others?". This **ranking** approach is far more robust to noise.
*   **The Power of Quantiles:** The quintile-based portfolio construction is a direct application of this ranking philosophy, focusing only on the tails (the strongest buy/sell signals) and ignoring the noisy predictions in the middle.

#### 3. Feature Engineering & Insights
*   **The Dominance of Market Regime:** The Random Forest model identified `vix_level` as the most important feature by a wide margin. This demonstrates that understanding the overall market context (risk-on/risk-off) is more critical than any single stock-specific indicator.
*   **Creating Stationary Features:** Stock prices are non-stationary. By using features based on *changes* and *relationships* (e.g., percentage returns, oscillators like RSI, spreads like Bollinger Bands), we transform the data into a more stationary form that machine learning models can learn from effectively.
*   **Handling Survivorship Bias:** By pulling the S&P 100 list at the beginning of the project, we introduce a slight survivorship bias. A more advanced system would use point-in-time historical constituent lists to ensure the model is only trained on stocks that were actually in the index at that time.

#### 4. The Rationale for a Market-Neutral Strategy
*   **Isolating Alpha:** A simple "long-only" strategy that buys the top-ranked stocks would still be highly correlated with the overall market (S&P 500). If the market goes up, the strategy looks good; if it goes down, it looks bad, regardless of the model's skill.
*   **Hedging Beta:** By simultaneously going Long the top quintile and Short the bottom quintile, the strategy aims to hedge out this general market risk (`beta`). This forces the portfolio's performance to be dependent on the model's genuine ability to select winners and losers, thereby isolating its **alpha**.

---

## 🔮 Future Enhancements & Next Steps

This project provides a strong foundation that can be extended in several key areas:

*   **More Sophisticated Backtesting:**
    -   **Transaction Costs & Slippage:** Incorporate trading costs to simulate more realistic net returns.
    -   **Walk-Forward Validation:** Implement a rolling window validation scheme for more robust performance metrics.
*   **Advanced Feature Engineering:**
    -   **Cross-Sectional Ranking:** For each day, rank features (e.g., RSI) across all stocks. A stock with an RSI of 70 is more significant if it's in the 99th percentile of all stocks' RSIs that day.
    -   **Alternative Data:** Integrate fundamental data (P/E, P/B ratios) or alternative data sets (e.g., news sentiment, options-implied volatility) to provide orthogonal signals.
*   **Advanced Modeling:**
    -   **Hyperparameter Tuning:** Use `GridSearchCV` with time-series splits to find the optimal parameters for the Random Forest or XGBoost models.
    -   **Deep Learning:** Experiment with LSTM (Long Short-Term Memory) networks, which are designed to capture complex temporal sequences in data.
*   **Risk Management Integration:**
    -   **Dynamic Position Sizing:** Size positions based on the model's confidence or on volatility forecasts (e.g., smaller positions in more volatile stocks).
    -   **Portfolio-Level Risk Models:** Implement Value-at-Risk (VaR) calculations or other risk management overlays.
 
  ---

## 🚀 How to Run This Project

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/10-Day-Stock-Return-Prediction.git
    cd 10-Day-Stock-Return-Prediction
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    *Note: If on macOS, you may need to install Homebrew and OpenMP first: `brew install libomp`*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Data Pipeline (Notebooks):**
    Open and run the Jupyter Notebooks in sequential order to generate the required data artifacts.
    - `01_data_collection.ipynb`
    - `02_feature_engineering.ipynb`
5.  **Launch the Interactive Dashboard:**
    ```bash
    streamlit run dashboard.py
    ```

---
