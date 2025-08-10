# dashboard.py

# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# --- ML and Processing Imports ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# --- CRITICAL: Add Project Root to the Python Path ---
# This allows the script to find the 'config.py' file.
# Replace this path with the absolute path you found using the 'pwd' command in your terminal.
PROJECT_ROOT_PATH = "/Users/aryadoshii/Desktop/stock-return-prediction"
sys.path.append(PROJECT_ROOT_PATH)
import config


# --- Page Configuration ---
# This must be the first Streamlit command.
st.set_page_config(
    page_title="Quantitative Trading Strategy Dashboard",
    page_icon="🚀",
    layout="wide"
)


# --- Caching ---
# Use caching to prevent expensive re-computation on every widget interaction.

@st.cache_data
def load_and_predict():
    """
    This function loads the data from the absolute path defined in config.py,
    trains the model, and makes predictions.
    """
    # 1. Load the processed data using the absolute path from the config file
    if not os.path.exists(config.PROCESSED_FILE):
        st.error(f"CRITICAL ERROR: The data file was not found at {config.PROCESSED_FILE}")
        st.error("Please run notebook '02_feature_engineering.ipynb' successfully to generate the data file.")
        st.stop() # Stop the app if data doesn't exist.

    df = pd.read_parquet(config.PROCESSED_FILE)
    df.index = pd.to_datetime(df.index)

    # 2. Re-create the test set predictions (logic from Notebook 03)
    features = df.drop(columns=['target_5d_forward_return', 'ticker']).select_dtypes(include=np.number)
    target = df['target_5d_forward_return']
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(features.median(), inplace=True)

    split_date = '2023-01-01'
    X_train = features[features.index < split_date]
    X_test = features[features.index >= split_date]
    y_train = target[target.index < split_date]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    predictions = rf_model.predict(X_test_scaled)

    # 3. Add predictions to the test set DataFrame
    test_df = df[df.index >= split_date].copy()
    test_df['prediction'] = predictions
    
    return test_df, rf_model, X_train.columns

@st.cache_data
def perform_backtest(data):
    """
    This function takes the data with predictions and runs the backtest.
    """
    # 1. Generate trading signals
    data['rank'] = data.groupby(level=0)['prediction'].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
    )
    data['signal'] = 0
    data.loc[data['rank'] == 4, 'signal'] = 1  # Long
    data.loc[data['rank'] == 0, 'signal'] = -1 # Short
    
    # 2. Calculate returns
    data['daily_return'] = data.groupby('ticker')['Close'].pct_change()
    data['strategy_return'] = data['daily_return'] * data['signal'].shift(1)
    
    daily_portfolio_returns = data.groupby(level=0)['strategy_return'].mean()
    cumulative_returns = (1 + daily_portfolio_returns).cumprod() - 1
    
    return daily_portfolio_returns, cumulative_returns


# --- Main Application ---

# 1. Title and Introduction
st.title("📈 Quantitative Trading Strategy Dashboard")
st.markdown("""
This dashboard presents the backtest results of a market-neutral, long-short equity strategy.
The strategy uses a Random Forest model to predict 5-day future returns for S&P 100 stocks.
It goes **Long** the top 20% of stocks with the highest predicted returns and **Short** the bottom 20%.
""")

# 2. Load data and run models
with st.spinner('Loading data, training model, and running backtest... This may take a minute on first load.'):
    test_data_with_preds, model, feature_names = load_and_predict()
    daily_returns, cumulative_returns = perform_backtest(test_data_with_preds)

st.success('Analysis Complete!')

# 3. Display Key Performance Indicators (KPIs)
st.header("Strategy Performance Metrics")
sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
rolling_max = (cumulative_returns + 1).cummax()
daily_drawdown = (cumulative_returns + 1) / rolling_max - 1
max_drawdown = daily_drawdown.min()
hit_rate = (daily_returns > 0).sum() / len(daily_returns)

col1, col2, col3 = st.columns(3)
col1.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
col2.metric("Maximum Drawdown", f"{max_drawdown:.2%}")
col3.metric("Hit Rate (Profitable Days)", f"{hit_rate:.2%}")

# 4. Display Charts in Tabs
st.header("Performance Analysis")
tab1, tab2, tab3 = st.tabs(["Cumulative Returns", "Drawdown Analysis", "Feature Importances"])

with tab1:
    st.subheader("Cumulative Portfolio Returns")
    fig_cumulative_returns = px.line(cumulative_returns, title='Long-Short Strategy Performance')
    fig_cumulative_returns.update_layout(xaxis_title="Date", yaxis_title="Cumulative Returns")
    st.plotly_chart(fig_cumulative_returns, use_container_width=True)
    
with tab2:
    st.subheader("Portfolio Drawdown")
    fig_drawdown = px.area(daily_drawdown, title='Portfolio Drawdown Over Time', labels={'value':'Drawdown', 'index':'Date'})
    st.plotly_chart(fig_drawdown, use_container_width=True)
    
with tab3:
    st.subheader("Top 20 Model Feature Importances")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    fig_importance = px.bar(feature_importance_df, x='importance', y='feature', orientation='h', title='Top 20 Feature Importances')
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_importance, use_container_width=True)