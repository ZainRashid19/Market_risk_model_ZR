import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="testing_Market Risk Model", layout="wide")
st.title("🛡️ Quantitative Risk Model (VaR & CVaR)")
st.markdown("Enter a stock ticker to calculate Downside Risk, Sortino Ratio, and Fat Tail events.")

# --- 1. CACHED DATA DOWNLOAD (The Speed Boost) ---
@st.cache_data
def get_stock_data(symbol):
    """Downloads data and caches it so the app is fast."""
    stock = yf.Ticker(symbol)
    history = stock.history(period="1y")
    return history

# --- 2. PLOTTING FUNCTION ---
def plot_graph(returns, var_cutoff_percent, symbol):
    fig, ax = plt.subplots(figsize=(12, 6))

    pos_returns = returns[returns > 0]
    neg_returns = returns[returns <= 0]

    ax.bar(pos_returns.index, pos_returns, color="green", alpha=0.5, label="Up Days")
    ax.bar(neg_returns.index, neg_returns, color="red", alpha=0.5, label="Down Days")
    ax.axhline(y=-var_cutoff_percent, color="darkred", linestyle="--", linewidth=2, label=f"VaR Limit ({-var_cutoff_percent:.1%})")

    failures = returns[returns < -var_cutoff_percent]
    ax.scatter(failures.index, failures, color="black", edgecolors="black", s=60, zorder=5, label="Failures")

    ax.set_title(f"Visualizing Risk: {symbol} vs. The Limit", fontsize=15, fontweight="bold")
    ax.set_ylabel("Daily Return (%)")
    ax.axhline(y=0, color="black", linewidth=0.5) 
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# --- 3. BACKTESTING FUNCTION ---
def backtest(returns, confidence_level=0.95):
    st.subheader("🔍 Model Validation: Kupiec Test")
    
    z_score = norm.ppf(confidence_level)
    daily_vol = returns.std()
    var_cutoff_percent = daily_vol * z_score

    actual_failures = returns[returns < -var_cutoff_percent]
    num_failures = len(actual_failures)
    total_days = len(returns)
    expected_failure_rate = 1 - confidence_level
    expected_failures = total_days * expected_failure_rate

    st.write(f"**Days Observed:** {total_days}")
    col1, col2 = st.columns(2)
    col1.metric("Expected Exceptions", f"{expected_failures:.1f}")
    col2.metric("Actual Exceptions", f"{num_failures}")

# --- 4. MAIN CALCULATION ---
def calculate_market_risk(symbol, position_size_usd):
    try:
        with st.spinner(f"Analyzing {symbol}..."):
            # Use the cached function!
            history = get_stock_data(symbol)

            if history.empty:
                st.error(f"❌ No data found for {symbol}")
                return

            #Volatility formula
        history['Log_Returns'] = np.log(history['Close']/history['Close'].shift(1))
        returns = history['Log_Returns'].dropna()

        daily_vol = returns.std()
        weekly_vol = daily_vol * np.sqrt(5)
        annual_vol = daily_vol * np.sqrt(252)
        # 252 is std # of trading days in the US Stock market

        # calculating the neg returns of the stock
        negative_returns = returns.copy()
        #ignores pos days 
        negative_returns[negative_returns>0]=0
        daily_downside_vol = np.sqrt(np.mean(negative_returns**2))
        weekly_downside_vol = daily_downside_vol * np.sqrt(5)
        annual_downside_vol = daily_downside_vol * np.sqrt(252)


        # calculating the pos returns of the stock
        positive_returns = returns.copy()
        #ignore the neg days 
        positive_returns[positive_returns<0]=0
        daily_upside_vol = np.sqrt(np.mean(positive_returns**2))
        weekly_upside_vol = daily_upside_vol * np.sqrt(5)
        annual_upside_vol = daily_upside_vol * np.sqrt(252)

        #Sortino ratio 

        avg_daily_returns = returns.mean()
        annual_returns = avg_daily_returns*252
        sortino_ratio = annual_returns/annual_downside_vol
        
        #95% confidence or 1.645 one tailed  z score 
        confidence_level = 0.95 
        z_score = norm.ppf(confidence_level)
        
        #VaR's
        one_day_var_percent = daily_vol * z_score
        one_day_var_dollar = position_size_usd * one_day_var_percent

        one_day_var_percent_gain = daily_upside_vol * z_score
        one_day_var_dollar_gain = position_size_usd * one_day_var_percent_gain 
            cutoff = -one_day_var_percent
            worst_days = returns[returns < cutoff]
            if len(worst_days) > 0:
                cvar_percent = worst_days.mean()
            else:
                cvar_percent = -daily_vol * (norm.pdf(z_score) / norm.cdf(-z_score))
            cvar_dollar = position_size_usd * cvar_percent

            # --- DISPLAY ---
            current_price = history['Close'].iloc[-1]
            st.markdown(f"### RISK ANALYSIS: {symbol.upper()}")
            
            # Metrics
            c1, c2 = st.columns(2)
            c1.metric("Current Price", f"${current_price:,.2f}")
            c2.metric("Position Size", f"${position_size_usd:,.2f}")
            
            st.divider()
            
            # Volatility
            st.subheader("1. Volatility Profile")
            v1, v2, v3 = st.columns(3)
            v1.metric("Total Volatility", f"{annual_vol:.2%}")
            v2.metric("Upside Volatility", f"{annual_upside_vol:.2%}", delta="Potential")
            v3.metric("Downside Volatility", f"{annual_downside_vol:.2%}", delta="-Risk", delta_color="inverse")
            
            st.divider()

            # Risk Scenarios
            st.subheader("2. Risk Scenarios (95% Confidence)")
            r1, r2 = st.columns(2)
            r1.error(f"**VaR (Limit Loss)**\n\n${one_day_var_dollar:,.2f}")
            r2.error(f"**CVaR (Expected Crash)**\n\n${abs(cvar_dollar):,.2f}")

            # Plot
            plot_graph(returns, one_day_var_percent, symbol)
            backtest(returns)

    except Exception as e:
        st.error(f"Error: {e}")

# --- 5. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ Settings")
    input_symbol = st.text_input("Stock Ticker", value="NVDA").upper()
    input_investment = st.number_input("Investment Amount ($)", value=10000)
    run_btn = st.button("Run Risk Model")

if run_btn:
    calculate_market_risk(input_symbol, input_investment)
