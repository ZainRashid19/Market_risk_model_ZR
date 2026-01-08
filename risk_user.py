import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt


#plotting graphs :)

st.set_page_config(page_title="Market Risk model",layout="wide")
st.title("Quantiative Risk Model (VaR & CVaR)")
st.markdown("Enter a stock ticker to calculate Downside Risk,Sortino Ratio, and Fat tail events")
st.markdown("Make smarter choices today")

def plot_graph(returns,var_cutoff_percent,symbol):
    fig, ax= plt.subplots(figsize=(12,6))

    #daily
    pos_returns = returns[returns>0]
    neg_returns= returns[returns<=0]

    ax.bar(pos_returns.index, pos_returns,color="green", alpha=0.5, label="Up days")
    ax.bar(neg_returns.index, neg_returns,color="red", alpha=0.5, label="Down days")

    #var line
    ax.axhline(y=-var_cutoff_percent, color="darkred", linestyle = "--", linewidth=2, label=f"VaR limit({-var_cutoff_percent:.1%})")

    #highlighting the failures
    failures = returns[returns<-var_cutoff_percent]
    ax.scatter(failures.index, failures,color="Black", edgecolors="Black", s=60, zorder=5, label="Failures")

    #colors
    ax.set_title("Visualizing Risk: {symbol} vs. The limit", fontsize=15, fontweight="bold")
    ax.set_ylabel("Daily Return (%)")
    ax.set_xlabel("Time frame")
    ax.axhline(y=0,color="black", linewidth=0.5) # zero line 
    ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


# BACKTESTING
def backtest(returns,position_size, confidence_level=0.95):
    st.subheader("Model Validation: Kupiec Test")
    z_score = norm.ppf(confidence_level)
    daily_vol = returns.std()
    var_cuttoff_percent = daily_vol * z_score

    actual_failures = returns[returns< -var_cuttoff_percent]
    num_failures = len(actual_failures)
    total_days = len(returns)
    num_success = total_days - num_failures

    expected_failure_rate = 1- confidence_level
    expected_success_rate = confidence_level

    actual_failure_rate = num_failures/total_days
    actual_success_rate = 1- actual_failure_rate
    expected_failures = total_days * expected_failure_rate

    st.write(f"Days Oberserved: {total_days}")

    col1,col2= st.columns(2)
    col1.metric("Expected Exceptions", f"{expected_failures:.1f} days", "Percentage of Portfolio", f"({expected_failure_rate:.1%})")
    col2.metric("Actual Expections",f"{num_failures}","Percentage of Portfolio", f"({actual_failure_rate:.1%})")
    
    try:
        #lr = likelood ratio
        if num_failures == 0:
            lr_stat = -2 * np.log(( expected_success_rate** total_days))
        
        else:
            #Probability of seeing this result if the Model is CORRECT
            numerator = (expected_success_rate**num_success) * (expected_failure_rate**num_failures)
            #Probability of seeing this result if the Reality is CORRECT
            denominator = (actual_success_rate**num_success) * (actual_failure_rate ** num_failures)
            lr_stat= -2*np.log(numerator/denominator)
        
        #critical val (chi sq distribution w 1 degree of freedom, still 95% confidence interval) 
        # cdf = cumlulative distribution function 
        p_val = 1 - chi2.cdf(lr_stat,1)

        if p_val>0.05: 
            # pass 
            st.success(f" \u2705 PASS: (P-val): {p_val:.3f}: The model is accurate.")
            st.success(f"(Failures are within the expected random range).")
        else:
            # warning sign 
            st.error(f" \u26A0 FAIL: (P-val): {p_val:.3f}: The model is inaccurate.")
            if(num_failures>expected_failures):
                st.write("(It UNDER-estimates risk. Losses happen too often).")
            else:
                st.write("(It OVER-estimates risk. You are too safe).")
        
        
    except Exception as e:
        st.error(f"Could not calculate Kupiec Test: {e}")

# assumtion of 10k stock
def calculate_market_risk(symbol, position_size_usd=10000):
    try:
        with st.spinner(f"Downloading data for {symbol}..."):
            stock = yf.Ticker(symbol)
            history = stock.history(period="1y")

        if history.empty:
            st.error (f"No data found for symbol: {symbol}")
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
    
        #Max losses based of 95% confidence interval
        cutoff = -one_day_var_percent

        worst_days=returns[returns<cutoff]

        if len(worst_days)>0:
            cvar_percent = worst_days.mean()
            cvar_dollar = position_size_usd*cvar_percent
        else:
            # Fallback if history was unusually calm
            pdf_at_z = norm.pdf(z_score)       # This is phi(z)
            cdf_at_z = norm.cdf(-z_score)      # This is (1-alpha) or 0.05
            # This is the formula: sigma * (pdf / cdf)
            cvar_percent = -daily_vol * (pdf_at_z / cdf_at_z)
            cvar_dollar = position_size_usd * cvar_percent

# -1 is the most recent data 
        current_price = history['Close'].iloc[-1]

        st.markdown(f"RISK ANALYSIS: {symbol.upper()}")

        col_head1, col_head2 = st.columns(2)

        col_head1.metric(f"Current Price: ${current_price:.2f}")
        col_head2.metric(f"Position Size: ${position_size_usd:,.2f}")
        st.divider()

        st.subheader("1. Volatility Profile")
        
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.metric("Total Volatility (Annual)", f"{annual_vol:.2%}")
            st.caption(f"Daily: {daily_vol:.2%} | Weekly: {weekly_vol:.2%}")
            
        with v2:
            st.metric("Upside Volatility (Good)", f"{annual_upside_vol:.2%}", delta="Potential", delta_color="normal")
            st.caption(f"Daily: {daily_upside_vol:.2%} | Weekly: {weekly_upside_vol:.2%}")
            
        with v3:
            st.metric("Downside Volatility (Bad)", f"{annual_downside_vol:.2%}", delta="Risk", delta_color="inverse")
            st.caption(f"Daily: {daily_downside_vol:.2%} | Weekly: {weekly_downside_vol:.2%}")
        
        # Comparison Logic (The "Verdict")
        if annual_downside_vol < annual_vol:
            st.success(f"✅ **Good News:** 'Bad' volatility ({annual_downside_vol:.2%}) is LOWER than total volatility.\n\nThis stock surges up more violently than it crashes down.")
        else:
            st.warning(f"⚠️ **Warning:** 'Bad' volatility ({annual_downside_vol:.2%}) is HIGHER than total volatility.\n\nThis stock crashes harder than it rallies.")
            
        st.divider()


        # Sortino Ratio 

        st.subheader("2. Efficiency (Sortino Ratio)")

        if sortino_ratio >2:
            s_label = "EXCELLENT"
            s_color = "normal" #normal = green
            s_msg = "High returns to low bad risk"

        elif sortino_ratio>1:
            s_label = "GOOD"
            s_color = "normal"
            s_msg = "You are getting paid for your risk"
        else:
            s_label = "CAUTION"
            s_color = "inverse" 
            s_msg= "Low returns compared to the downside risk"

        st.metric("Sortino Ratio", f"{sortino_ratio:>2f}", delta = s_label,
                delta_color="normal"
                if sortino_ratio>1 else "inverse")
                
        st.info(s_msg)
        st.divider()

        st.subheader("3. Risk Scenarios (95% Confidence)")
        
        r1, r2 = st.columns(2)
        # VaR Display
        r1.error(f"**Normal Worst Case (VaR)**\n\nLimit Loss: **${one_day_var_dollar:,.2f}**\n\n({one_day_var_percent:.2%})")
        
        # CVaR Display
        r2.error(f"**True Disaster (CVaR)**\n\nExpected Crash: **${abs(cvar_dollar):,.2f}**\n\n({abs(cvar_percent):.2%})")
        
        # Fat Tail Logic
        ratio = abs(cvar_dollar / one_day_var_dollar)
        
        if ratio > 1.3:
            st.warning(f"⚠️ **FAT TAIL WARNING:** Huge Crash Risk.\n\nWhen this stock fails, it fails CATASTROPHICALLY. The crash is **{((ratio-1)*100):.0f}%** worse than predicted.")
        else:
            st.success(f"✅ **Normal Risk:** Crashes are predictable. The crash is close to the prediction.")

        st.divider()

        backtest(returns,confidence_level)
        plot_graph(returns,one_day_var_percent,symbol)

    except Exception as e:
        print(f"Error calculating risk for {symbol}: {e}")

with st.sidebar:
    st.header("⚙️ Settings")
    input_symbol = st.text_input("Stock Ticker", value="NVDA").upper()
    input_investment = st.number_input("Investment Amount ($)", value=10000)
    
    if st.button("Run Risk Model"):
        calculate_market_risk(input_symbol, input_investment)








