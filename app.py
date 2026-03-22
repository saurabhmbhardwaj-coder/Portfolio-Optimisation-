import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nifty Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #7c6cf8;
        color: white;
    }
    .metric-title { font-size: 12px; color: #aaa; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 26px; font-weight: 700; color: #fff; margin: 4px 0; }
    .metric-interp { font-size: 12px; color: #ccc; margin-top: 6px; line-height: 1.5; }
    .good   { color: #4ade80; font-weight: 700; }
    .bad    { color: #f87171; font-weight: 700; }
    .neutral{ color: #fbbf24; font-weight: 700; }
    .section-header {
        background: linear-gradient(90deg, #7c6cf8, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 22px;
        font-weight: 700;
        margin: 24px 0 12px 0;
    }
    div[data-testid="stSidebar"] { background: #0f0f1a; }
    div[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ── TICKER UNIVERSE ───────────────────────────────────────────────────────────
NIFTY50 = {
    "Reliance Industries":"RELIANCE.NS","TCS":"TCS.NS","HDFC Bank":"HDFCBANK.NS",
    "Infosys":"INFY.NS","ICICI Bank":"ICICIBANK.NS","Bharti Airtel":"BHARTIARTL.NS",
    "SBI":"SBIN.NS","Hindustan Unilever":"HINDUNILVR.NS","ITC":"ITC.NS",
    "Kotak Mahindra Bank":"KOTAKBANK.NS","L&T":"LT.NS","Axis Bank":"AXISBANK.NS",
    "Asian Paints":"ASIANPAINT.NS","Bajaj Finance":"BAJFINANCE.NS","HCL Tech":"HCLTECH.NS",
    "Wipro":"WIPRO.NS","M&M":"M&M.NS","Tata Motors":"TATAMOTORS.NS",
    "NTPC":"NTPC.NS","Power Grid":"POWERGRID.NS","Sun Pharma":"SUNPHARMA.NS",
    "Titan":"TITAN.NS","UltraTech Cement":"ULTRACEMCO.NS","Nestle India":"NESTLEIND.NS",
    "Maruti Suzuki":"MARUTI.NS","Tech Mahindra":"TECHM.NS","Tata Steel":"TATASTEEL.NS",
    "JSW Steel":"JSWSTEEL.NS","ONGC":"ONGC.NS","Coal India":"COALINDIA.NS",
    "Adani Ports":"ADANIPORTS.NS","Adani Enterprises":"ADANIENT.NS",
    "Dr Reddy's":"DRREDDY.NS","Cipla":"CIPLA.NS","Bajaj Auto":"BAJAJ-AUTO.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS","Eicher Motors":"EICHERMOT.NS",
    "IndusInd Bank":"INDUSINDBK.NS","Grasim":"GRASIM.NS","BPCL":"BPCL.NS",
    "Hindalco":"HINDALCO.NS","Apollo Hospitals":"APOLLOHOSP.NS","Divi's Labs":"DIVISLAB.NS",
    "BEL":"BEL.NS","Trent":"TRENT.NS","Tata Consumer":"TATACONSUM.NS",
    "HDFC Life":"HDFCLIFE.NS","SBI Life":"SBILIFE.NS","Shriram Finance":"SHRIRAMFIN.NS",
}

BANK_NIFTY = {
    "HDFC Bank":"HDFCBANK.NS","ICICI Bank":"ICICIBANK.NS","SBI":"SBIN.NS",
    "Kotak Mahindra Bank":"KOTAKBANK.NS","Axis Bank":"AXISBANK.NS",
    "IndusInd Bank":"INDUSINDBK.NS","Bank of Baroda":"BANKBARODA.NS",
    "PNB":"PNB.NS","Federal Bank":"FEDERALBNK.NS","IDFC First Bank":"IDFCFIRSTB.NS",
    "AU Small Finance":"AUBANK.NS","Bandhan Bank":"BANDHANBNK.NS",
}

MIDCAP150 = {
    "Persistent Systems":"PERSISTENT.NS","Coforge":"COFORGE.NS","Dixon Technologies":"DIXON.NS",
    "Voltas":"VOLTAS.NS","Mphasis":"MPHASIS.NS","Tube Investments":"TIINDIA.NS",
    "Cummins India":"CUMMINSIND.NS","Bharat Forge":"BHARATFORG.NS",
    "Sundaram Finance":"SUNDARMFIN.NS","Max Healthcare":"MAXHEALTH.NS",
    "Crompton Greaves":"CROMPTON.NS","Oberoi Realty":"OBEROIRLTY.NS",
    "Kajaria Ceramics":"KAJARIACER.NS","Bata India":"BATAIND.NS",
    "Astral":"ASTRAL.NS","Polycab India":"POLYCAB.NS","KEI Industries":"KEI.NS",
    "Laurus Labs":"LAURUSLABS.NS","Alkem Laboratories":"ALKEM.NS",
    "Ajanta Pharma":"AJANTPHARM.NS","Aarti Industries":"AARTIIND.NS",
    "Deepak Nitrite":"DEEPAKNTR.NS","Fine Organic":"FINEORG.NS",
    "Navin Fluorine":"NAVINFLUOR.NS","Galaxy Surfactants":"GALAXYSURF.NS",
}

ALL_STOCKS = {}
ALL_STOCKS.update({f"[N50] {k}": v for k, v in NIFTY50.items()})
ALL_STOCKS.update({f"[BN] {k}": v for k, v in BANK_NIFTY.items()})
ALL_STOCKS.update({f"[MC] {k}": v for k, v in MIDCAP150.items()})

RISK_FREE_RATE = 0.068  # 6.8% — approx India 10Y G-Sec yield

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_prices(tickers: list, period: str) -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices.dropna(axis=1, how="all").ffill()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_performance(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    port_return = np.dot(weights, mean_returns) * 252
    port_vol    = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    sharpe      = (port_return - rf) / port_vol
    return port_return, port_vol, sharpe


def neg_sharpe(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    r, v, s = portfolio_performance(weights, mean_returns, cov_matrix, rf)
    return -s


def min_volatility(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    return portfolio_performance(weights, mean_returns, cov_matrix, rf)[1]


def optimize_portfolio(mean_returns, cov_matrix, objective="sharpe"):
    n = len(mean_returns)
    init_guess = np.array([1 / n] * n)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    fun = neg_sharpe if objective == "sharpe" else min_volatility
    result = minimize(fun, init_guess, args=(mean_returns, cov_matrix),
                      method="SLSQP", bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000})
    return result


def efficient_frontier_points(mean_returns, cov_matrix, n_points=200):
    results = {"returns": [], "volatility": [], "sharpe": []}
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, n_points)
    for target in target_returns:
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x, t=target: portfolio_performance(x, mean_returns, cov_matrix)[0] - t}]
        res = minimize(min_volatility, [1/n]*n, args=(mean_returns, cov_matrix),
                       method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500})
        if res.success:
            r, v, s = portfolio_performance(res.x, mean_returns, cov_matrix)
            results["returns"].append(r * 100)
            results["volatility"].append(v * 100)
            results["sharpe"].append(s)
    return results


def compute_ratios(returns_series: pd.Series, portfolio_returns: pd.Series,
                   benchmark_returns: pd.Series = None):
    ann_return = portfolio_returns.mean() * 252
    ann_vol    = portfolio_returns.std() * np.sqrt(252)
    sharpe     = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0

    downside   = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
    sortino    = (ann_return - RISK_FREE_RATE) / downside if downside != 0 else 0

    cum_return = (1 + portfolio_returns).cumprod()
    rolling_max = cum_return.cummax()
    drawdown    = (cum_return - rolling_max) / rolling_max
    max_dd      = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()

    beta, alpha = None, None
    treynor = None
    if benchmark_returns is not None and len(benchmark_returns) > 10:
        try:
            # Ensure both are plain float Series with DatetimeIndex
            p_series = pd.Series(portfolio_returns.values.flatten(),
                                 index=pd.to_datetime(portfolio_returns.index))
            b_series = pd.Series(benchmark_returns.values.flatten(),
                                 index=pd.to_datetime(benchmark_returns.index))
            # Align on common dates
            p_al, b_al = p_series.align(b_series, join="inner")
            p_al = p_al.dropna()
            b_al = b_al.dropna()
            # Re-align after dropna
            p_al, b_al = p_al.align(b_al, join="inner")
            if len(p_al) > 10 and len(b_al) > 10:
                p_arr = np.array(p_al, dtype=float)
                b_arr = np.array(b_al, dtype=float)
                cov_mat = np.cov(p_arr, b_arr)
                var_b   = np.var(b_arr, ddof=1)
                if var_b != 0:
                    beta    = cov_mat[0, 1] / var_b
                    b_ann   = float(b_arr.mean()) * 252
                    alpha   = (ann_return - RISK_FREE_RATE) - beta * (b_ann - RISK_FREE_RATE)
                    treynor = (ann_return - RISK_FREE_RATE) / beta if beta != 0 else None
        except Exception:
            beta, alpha, treynor = None, None, None

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "beta": beta,
        "alpha": alpha,
        "treynor": treynor,
    }


def interpret(name, value):
    if value is None:
        return "N/A — benchmark not available"
    if name == "sharpe":
        if value > 2:   return "🟢 Excellent — very high risk-adjusted return"
        if value > 1:   return "🟢 Good — solid risk-adjusted performance"
        if value > 0.5: return "🟡 Acceptable — moderate risk-adjusted return"
        return "🔴 Poor — low or negative risk-adjusted return"
    if name == "sortino":
        if value > 2:   return "🟢 Excellent — great downside protection"
        if value > 1:   return "🟢 Good — handles downside risk well"
        if value > 0.5: return "🟡 Moderate — some downside risk present"
        return "🔴 Poor — high downside risk"
    if name == "calmar":
        if value > 3:   return "🟢 Excellent — strong return vs max drawdown"
        if value > 1:   return "🟢 Good — return outpaces drawdown risk"
        if value > 0.5: return "🟡 Moderate — watch drawdowns closely"
        return "🔴 Poor — drawdown risk is high relative to return"
    if name == "max_drawdown":
        if value > -0.10: return "🟢 Low drawdown — portfolio is resilient"
        if value > -0.20: return "🟡 Moderate drawdown — manageable risk"
        if value > -0.35: return "🟠 High drawdown — significant losses in bad periods"
        return "🔴 Severe drawdown — portfolio experienced large peak-to-trough loss"
    if name == "beta":
        if value < 0.8:  return "🟢 Low beta — moves less than the market"
        if value < 1.2:  return "🟡 Market-like — similar swings to benchmark"
        return "🔴 High beta — amplifies market moves (higher risk)"
    if name == "alpha":
        if value > 0.05: return "🟢 Positive alpha — outperforming benchmark after risk adjustment"
        if value > 0:    return "🟡 Slight alpha — marginally beating benchmark"
        return "🔴 Negative alpha — underperforming benchmark on risk-adjusted basis"
    if name == "treynor":
        if value > 0.10: return "🟢 Excellent — high return per unit of market risk"
        if value > 0.05: return "🟡 Moderate — decent market-risk-adjusted return"
        return "🔴 Poor — low return for market risk taken"
    if name == "var_95":
        return f"🔴 On worst 5% of days, expect to lose ≥ {abs(value)*100:.2f}% of portfolio value"
    if name == "cvar_95":
        return f"🔴 Average loss on those worst 5% days: {abs(value)*100:.2f}% (tail risk measure)"
    return ""


def ratio_card(title, value, fmt, name):
    if value is None:
        display = "N/A"
        interp  = "Benchmark data unavailable"
    else:
        display = fmt.format(value)
        interp  = interpret(name, value)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{display}</div>
        <div class="metric-interp">{interp}</div>
    </div>""", unsafe_allow_html=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Portfolio Settings")
    st.markdown("---")

    investment = st.number_input(
        "💰 Investment Amount (₹)",
        min_value=10_000, max_value=100_000_000,
        value=1_000_000, step=10_000,
        format="%d"
    )
    st.caption(f"= ₹ {investment:,.0f}")

    period = st.selectbox("📅 Historical Period",
        ["6mo", "1y", "2y", "3y", "5y"],
        index=1,
        format_func=lambda x: {"6mo":"6 Months","1y":"1 Year","2y":"2 Years",
                                "3y":"3 Years","5y":"5 Years"}[x]
    )

    index_filter = st.multiselect(
        "🏦 Index Universe",
        ["Nifty 50", "Bank Nifty", "Nifty Midcap 150"],
        default=["Nifty 50"]
    )

    # Build filtered stock list
    available = {}
    if "Nifty 50"          in index_filter: available.update({f"[N50] {k}": v for k, v in NIFTY50.items()})
    if "Bank Nifty"        in index_filter: available.update({f"[BN] {k}":  v for k, v in BANK_NIFTY.items()})
    if "Nifty Midcap 150"  in index_filter: available.update({f"[MC] {k}":  v for k, v in MIDCAP150.items()})

    stock_names = list(available.keys())
    selected_names = st.multiselect(
        "📊 Select Stocks (min 2)",
        stock_names,
        default=stock_names[:6] if len(stock_names) >= 6 else stock_names
    )

    opt_objective = st.radio(
        "🎯 Optimization Goal",
        ["Max Sharpe Ratio", "Min Volatility"],
        index=0
    )

    run_btn = st.button("🚀 Run Optimization", use_container_width=True, type="primary")

# ── MAIN ───────────────────────────────────────────────────────────────────────
st.title("📈 Nifty Portfolio Optimizer")
st.caption("Markowitz Mean-Variance Optimization | Efficient Frontier | Risk Analytics")
st.markdown("---")

if not run_btn:
    st.info("👈  Configure your settings in the sidebar and click **Run Optimization**.")
    st.markdown("""
    ### How to use
    1. **Enter your investment amount** in INR
    2. **Choose the index universe** (Nifty 50, Bank Nifty, Midcap 150)
    3. **Select individual stocks** you want to include
    4. **Pick your optimization goal** — Max Sharpe (best risk-adjusted return) or Min Volatility (safest)
    5. Click **Run Optimization**

    ### What you'll get
    - ✅ Optimal portfolio weights and stock allocation in ₹
    - ✅ Efficient Frontier chart
    - ✅ Individual stock risk-return scatter
    - ✅ Rolling volatility chart
    - ✅ Correlation heatmap
    - ✅ All key ratios with interpretation
    """)
    st.stop()

if len(selected_names) < 2:
    st.error("Please select at least 2 stocks.")
    st.stop()

selected_tickers = [available[n] for n in selected_names]

# ── FETCH DATA ────────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching market data from Yahoo Finance…"):
    prices = fetch_prices(selected_tickers, period)

if prices.empty or prices.shape[1] < 2:
    st.error("Could not fetch enough data. Try different stocks or a longer period.")
    st.stop()

# Re-map column names back to display names
ticker_to_name = {v: k for k, v in available.items()}
prices.columns = [ticker_to_name.get(c, c) for c in prices.columns]

# Filter selected_names to only those successfully fetched
valid_names = [n for n in selected_names if n in prices.columns]
prices = prices[valid_names]

returns = compute_returns(prices)
mean_returns = returns.mean()
cov_matrix   = returns.cov()

# ── OPTIMIZE ──────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Running Markowitz optimization…"):
    obj = "sharpe" if opt_objective == "Max Sharpe Ratio" else "min_vol"
    result = optimize_portfolio(mean_returns, cov_matrix, objective=obj)
    weights = result.x
    opt_return, opt_vol, opt_sharpe = portfolio_performance(weights, mean_returns, cov_matrix)

# Benchmark: Nifty 50 index
with st.spinner("📡 Fetching Nifty 50 benchmark…"):
    bench_raw = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
    bench_prices = bench_raw["Close"].ffill() if "Close" in bench_raw else None
    bench_returns = bench_prices.pct_change().dropna() if bench_prices is not None else None

# Portfolio daily returns
port_daily = returns[valid_names].dot(weights)

# ── RATIOS ────────────────────────────────────────────────────────────────────
ratios = compute_ratios(returns, port_daily, bench_returns)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Optimal Portfolio",
    "📉 Efficient Frontier",
    "📊 Risk & Volatility",
    "🔗 Correlation",
    "📐 Ratios & Interpretation"
])

# ─── TAB 1: OPTIMAL PORTFOLIO ─────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">Optimal Portfolio Allocation</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Annual Return", f"{opt_return*100:.2f}%")
    with col2:
        st.metric("Annual Volatility (Risk)", f"{opt_vol*100:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{opt_sharpe:.3f}")
    with col4:
        st.metric("Investment", f"₹{investment:,.0f}")

    st.markdown("---")

    # Weights table
    weights_df = pd.DataFrame({
        "Stock": valid_names,
        "Weight (%)": (weights * 100).round(2),
        "Amount (₹)": (weights * investment).round(0).astype(int),
        "Shares*": ["~" for _ in valid_names],
    })
    weights_df = weights_df[weights_df["Weight (%)"] > 0.01].sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    st.subheader("Recommended Allocation")
    st.dataframe(weights_df.style.bar(subset=["Weight (%)"], color="#7c6cf8"), use_container_width=True)
    st.caption("*Approximate share count depends on current market price. Use the amount column for actual investment guidance.")

    # Pie chart
    fig_pie = px.pie(
        weights_df[weights_df["Weight (%)"] > 0.5],
        names="Stock", values="Weight (%)",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="Portfolio Weight Distribution"
    )
    fig_pie.update_traces(textposition="inside", textinfo="percent+label")
    fig_pie.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative returns comparison
    st.subheader("Cumulative Return vs Nifty 50")
    cum_port = (1 + port_daily).cumprod() * 100 - 100
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_port.index, y=cum_port.values,
        name="Optimized Portfolio", line=dict(color="#7c6cf8", width=2.5)))
    if bench_returns is not None:
        cum_bench = (1 + bench_returns).cumprod() * 100 - 100
        cum_bench = cum_bench.reindex(cum_port.index, method="ffill")
        fig_cum.add_trace(go.Scatter(x=cum_bench.index, y=cum_bench.values,
            name="Nifty 50", line=dict(color="#f97316", width=1.8, dash="dot")))
    fig_cum.update_layout(
        title="Cumulative Return (%)", yaxis_title="Return (%)", xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified"
    )
    fig_cum.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig_cum.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    st.plotly_chart(fig_cum, use_container_width=True)


# ─── TAB 2: EFFICIENT FRONTIER ────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Efficient Frontier</div>', unsafe_allow_html=True)
    st.caption("Each dot on the frontier represents a portfolio with maximum return for a given level of risk.")

    with st.spinner("Calculating efficient frontier (200 portfolios)…"):
        ef = efficient_frontier_points(mean_returns, cov_matrix, n_points=150)

    # Individual stock scatter
    stock_vols    = [returns[n].std() * np.sqrt(252) * 100 for n in valid_names]
    stock_returns = [mean_returns[n] * 252 * 100 for n in valid_names]

    fig_ef = go.Figure()

    # Frontier curve
    if ef["volatility"]:
        fig_ef.add_trace(go.Scatter(
            x=ef["volatility"], y=ef["returns"],
            mode="lines", name="Efficient Frontier",
            line=dict(color="#06b6d4", width=3),
            hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
        ))

    # Random portfolios
    n_sim = 3000
    sim_w = np.random.dirichlet(np.ones(len(valid_names)), n_sim)
    sim_r, sim_v, sim_s = [], [], []
    for w in sim_w:
        r, v, s = portfolio_performance(w, mean_returns, cov_matrix)
        sim_r.append(r * 100); sim_v.append(v * 100); sim_s.append(s)

    fig_ef.add_trace(go.Scatter(
        x=sim_v, y=sim_r, mode="markers", name="Random Portfolios",
        marker=dict(size=4, color=sim_s, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="Sharpe"), opacity=0.6),
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>"
    ))

    # Individual stocks
    fig_ef.add_trace(go.Scatter(
        x=stock_vols, y=stock_returns, mode="markers+text",
        text=[n.split("] ")[-1][:12] for n in valid_names],
        textposition="top center", name="Individual Stocks",
        marker=dict(size=10, color="#f97316", symbol="diamond"),
    ))

    # Optimal portfolio star
    fig_ef.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_return * 100],
        mode="markers+text", text=["⭐ Optimal"],
        textposition="top right", name="Optimal Portfolio",
        marker=dict(size=18, color="#facc15", symbol="star"),
    ))

    fig_ef.update_layout(
        title="Efficient Frontier — Risk vs Return",
        xaxis_title="Annual Volatility / Risk (%)",
        yaxis_title="Expected Annual Return (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="closest", legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig_ef.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    fig_ef.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown("""
    **How to read this chart:**
    - 🟦 **Frontier curve** — the set of portfolios offering the best return for each risk level
    - 🔵 **Dots** — 3000 randomly generated portfolios (colour = Sharpe ratio)
    - 🔶 **Diamonds** — individual stocks
    - ⭐ **Star** — your optimal portfolio
    """)


# ─── TAB 3: RISK & VOLATILITY ─────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Risk & Volatility Analysis</div>', unsafe_allow_html=True)

    # Rolling volatility
    window = st.slider("Rolling window (trading days)", 10, 60, 21)
    roll_vol = returns[valid_names].rolling(window).std() * np.sqrt(252) * 100

    fig_vol = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, name in enumerate(valid_names):
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol[name],
            name=name.split("] ")[-1], mode="lines",
            line=dict(width=1.5, color=colors[i % len(colors)])
        ))

    # Portfolio rolling vol
    port_roll_vol = port_daily.rolling(window).std() * np.sqrt(252) * 100
    fig_vol.add_trace(go.Scatter(
        x=port_roll_vol.index, y=port_roll_vol.values,
        name="PORTFOLIO", mode="lines",
        line=dict(width=3, color="white", dash="dash")
    ))

    fig_vol.update_layout(
        title=f"{window}-Day Rolling Annualised Volatility (%)",
        yaxis_title="Volatility (%)", xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig_vol.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.1)")
    fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.1)")
    st.plotly_chart(fig_vol, use_container_width=True)

    # Risk-Return scatter per stock
    st.subheader("Individual Stock Risk-Return Map")
    stock_df = pd.DataFrame({
        "Stock": [n.split("] ")[-1] for n in valid_names],
        "Annual Return (%)": [mean_returns[n] * 252 * 100 for n in valid_names],
        "Annual Volatility (%)": [returns[n].std() * np.sqrt(252) * 100 for n in valid_names],
        "Sharpe": [(mean_returns[n] * 252 - RISK_FREE_RATE) / (returns[n].std() * np.sqrt(252))
                   for n in valid_names],
        "Weight (%)": (weights * 100).round(2),
    })

    fig_rr = px.scatter(
        stock_df, x="Annual Volatility (%)", y="Annual Return (%)",
        text="Stock", size="Weight (%)", color="Sharpe",
        color_continuous_scale="RdYlGn", size_max=40,
        title="Risk-Return Map (bubble size = portfolio weight)"
    )
    fig_rr.update_traces(textposition="top center")
    fig_rr.add_vline(x=stock_df["Annual Volatility (%)"].mean(),
                     line_dash="dot", line_color="gray", annotation_text="Avg Vol")
    fig_rr.add_hline(y=stock_df["Annual Return (%)"].mean(),
                     line_dash="dot", line_color="gray", annotation_text="Avg Ret")
    fig_rr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rr, use_container_width=True)

    # Drawdown chart
    st.subheader("Portfolio Drawdown")
    cum = (1 + port_daily).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, fill="tozeroy",
        line=dict(color="#ef4444", width=1.5), fillcolor="rgba(239,68,68,0.2)",
        name="Drawdown"
    ))
    fig_dd.update_layout(
        title="Portfolio Drawdown (%)",
        yaxis_title="Drawdown (%)", xaxis_title="Date",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Return distribution
    st.subheader("Portfolio Daily Return Distribution")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=port_daily * 100, nbinsx=60,
        marker_color="#7c6cf8", opacity=0.8, name="Daily Returns"
    ))
    fig_hist.add_vline(x=ratios["var_95"] * 100, line_color="red", line_dash="dash",
                       annotation_text=f"VaR 95%: {ratios['var_95']*100:.2f}%")
    fig_hist.update_layout(
        title="Distribution of Daily Portfolio Returns",
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ─── TAB 4: CORRELATION ────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    st.caption("Lower correlation between stocks = better diversification = lower portfolio risk.")

    corr = returns[valid_names].corr()
    short_names = [n.split("] ")[-1][:15] for n in valid_names]
    corr.index   = short_names
    corr.columns = short_names

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="Stocks: %{x} & %{y}<br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig_corr.update_layout(
        title="Pairwise Correlation Matrix",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(400, len(valid_names) * 40)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("""
    **Interpreting correlation:**
    | Range | Meaning |
    |-------|---------|
    | +0.8 to +1.0 | Very high — stocks move almost identically |
    | +0.4 to +0.8 | Moderate positive — partially correlated |
    | -0.2 to +0.4 | Low — good diversification |
    | Below -0.2   | Negative — excellent hedge, move in opposite directions |
    """)


# ─── TAB 5: RATIOS ────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Portfolio Ratios & Interpretation</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        ratio_card("Sharpe Ratio",
                   ratios["sharpe"], "{:.3f}", "sharpe")
        ratio_card("Sortino Ratio",
                   ratios["sortino"], "{:.3f}", "sortino")
        ratio_card("Calmar Ratio",
                   ratios["calmar"], "{:.3f}", "calmar")

    with c2:
        ratio_card("Max Drawdown",
                   ratios["max_drawdown"], "{:.2%}", "max_drawdown")
        ratio_card("VaR (95%, 1-day)",
                   ratios["var_95"], "{:.2%}", "var_95")
        ratio_card("CVaR / Expected Shortfall (95%)",
                   ratios["cvar_95"], "{:.2%}", "cvar_95")

    with c3:
        ratio_card("Beta (vs Nifty 50)",
                   ratios["beta"], "{:.3f}", "beta")
        ratio_card("Alpha (Jensen's)",
                   ratios["alpha"], "{:.2%}" if ratios["alpha"] else "{}", "alpha")
        ratio_card("Treynor Ratio",
                   ratios["treynor"], "{:.4f}" if ratios["treynor"] else "{}", "treynor")

    st.markdown("---")
    st.subheader("📖 Ratio Glossary")
    with st.expander("Click to expand full definitions"):
        st.markdown("""
| Ratio | Formula | What it tells you |
|-------|---------|-------------------|
| **Sharpe Ratio** | (Return − Rf) / Volatility | Return earned per unit of total risk. Higher = better. |
| **Sortino Ratio** | (Return − Rf) / Downside Std Dev | Like Sharpe but only penalises downside risk. Higher = better. |
| **Calmar Ratio** | Annual Return / Max Drawdown | How well the portfolio recovers from its worst period. Higher = better. |
| **Max Drawdown** | (Trough − Peak) / Peak | Worst peak-to-trough loss experienced. Closer to 0 = better. |
| **VaR (95%)** | 5th percentile of daily returns | On a bad day (1-in-20), maximum expected daily loss. |
| **CVaR (95%)** | Average of returns below VaR | Average loss on those very bad days. A tail-risk measure. |
| **Beta** | Cov(portfolio, market) / Var(market) | Market sensitivity. β=1 means moves with market. |
| **Alpha (Jensen's)** | Return − [Rf + β × (Rm − Rf)] | Excess return above CAPM expectation. Positive = skill/edge. |
| **Treynor Ratio** | (Return − Rf) / Beta | Return per unit of *market* risk (unlike Sharpe which uses total risk). |
        """)

    st.markdown("---")
    # Summary scorecard
    st.subheader("📋 Quick Summary")
    scores = []
    if ratios["sharpe"] is not None:
        scores.append(("Sharpe", f"{ratios['sharpe']:.2f}", "🟢" if ratios["sharpe"] > 1 else "🟡" if ratios["sharpe"] > 0.5 else "🔴"))
    if ratios["sortino"] is not None:
        scores.append(("Sortino", f"{ratios['sortino']:.2f}", "🟢" if ratios["sortino"] > 1 else "🟡" if ratios["sortino"] > 0.5 else "🔴"))
    if ratios["max_drawdown"] is not None:
        scores.append(("Max DD", f"{ratios['max_drawdown']:.1%}", "🟢" if ratios["max_drawdown"] > -0.15 else "🟡" if ratios["max_drawdown"] > -0.30 else "🔴"))
    if ratios["beta"] is not None:
        scores.append(("Beta", f"{ratios['beta']:.2f}", "🟢" if ratios["beta"] < 0.9 else "🟡" if ratios["beta"] < 1.2 else "🔴"))
    if ratios["alpha"] is not None:
        scores.append(("Alpha", f"{ratios['alpha']:.2%}", "🟢" if ratios["alpha"] > 0 else "🔴"))

    score_cols = st.columns(len(scores))
    for col, (name, val, emoji) in zip(score_cols, scores):
        col.metric(f"{emoji} {name}", val)
