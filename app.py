import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

MAX_STOCKS    = 15
RISK_FREE_RATE = 0.068   # India 10-Y G-Sec
TRADING_DAYS   = 252

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optimisation Solutions — Portfolio Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── HIDE SIDEBAR ── */
    [data-testid="collapsedControl"]  { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    /* ── APP BACKGROUND: rich layered blue ── */
    .stApp {
        background: linear-gradient(160deg, #020b18 0%, #051830 35%, #072a52 65%, #0a1f3d 100%);
        min-height: 100vh;
    }
    .main .block-container { padding-top: 1.5rem; }

    /* ── HEADER ── */
    .brand-title {
        text-align: center; font-size: 42px; font-weight: 900;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px; letter-spacing: 2px;
        text-shadow: 0 0 40px rgba(56,189,248,0.3);
    }
    .brand-sub {
        text-align: center; font-size: 12px; color: #f5e6c8;
        color: #f0deb4;
        margin-bottom: 24px; letter-spacing: 4px; text-transform: uppercase;
        opacity: 0.85;
    }
    .brand-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, #06b6d4, transparent);
        margin: 0 auto 24px auto; width: 60%; border-radius: 2px;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: linear-gradient(135deg, #0c2340 0%, #0e2d52 100%);
        border-radius: 14px; padding: 16px 20px; margin-bottom: 12px;
        border-left: 4px solid #38bdf8;
        border-top: 1px solid rgba(56,189,248,0.2);
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(56,189,248,0.1);
    }
    .metric-title  {
        font-size: 10px; color: #f5e6c8; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.5px;
    }
    .metric-value  { font-size: 22px; font-weight: 800; color: #fdf6e3; margin: 6px 0 4px; }
    .metric-interp { font-size: 12px; color: #e8d5a0; line-height: 1.6; }

    /* ── LEARN CARDS ── */
    .learn-card {
        background: linear-gradient(135deg, #0c1f38 0%, #0f2847 100%);
        border-radius: 12px; padding: 16px 20px; margin-bottom: 10px;
        border-left: 4px solid #06b6d4;
        border-top: 1px solid rgba(6,182,212,0.2);
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    }
    .learn-title   { font-size: 15px; font-weight: 800; color: #fde68a; margin-bottom: 6px; }
    .learn-formula {
        font-size: 12px; color: #fde68a; font-family: monospace;
        background: rgba(0,0,0,0.4); padding: 5px 12px; border-radius: 6px;
        margin: 6px 0 8px; display: inline-block;
        border: 1px solid rgba(253,230,138,0.25);
    }
    .learn-desc    { font-size: 13px; color: #f0deb4; line-height: 1.65; }

    /* ── SECTION HEADERS ── */
    .section-hdr {
        font-size: 22px; font-weight: 800; margin: 20px 0 14px 0;
        background: linear-gradient(90deg, #fde68a, #f0abfc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(5,24,48,0.8); border-radius: 12px;
        padding: 4px; gap: 4px;
        border: 1px solid rgba(56,189,248,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; border-radius: 8px;
        color: #f5e6c8; font-weight: 600; font-size: 14px;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e40af, #0e7490) !important;
        color: #fdf6e3 !important;
        box-shadow: 0 2px 12px rgba(56,189,248,0.3);
    }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(90deg, #1d4ed8, #0891b2);
        color: white; border: 1px solid rgba(56,189,248,0.4);
        border-radius: 10px; font-weight: 700; font-size: 15px;
        box-shadow: 0 4px 16px rgba(29,78,216,0.4);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2563eb, #0ea5e9);
        box-shadow: 0 6px 20px rgba(56,189,248,0.5);
        transform: translateY(-1px);
    }

    /* ── INPUT LABELS ── */
    label, .stMarkdown p strong { color: #f5e6c8 !important; }

    /* ── DIVIDER ── */
    hr { border-color: rgba(56,189,248,0.15) !important; }

    /* ── METRICS (st.metric) ── */
    [data-testid="stMetricValue"]  { color: #fdf6e3 !important; font-weight: 800; }
    [data-testid="stMetricLabel"]  { color: #f5e6c8 !important; }

    /* ── DATAFRAME ── */
    [data-testid="stDataFrame"] { border: 1px solid rgba(56,189,248,0.2); border-radius: 10px; }

    /* ── SLIDERS ── */
    [data-testid="stSlider"] > div > div > div > div { background: #1d4ed8 !important; }

    /* ── INFO / WARNING / ERROR BOXES ── */
    [data-testid="stAlert"] { border-radius: 10px; }

    /* ── CAPTION ── */
    .stCaption, small { color: #e8d5a0 !important; }

    /* ── CHART CONTAINER ── */
    [data-testid="stPlotlyChart"] > div {
        border-radius: 12px;
        border: 1px solid rgba(56,189,248,0.15);
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ── TICKER LISTS ──────────────────────────────────────────────────────────────
NIFTY50 = {
    "Adani Enterprises":"ADANIENT.NS","Adani Ports":"ADANIPORTS.NS",
    "Apollo Hospitals":"APOLLOHOSP.NS","Asian Paints":"ASIANPAINT.NS",
    "Axis Bank":"AXISBANK.NS","Bajaj Auto":"BAJAJ-AUTO.NS",
    "Bajaj Finance":"BAJFINANCE.NS","Bajaj Finserv":"BAJAJFINSV.NS",
    "BPCL":"BPCL.NS","Bharti Airtel":"BHARTIARTL.NS",
    "Britannia":"BRITANNIA.NS","Cipla":"CIPLA.NS",
    "Coal India":"COALINDIA.NS","Divi's Labs":"DIVISLAB.NS",
    "Dr Reddy's":"DRREDDY.NS","Eicher Motors":"EICHERMOT.NS",
    "Grasim":"GRASIM.NS","HCL Tech":"HCLTECH.NS",
    "HDFC Bank":"HDFCBANK.NS","HDFC Life":"HDFCLIFE.NS",
    "Hero MotoCorp":"HEROMOTOCO.NS","Hindalco":"HINDALCO.NS",
    "Hindustan Unilever":"HINDUNILVR.NS","ICICI Bank":"ICICIBANK.NS",
    "ITC":"ITC.NS","IndusInd Bank":"INDUSINDBK.NS",
    "Infosys":"INFY.NS","JSW Steel":"JSWSTEEL.NS",
    "Kotak Mahindra Bank":"KOTAKBANK.NS","L&T":"LT.NS",
    "LTIMindtree":"LTIM.NS","M&M":"M&M.NS",
    "Maruti Suzuki":"MARUTI.NS","Nestle India":"NESTLEIND.NS",
    "NTPC":"NTPC.NS","ONGC":"ONGC.NS",
    "Power Grid":"POWERGRID.NS","Reliance Industries":"RELIANCE.NS",
    "SBI":"SBIN.NS","SBI Life":"SBILIFE.NS",
    "Shriram Finance":"SHRIRAMFIN.NS","Sun Pharma":"SUNPHARMA.NS",
    "TCS":"TCS.NS","Tata Consumer":"TATACONSUM.NS",
    "Tata Motors":"TATAMOTORS.NS","Tata Steel":"TATASTEEL.NS",
    "Tech Mahindra":"TECHM.NS","Titan":"TITAN.NS",
    "Trent":"TRENT.NS","UltraTech Cement":"ULTRACEMCO.NS","Wipro":"WIPRO.NS",
}

BANK_NIFTY = {
    "AU Small Finance Bank":"AUBANK.NS","Axis Bank":"AXISBANK.NS",
    "Bandhan Bank":"BANDHANBNK.NS","Bank of Baroda":"BANKBARODA.NS",
    "Bank of India":"BANKINDIA.NS","Bank of Maharashtra":"MAHABANK.NS",
    "Canara Bank":"CANBK.NS","City Union Bank":"CUB.NS",
    "CSB Bank":"CSBBANK.NS","DCB Bank":"DCBBANK.NS",
    "Federal Bank":"FEDERALBNK.NS","HDFC Bank":"HDFCBANK.NS",
    "ICICI Bank":"ICICIBANK.NS","IDBI Bank":"IDBI.NS",
    "IDFC First Bank":"IDFCFIRSTB.NS","Indian Bank":"INDIANB.NS",
    "Indian Overseas Bank":"IOB.NS","IndusInd Bank":"INDUSINDBK.NS",
    "Karnataka Bank":"KTKBANK.NS","Karur Vysya Bank":"KARURVYSYA.NS",
    "Kotak Mahindra Bank":"KOTAKBANK.NS","Punjab National Bank":"PNB.NS",
    "RBL Bank":"RBLBANK.NS","SBI":"SBIN.NS",
    "South Indian Bank":"SOUTHBANK.NS","UCO Bank":"UCOBANK.NS",
    "Union Bank of India":"UNIONBANK.NS","Yes Bank":"YESBANK.NS",
}

MIDCAP150 = {
    "Aarti Industries":"AARTIIND.NS","Ajanta Pharma":"AJANTPHARM.NS",
    "Alkem Laboratories":"ALKEM.NS","Apollo Tyres":"APOLLOTYRE.NS",
    "Ashok Leyland":"ASHOKLEY.NS","Astral":"ASTRAL.NS",
    "Balkrishna Industries":"BALKRISIND.NS","Bata India":"BATAIND.NS",
    "Bharat Dynamics":"BDL.NS","Bharat Forge":"BHARATFORG.NS",
    "Blue Star":"BLUESTARCO.NS","BSE":"BSE.NS",
    "Can Fin Homes":"CANFINHOME.NS","CG Power":"CGPOWER.NS",
    "Cholamandalam Investment":"CHOLAFIN.NS","Coforge":"COFORGE.NS",
    "Crompton Greaves":"CROMPTON.NS","Cummins India":"CUMMINSIND.NS",
    "Deepak Nitrite":"DEEPAKNTR.NS","Dixon Technologies":"DIXON.NS",
    "Emami":"EMAMILTD.NS","Endurance Technologies":"ENDURANCE.NS",
    "Exide Industries":"EXIDEIND.NS","Fine Organic":"FINEORG.NS",
    "Galaxy Surfactants":"GALAXYSURF.NS","Gujarat Fluorochemicals":"FLUOROCHEM.NS",
    "Gujarat Gas":"GUJGASLTD.NS","Happiest Minds":"HAPPSTMNDS.NS",
    "HDFC AMC":"HDFCAMC.NS","Indian Hotels":"INDHOTEL.NS",
    "J B Chemicals":"JBCHEPHARM.NS","Kajaria Ceramics":"KAJARIACER.NS",
    "KEI Industries":"KEI.NS","Laurus Labs":"LAURUSLABS.NS",
    "LIC Housing Finance":"LICHSGFIN.NS","Lupin":"LUPIN.NS",
    "Mahindra Finance":"M&MFIN.NS","Manappuram Finance":"MANAPPURAM.NS",
    "Max Healthcare":"MAXHEALTH.NS","MCX":"MCX.NS",
    "Mphasis":"MPHASIS.NS","MRF":"MRF.NS",
    "Navin Fluorine":"NAVINFLUOR.NS","Nippon Life AMC":"NAM-INDIA.NS",
    "Oberoi Realty":"OBEROIRLTY.NS","Oracle Financial":"OFSS.NS",
    "Persistent Systems":"PERSISTENT.NS","PI Industries":"PIIND.NS",
    "Polycab India":"POLYCAB.NS","Prestige Estates":"PRESTIGE.NS",
    "PVR Inox":"PVRINOX.NS","Radico Khaitan":"RADICO.NS",
    "REC":"RECLTD.NS","Solar Industries":"SOLARINDS.NS",
    "Star Health":"STARHEALTH.NS","Sundaram Finance":"SUNDARMFIN.NS",
    "Supreme Industries":"SUPREMEIND.NS","Syngene International":"SYNGENE.NS",
    "Tata Chemicals":"TATACHEM.NS","Tata Communications":"TATACOMM.NS",
    "Tata Elxsi":"TATAELXSI.NS","Thermax":"THERMAX.NS",
    "Torrent Pharma":"TORNTPHARM.NS","Torrent Power":"TORNTPOWER.NS",
    "Tube Investments":"TIINDIA.NS","V-Guard Industries":"VGUARD.NS",
    "Varun Beverages":"VBL.NS","Vinati Organics":"VINATIORGA.NS",
    "Voltas":"VOLTAS.NS","Zydus Lifesciences":"ZYDUSLIFE.NS",
}

ALL_OPTIONS = (
    [f"[Nifty 50] {k}"   for k in NIFTY50] +
    [f"[Bank Nifty] {k}" for k in BANK_NIFTY] +
    [f"[Midcap 150] {k}" for k in MIDCAP150]
)

def get_ticker(name: str):
    if name.startswith("[Nifty 50] "):    return NIFTY50.get(name[11:])
    if name.startswith("[Bank Nifty] "): return BANK_NIFTY.get(name[13:])
    if name.startswith("[Midcap 150] "): return MIDCAP150.get(name[13:])
    return None

def short_name(name: str) -> str:
    for pre in ["[Nifty 50] ", "[Bank Nifty] ", "[Midcap 150] "]:
        if name.startswith(pre):
            return name[len(pre):]
    return name

# ── PERIOD OPTIONS ────────────────────────────────────────────────────────────
PERIOD_OPTIONS, PERIOD_LABELS = [], []
for m in [3, 6, 9]:
    PERIOD_OPTIONS.append(f"{m}mo")
    PERIOD_LABELS.append(f"{m} Months")
for y in range(1, 26):
    PERIOD_OPTIONS.append(f"{y}y")
    PERIOD_LABELS.append(f"{y} Year{'s' if y > 1 else ''}")

# ── ROBUST MATH HELPERS ───────────────────────────────────────────────────────

def safe_float(x, default=np.nan):
    """Convert anything to float safely."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default

def safe_std(series, min_obs=5, annualise=False):
    try:
        s = series.dropna()
        if len(s) < min_obs:
            return np.nan
        v = float(s.std())
        return v * np.sqrt(TRADING_DAYS) if annualise else v
    except Exception:
        return np.nan

def safe_mean(series, annualise=False):
    try:
        s = series.dropna()
        if len(s) < 2:
            return np.nan
        v = float(s.mean())
        return v * TRADING_DAYS if annualise else v
    except Exception:
        return np.nan


# ── DATA FETCH ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_prices(tickers: tuple, period: str) -> pd.DataFrame:
    """Download OHLCV and return a clean Close price DataFrame."""
    try:
        raw = yf.download(list(tickers), period=period,
                          auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"].copy()
        else:
            prices = raw[["Close"]].copy()
            if len(tickers) == 1:
                prices.columns = list(tickers)
        prices = prices.dropna(axis=1, how="all").ffill().bfill()
        # Drop columns that are still all-NaN or constant
        prices = prices.loc[:, prices.std() > 0]
        return prices
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return pd.DataFrame()

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, cleaned."""
    try:
        r = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
        # Drop any column with >20% missing after the first row
        threshold = int(len(r) * 0.20)
        r = r.dropna(axis=1, thresh=len(r) - threshold)
        r = r.fillna(0)   # fill any remaining NaN with 0 (no change)
        return r
    except Exception:
        return pd.DataFrame()


# ── PORTFOLIO MATH ────────────────────────────────────────────────────────────

def port_perf(w: np.ndarray, mr: np.ndarray, cov: np.ndarray,
              rf: float = RISK_FREE_RATE):
    """Return (ann_return, ann_vol, sharpe). Never raises."""
    try:
        w  = np.asarray(w, dtype=float)
        r  = float(np.dot(w, mr)) * TRADING_DAYS
        v2 = float(w @ cov @ w)
        v  = np.sqrt(max(v2, 0.0)) * np.sqrt(TRADING_DAYS)
        s  = (r - rf) / v if v > 1e-10 else 0.0
        return r, v, s
    except Exception:
        return 0.0, 1.0, 0.0


# Objective functions — explicit, no lambdas (Python 3.14 / SciPy compat)
def _obj_neg_sharpe(w, mr, cov):
    _, _, s = port_perf(w, mr, cov)
    return -s

def _obj_min_vol(w, mr, cov):
    _, v, _ = port_perf(w, mr, cov)
    return v

def _con_sum_one(w):
    return float(np.sum(w)) - 1.0


def optimize_portfolio(mr: pd.Series, cov: pd.DataFrame,
                       objective: str = "sharpe") -> np.ndarray:
    """
    Run Markowitz optimisation. Returns weight array.
    Falls back to equal weights if optimiser fails.
    """
    n    = len(mr)
    mr_a = mr.values.astype(float)
    cv_a = cov.values.astype(float)

    # Make covariance PSD
    cv_a = (cv_a + cv_a.T) / 2
    eigvals = np.linalg.eigvalsh(cv_a)
    if eigvals.min() < 0:
        cv_a -= (eigvals.min() - 1e-8) * np.eye(n)

    x0  = np.full(n, 1.0 / n)
    bnd = [(0.0, 1.0)] * n
    con = [{"type": "eq", "fun": _con_sum_one}]
    obj = _obj_neg_sharpe if objective == "sharpe" else _obj_min_vol

    best_w, best_val = x0.copy(), np.inf

    # Try multiple starting points for robustness
    starts = [x0]
    rng    = np.random.default_rng(42)
    for _ in range(5):
        s = rng.dirichlet(np.ones(n))
        starts.append(s)

    for s0 in starts:
        try:
            res = minimize(obj, s0, args=(mr_a, cv_a),
                           method="SLSQP", bounds=bnd, constraints=con,
                           options={"maxiter": 2000, "ftol": 1e-9})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_w   = res.x.copy()
        except Exception:
            continue

    # Clip and renormalise
    best_w = np.clip(best_w, 0.0, 1.0)
    total  = best_w.sum()
    if total < 1e-10:
        best_w = x0
    else:
        best_w /= total
    return best_w


def efficient_frontier(mr: pd.Series, cov: pd.DataFrame,
                       n_points: int = 120) -> dict:
    """Compute efficient frontier. Returns dict with lists."""
    result = {"returns": [], "volatility": [], "sharpe": []}
    n    = len(mr)
    mr_a = mr.values.astype(float)
    cv_a = cov.values.astype(float)
    cv_a = (cv_a + cv_a.T) / 2
    eigvals = np.linalg.eigvalsh(cv_a)
    if eigvals.min() < 0:
        cv_a -= (eigvals.min() - 1e-8) * np.eye(n)

    r_min = float(mr_a.min()) * TRADING_DAYS
    r_max = float(mr_a.max()) * TRADING_DAYS
    bnd   = [(0.0, 1.0)] * n
    x0    = np.full(n, 1.0 / n)

    for target in np.linspace(r_min, r_max, n_points):
        _t = float(target)
        def _ret_eq(w, _mr=mr_a, _cov=cv_a, _t=_t):
            return float(np.dot(w, _mr)) * TRADING_DAYS - _t

        cons = [
            {"type": "eq", "fun": _con_sum_one},
            {"type": "eq", "fun": _ret_eq},
        ]
        try:
            res = minimize(_obj_min_vol, x0.copy(), args=(mr_a, cv_a),
                           method="SLSQP", bounds=bnd, constraints=cons,
                           options={"maxiter": 500, "ftol": 1e-8})
            if res.success:
                rv, vv, sv = port_perf(res.x, mr_a, cv_a)
                result["returns"].append(rv * 100)
                result["volatility"].append(vv * 100)
                result["sharpe"].append(sv)
        except Exception:
            continue
    return result


def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    try:
        ts = np.asarray(ts, dtype=float)
        ts = ts[np.isfinite(ts)]
        if len(ts) < max_lag * 2:
            return np.nan
        lags = range(2, min(max_lag, len(ts) // 2))
        tau  = [float(np.std(np.subtract(ts[l:], ts[:-l]))) for l in lags]
        tau  = [t for t in tau if t > 0]
        if len(tau) < 2:
            return np.nan
        lags_used = list(range(2, 2 + len(tau)))
        poly = np.polyfit(np.log(lags_used), np.log(tau), 1)
        return float(poly[0])
    except Exception:
        return np.nan


def compute_ratios(port_ret: pd.Series,
                   bench_ret: pd.Series | None = None) -> dict:
    """
    Compute all 34 ratios. Every calculation is wrapped in try/except.
    Returns a dict — all values are either float or None.
    """
    def _s(v):
        return safe_float(v)

    r = port_ret.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 10:
        return {k: None for k in [
            "ann_return","ann_excess","ann_outperf","ann_abnormal","win_days",
            "rf","ann_var","ann_vol","beta","sys_risk","unsys_risk","te",
            "var_ann","cvar_95","downside_dev","semidev","max_dd","avg_dd",
            "skewness","ex_kurtosis","tail_ratio","omega","common_sense",
            "exp_return","alpha","sharpe","treynor","sortino","info_ratio",
            "idio_vol","appraisal","sterling","calmar","hurst"]}

    rf = RISK_FREE_RATE

    # ── Returns ──────────────────────────────────────────────────────────────
    try: ann_ret = _s(r.mean() * TRADING_DAYS)
    except: ann_ret = None

    try: ann_vol = _s(r.std() * np.sqrt(TRADING_DAYS))
    except: ann_vol = None

    try: ann_var = _s(r.var() * TRADING_DAYS)
    except: ann_var = None

    try: win_days = _s((r > 0).mean())
    except: win_days = None

    try: exp_ret = _s(r.mean() * TRADING_DAYS)
    except: exp_ret = None

    # ── Drawdown ─────────────────────────────────────────────────────────────
    try:
        cum     = (1 + r).cumprod()
        roll_mx = cum.cummax()
        dd_ser  = (cum - roll_mx) / roll_mx
        max_dd  = _s(dd_ser.min())
        neg_dd  = dd_ser[dd_ser < 0]
        avg_dd  = _s(neg_dd.mean()) if len(neg_dd) > 0 else 0.0
    except:
        max_dd = avg_dd = None

    # ── Downside / semi ──────────────────────────────────────────────────────
    try:
        neg_r      = r[r < 0]
        downside   = _s(neg_r.std() * np.sqrt(TRADING_DAYS)) if len(neg_r) > 1 else None
    except: downside = None

    try:
        below_mean = r[r < float(r.mean())]
        semidev    = _s(below_mean.std() * np.sqrt(TRADING_DAYS)) if len(below_mean) > 1 else None
    except: semidev = None

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    try:
        var95   = _s(np.percentile(r, 5))
        cvar95  = _s(r[r <= var95].mean()) if (r <= var95).any() else None
        var_ann = _s(var95 * np.sqrt(TRADING_DAYS))
    except: var95 = cvar95 = var_ann = None

    # ── Ratios ───────────────────────────────────────────────────────────────
    try:
        sharpe = _s((ann_ret - rf) / ann_vol) if ann_vol and ann_vol > 1e-10 else None
    except: sharpe = None

    try:
        sortino = _s((ann_ret - rf) / downside) if downside and downside > 1e-10 else None
    except: sortino = None

    try:
        calmar = _s(ann_ret / abs(max_dd)) if max_dd and abs(max_dd) > 1e-10 else None
    except: calmar = None

    try:
        sterling = _s(ann_ret / abs(avg_dd)) if avg_dd and abs(avg_dd) > 1e-10 else None
    except: sterling = None

    try:
        pos_r = r[r > 0]; neg_r2 = r[r < 0]
        omega = _s(pos_r.sum() / abs(neg_r2.sum())) \
                if len(neg_r2) > 0 and neg_r2.sum() != 0 else None
    except: omega = None

    try:
        p95 = float(np.percentile(r, 95))
        p05 = float(np.percentile(r, 5))
        tail_ratio = _s(abs(p95) / abs(p05)) if abs(p05) > 1e-10 else None
    except: tail_ratio = None

    try:
        common_sense = _s(omega * tail_ratio) \
                       if omega is not None and tail_ratio is not None else None
    except: common_sense = None

    try:
        sk = _s(float(skew(r)))
    except: sk = None

    try:
        ek = _s(float(kurtosis(r, fisher=True)))
    except: ek = None

    h = hurst_exponent(r.values)

    # ── Beta / Alpha / benchmark metrics ─────────────────────────────────────
    beta = alpha = treynor = info_r = te = idio = appr = None
    sys_r = unsys_r = ann_exc = ann_out = ann_abn = None

    if bench_ret is not None:
        try:
            ps = pd.Series(
                np.asarray(r.values, dtype=float).flatten(),
                index=pd.to_datetime(r.index)
            )
            bs = pd.Series(
                np.asarray(bench_ret.values, dtype=float).flatten(),
                index=pd.to_datetime(bench_ret.index)
            )
            pa, ba = ps.align(bs, join="inner")
            pa = pa.replace([np.inf, -np.inf], np.nan).dropna()
            ba = ba.replace([np.inf, -np.inf], np.nan).dropna()
            pa, ba = pa.align(ba, join="inner")

            if len(pa) >= 20:
                pv = np.asarray(pa, dtype=float)
                bv = np.asarray(ba, dtype=float)
                vb = float(np.var(bv, ddof=1))
                if vb > 1e-10:
                    cm      = np.cov(pv, bv)
                    beta    = _s(cm[0, 1] / vb)
                    b_ann   = float(bv.mean()) * TRADING_DAYS
                    if ann_ret is not None and beta is not None:
                        alpha   = _s((ann_ret - rf) - beta * (b_ann - rf))
                        treynor = _s((ann_ret - rf) / beta) if abs(beta) > 1e-10 else None
                        sys_r   = _s(beta ** 2 * vb * TRADING_DAYS)
                        unsys_r = _s(ann_var - sys_r) if ann_var is not None and sys_r is not None else None
                        diff    = pv - bv
                        te_val  = float(diff.std()) * np.sqrt(TRADING_DAYS)
                        te      = _s(te_val)
                        info_r  = _s((ann_ret - b_ann) / te_val) if te_val > 1e-10 else None
                        idio    = _s(np.sqrt(max(float(unsys_r), 0.0))) if unsys_r is not None else None
                        appr    = _s(alpha / idio) if (
                            alpha is not None and idio is not None and abs(idio) > 1e-10
                        ) else None
                        ann_exc  = _s(ann_ret - rf)
                        ann_out  = _s(ann_ret - b_ann)
                        ann_abn  = alpha
        except Exception:
            pass

    return dict(
        ann_return=ann_ret, ann_excess=ann_exc, ann_outperf=ann_out,
        ann_abnormal=ann_abn, win_days=win_days, rf=rf,
        ann_var=ann_var, ann_vol=ann_vol, beta=beta,
        sys_risk=sys_r, unsys_risk=unsys_r, te=te,
        var_ann=var_ann, cvar_95=cvar95, downside_dev=downside,
        semidev=semidev, max_dd=max_dd, avg_dd=avg_dd,
        skewness=sk, ex_kurtosis=ek, tail_ratio=tail_ratio,
        omega=omega, common_sense=common_sense, exp_return=exp_ret,
        alpha=alpha, sharpe=sharpe, treynor=treynor, sortino=sortino,
        info_ratio=info_r, idio_vol=idio, appraisal=appr,
        sterling=sterling, calmar=calmar, hurst=h,
    )


# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────

def fmt(v, pct: bool = False, dec: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    if pct:
        return f"{v * 100:.2f}%"
    return f"{v:.{dec}f}"

def sig(v, ga=None, bl=None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if ga is not None and v > ga:
        return "Good —"
    if bl is not None and v < bl:
        return "Weak —"
    return "Moderate —"

def card(title: str, value_str: str, interp: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value_str}</div>
        <div class="metric-interp">{interp}</div>
    </div>""", unsafe_allow_html=True)


# ── GLOSSARY ──────────────────────────────────────────────────────────────────
GLOSSARY = [
    ("Annualised Return", "Mean Daily Return × 252",
     "The compounded annual growth rate of the portfolio. Compare against Nifty 50 (12–14%). Higher is better."),
    ("Annualised Excess Return", "Portfolio Return − Risk-Free Rate",
     "Extra return earned above the G-Sec risk-free rate. This is the reward for taking equity risk over safe assets."),
    ("Annualised Outperformance", "Portfolio Return − Nifty 50 Return",
     "How much the portfolio beats or lags the Nifty 50 annually. Positive means you are beating the index."),
    ("Annualised Abnormal Return", "Jensen's Alpha (annualised)",
     "Return unexplained by market beta. Positive means the portfolio earns more than CAPM predicts."),
    ("Winning Days Ratio", "Days with Positive Return / Total Days",
     "Fraction of trading days the portfolio ended in the green. Above 50% is generally healthy."),
    ("Annual Risk-Free Rate", "India 10-Year G-Sec Yield (approx. 6.8%)",
     "The baseline return available with zero risk. Used as the hurdle rate in Sharpe, Sortino, Treynor, and Alpha."),
    ("Annualised Variance", "Daily Variance × 252",
     "Squared measure of return dispersion scaled to one year. Used to decompose systematic vs unsystematic risk."),
    ("Annualised Risk (Volatility)", "Daily Std Dev × sqrt(252)",
     "Standard deviation of returns annualised. The most common measure of total portfolio risk. Nifty 50 is typically 15–18% pa."),
    ("Beta", "Cov(Portfolio, Market) / Var(Market)",
     "Sensitivity to market moves. Beta of 1 moves with the market. Above 1 amplifies swings. Below 1 is defensive. Negative inverts."),
    ("Systematic Risk", "Beta squared × Market Variance × 252",
     "Portion of total portfolio risk driven by broad market movements. Cannot be eliminated through diversification."),
    ("Unsystematic Risk", "Total Variance − Systematic Risk",
     "Portfolio-specific risk that CAN be reduced by diversifying across uncorrelated assets. High value means underdiversified."),
    ("Tracking Error", "Std Dev(Portfolio Returns − Benchmark Returns) × sqrt(252)",
     "How closely the portfolio follows the benchmark. Low TE means index-like. High TE means significant active bets vs the index."),
    ("Value at Risk — Annualised (95%)", "5th Percentile of Daily Returns × sqrt(252)",
     "Worst expected annual loss on 95% of scenarios. For example VaR of -15% means you lose no more than 15% pa in 95% of cases."),
    ("Conditional VaR (Expected Shortfall)", "Average of returns below the VaR threshold",
     "Average loss on those extreme bad days beyond VaR. A tail-risk measure that shows how bad the worst case really is."),
    ("Downside Deviation", "Std Dev of Negative Returns × sqrt(252)",
     "Like volatility but only penalises losses, ignoring upside. Used in the Sortino Ratio. Lower means better downside protection."),
    ("Semideviation", "Std Dev of Returns Below Mean × sqrt(252)",
     "Deviation of returns falling below the average. Similar to downside deviation but uses the mean as the threshold."),
    ("Max Drawdown", "(Trough Value − Peak Value) / Peak Value",
     "Largest peak-to-trough percentage loss over the entire period. Closer to 0% means more resilient."),
    ("Average Drawdown", "Mean of all drawdown periods",
     "Typical severity of losing streaks over the full history. Gives context on how common bad periods are."),
    ("Skewness", "Third standardised moment of returns",
     "Positive skew means rare large gains with frequent small losses (good). Negative means rare large losses with frequent small gains (risky)."),
    ("Excess Kurtosis", "Fourth moment − 3 (Fisher definition)",
     "Fat-tails indicator. Positive kurtosis means extreme events happen more often than a normal distribution predicts."),
    ("Tail Ratio", "|95th percentile| / |5th percentile|",
     "Ratio of upside tail to downside tail. Greater than 1 means big gains in good times outsize big losses in bad times."),
    ("Omega Ratio", "Sum of Gains / Absolute Sum of Losses",
     "Considers the entire return distribution. Greater than 1 means portfolio gains more on up days than it loses on down days."),
    ("Common Sense Ratio", "Omega Ratio × Tail Ratio",
     "Composite of Omega and Tail ratios capturing both frequency and magnitude of wins versus losses. Higher is better."),
    ("Expected Return", "Arithmetic Mean of Daily Returns × 252",
     "Simple annualised average of daily returns. Your baseline expectation for a normal year."),
    ("Jensen's Alpha", "(Portfolio Return − Rf) − Beta × (Market Return − Rf)",
     "Excess return after adjusting for market risk. Positive means genuine skill or edge. Negative means underperformance vs CAPM."),
    ("Sharpe Ratio", "(Return − Rf) / Total Volatility",
     "Return per unit of total risk. The most widely used risk-adjusted measure. Above 1 is good, above 2 is excellent."),
    ("Treynor Ratio", "(Return − Rf) / Beta",
     "Return per unit of market risk only. Unlike Sharpe it ignores unsystematic risk. Best for comparing well-diversified portfolios."),
    ("Sortino Ratio", "(Return − Rf) / Downside Deviation",
     "Like Sharpe but only penalises downside volatility not upside gains. Fairer for asymmetric strategies. Above 1 is good."),
    ("Information Ratio", "(Portfolio Return − Benchmark Return) / Tracking Error",
     "Active return per unit of active risk. Measures consistency of outperformance. Above 0.5 is good, above 1.0 is excellent."),
    ("Idiosyncratic Volatility", "sqrt(Unsystematic Risk)",
     "Non-market component of total volatility. High value means concentrated single-stock bets. Reduces with better diversification."),
    ("Appraisal Ratio", "Alpha / Idiosyncratic Volatility",
     "Alpha earned per unit of diversifiable risk taken. Measures the quality of active stock-picking decisions. Higher is better."),
    ("Sterling Ratio", "Annual Return / Absolute Average Drawdown",
     "Like Calmar but uses average drawdown instead of maximum, making it less sensitive to a single extreme event."),
    ("Calmar Ratio", "Annual Return / Absolute Max Drawdown",
     "Return relative to worst-case loss. A Calmar of 1 means the portfolio earns back its max drawdown in exactly one year."),
    ("Hurst Exponent", "Estimated from rescaled range analysis",
     "H above 0.5 means trending or momentum-driven returns. H of 0.5 means random walk. H below 0.5 means mean-reverting."),
]


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="brand-title">Optimisation Solutions</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Portfolio Optimisation Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-divider"></div>', unsafe_allow_html=True)

# ── SEARCH BAR ────────────────────────────────────────────────────────────────
_, sc, _ = st.columns([1, 6, 1])
with sc:
    selected_names = st.multiselect(
        f"Search and select stocks  —  Nifty 50  ·  Bank Nifty  ·  Nifty Midcap 150   (max {MAX_STOCKS})",
        options=ALL_OPTIONS,
        default=["[Nifty 50] Reliance Industries", "[Nifty 50] TCS",
                 "[Nifty 50] Infosys", "[Nifty 50] HDFC Bank",
                 "[Nifty 50] ICICI Bank", "[Nifty 50] Axis Bank"],
        placeholder="Type a company name…",
    )
    if len(selected_names) > MAX_STOCKS:
        st.error(
            f"Maximum {MAX_STOCKS} stocks allowed for portfolio optimisation. "
            f"You have selected {len(selected_names)}. "
            f"Please remove {len(selected_names) - MAX_STOCKS} stock(s) to continue."
        )

# ── AMOUNT + HORIZON ──────────────────────────────────────────────────────────
_, ca, cb, _ = st.columns([1, 2, 2, 1])

with ca:
    st.markdown("**Investment Amount (₹)**")
    investment = st.number_input(
        "inv_amount", label_visibility="collapsed",
        min_value=1, max_value=100_000_000,
        value=1_000_000, step=1, format="%d",
        help="Enter any whole number between 1 and 10,00,00,000 (10 Crore).",
    )
    if investment > 0:
        st.caption(f"₹ {investment:,}  =  ₹ {investment/1e5:.2f} L  =  ₹ {investment/1e7:.4f} Cr")

with cb:
    st.markdown("**Investment Horizon**")
    pidx = st.selectbox(
        "horizon", label_visibility="collapsed",
        options=list(range(len(PERIOD_OPTIONS))),
        index=6,
        format_func=lambda i: PERIOD_LABELS[i],
        help="Minimum 3 months · Maximum 25 years",
    )
    period = PERIOD_OPTIONS[pidx]

# ── OPTIMISATION GOAL + RUN ───────────────────────────────────────────────────
_, cg, ch, _ = st.columns([1, 2, 2, 1])

with cg:
    opt_obj = st.radio(
        "Optimisation Goal",
        ["Max Sharpe Ratio", "Min Volatility"],
        horizontal=True
    )

with ch:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Run Optimisation", use_container_width=True, type="primary")

st.markdown("---")

# ── SESSION STATE CACHE KEY ───────────────────────────────────────────────────
# Build a key from the current inputs. If it matches what's cached, we skip
# re-fetching and re-optimising — even when the slider triggers a rerun.
_cache_key = f"{sorted(selected_names)}|{period}|{opt_obj}|{investment}"

# If Run was clicked, clear any stale cache for a different config
if run:
    st.session_state["_computed"]  = None
    st.session_state["_cache_key"] = None

# Check if we have fresh cached results for the current inputs
_have_results = (
    st.session_state.get("_cache_key") == _cache_key
    and st.session_state.get("_computed") is not None
)

# ── SHOW LANDING PAGE if nothing computed yet ─────────────────────────────────
if not _have_results and not run:
    st.info("Select your stocks, set your amount and horizon, then click Run Optimisation.")
    st.markdown("""
    **What this tool does**
    - Pulls live historical data from NSE via Yahoo Finance
    - Runs Markowitz Mean-Variance Optimisation to find the best weights
    - Plots the Efficient Frontier with 2000 simulated portfolios
    - Shows Rolling Volatility, Drawdown, and Return Distribution
    - Computes 34 risk and performance ratios with full interpretation
    - Learn tab explains every ratio with its formula and meaning

    **Limits**
    - Maximum 15 stocks per portfolio
    - Minimum 2 stocks required
    - Minimum 3-month data period
    """)
    st.stop()

# ── VALIDATE INPUTS ───────────────────────────────────────────────────────────
if not _have_results:
    if investment <= 0:
        st.error("Investment amount must be greater than zero.")
        st.stop()
    if len(selected_names) < 2:
        st.error("Please select at least 2 stocks.")
        st.stop()
    if len(selected_names) > MAX_STOCKS:
        st.error(f"Please reduce your selection to {MAX_STOCKS} stocks or fewer.")
        st.stop()

    tickers = [t for t in [get_ticker(n) for n in selected_names] if t]
    if len(tickers) < 2:
        st.error("Could not resolve stock tickers. Please re-select your stocks.")
        st.stop()

    # ── FETCH ─────────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data from NSE…"):
        prices = fetch_prices(tuple(tickers), period)

    if prices.empty or prices.shape[1] < 2:
        st.error("Not enough data returned. Try a longer horizon or different stocks.")
        st.stop()

    t2n    = {get_ticker(n): short_name(n) for n in selected_names}
    prices.columns = [t2n.get(c, c) for c in prices.columns]
    vn     = list(prices.columns)

    if len(vn) < 2:
        st.error("Too few stocks had data. Try different selections.")
        st.stop()
    if len(prices) < max(30, len(vn) + 5):
        st.error("Not enough trading days. Choose a longer horizon.")
        st.stop()

    rets = compute_returns(prices)
    if rets.empty or rets.shape[1] < 2:
        st.error("Could not compute returns. Try different stocks or a longer period.")
        st.stop()

    vn   = [c for c in vn if c in rets.columns]
    rets = rets[vn]
    mr   = rets.mean()
    cov  = rets.cov()

    # ── OPTIMISE ──────────────────────────────────────────────────────────────
    with st.spinner("Running Markowitz optimisation…"):
        obj_key = "sharpe" if "Sharpe" in opt_obj else "min_vol"
        w       = optimize_portfolio(mr, cov, objective=obj_key)

    if w is None or len(w) != len(vn):
        st.error("Optimisation failed. Try different stocks or a longer period.")
        st.stop()

    opt_r, opt_v, opt_s = port_perf(w, mr.values, cov.values)

    # ── BENCHMARK ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching Nifty 50 benchmark…"):
        try:
            braw = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
            bp   = braw["Close"].ffill().bfill() if "Close" in braw.columns else None
            br   = bp.pct_change().replace([np.inf, -np.inf], np.nan).dropna() \
                   if bp is not None and len(bp) > 5 else None
        except Exception:
            br = None

    pd_  = rets[vn].dot(w)
    rat  = compute_ratios(pd_, br)

    # ── CACHE EVERYTHING ──────────────────────────────────────────────────────
    st.session_state["_computed"] = dict(
        vn=vn, rets=rets, mr=mr, cov=cov, w=w,
        opt_r=opt_r, opt_v=opt_v, opt_s=opt_s,
        pd_=pd_, rat=rat, br=br, investment=investment,
    )
    st.session_state["_cache_key"] = _cache_key

# ── RESTORE FROM CACHE (covers slider reruns) ─────────────────────────────────
_c       = st.session_state["_computed"]
vn       = _c["vn"]
rets     = _c["rets"]
mr       = _c["mr"]
cov      = _c["cov"]
w        = _c["w"]
opt_r    = _c["opt_r"]
opt_v    = _c["opt_v"]
opt_s    = _c["opt_s"]
pd_      = _c["pd_"]
rat      = _c["rat"]
br       = _c["br"]
investment = _c["investment"]

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Portfolio",
    "Efficient Frontier",
    "Risk and Volatility",
    "Correlation",
    "All Ratios",
    "Learn",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — PORTFOLIO
# ──────────────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="section-hdr">Optimal Portfolio Allocation</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Annual Return", fmt(opt_r, pct=True))
    c2.metric("Annual Volatility",      fmt(opt_v, pct=True))
    c3.metric("Sharpe Ratio",           fmt(opt_s))
    c4.metric("Investment",             f"₹{investment:,}")

    st.markdown("---")

    wdf = pd.DataFrame({
        "Stock":      vn,
        "Weight (%)": (w * 100).round(2),
        "Amount (₹)": (w * investment).round(0).astype(int),
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)
    wdf = wdf[wdf["Weight (%)"] > 0.01]

    st.subheader("Recommended Allocation")
    st.dataframe(wdf.style.bar(subset=["Weight (%)"], color="#7c6cf8"),
                 use_container_width=True)

    # Pie chart
    pie_data = wdf[wdf["Weight (%)"] > 0.5]
    if not pie_data.empty:
        fig_pie = px.pie(
            pie_data, names="Stock", values="Weight (%)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title="Weight Distribution"
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False,
                              paper_bgcolor="rgba(2,11,24,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Cumulative return
    st.subheader("Cumulative Return vs Nifty 50")
    try:
        cp = (1 + pd_).cumprod() * 100 - 100
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cp.index, y=cp.values, name="Portfolio",
            line=dict(color="#7c6cf8", width=2.5)))
        if br is not None:
            cb2 = (1 + br).cumprod() * 100 - 100
            cb2 = cb2.reindex(cp.index, method="ffill")
            fig_cum.add_trace(go.Scatter(
                x=cb2.index, y=cb2.values, name="Nifty 50",
                line=dict(color="#f97316", width=1.8, dash="dot")))
        fig_cum.update_layout(
            title="Cumulative Return (%)",
            paper_bgcolor="rgba(2,11,24,0)",
            plot_bgcolor="rgba(5,24,48,0.6)",
            hovermode="x unified",
            font=dict(color="#f0deb4"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0deb4")),
        )
        fig_cum.update_xaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)")
        fig_cum.update_yaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)")
        st.plotly_chart(fig_cum, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render cumulative return chart: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — EFFICIENT FRONTIER
# ──────────────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="section-hdr">Efficient Frontier</div>',
                unsafe_allow_html=True)
    try:
        with st.spinner("Calculating efficient frontier…"):
            ef = efficient_frontier(mr, cov, n_points=120)

        # Simulated portfolios
        n_sim  = 2000
        rng    = np.random.default_rng(0)
        sim_w  = rng.dirichlet(np.ones(len(vn)), n_sim)
        sim_r, sim_v, sim_s = [], [], []
        for sw in sim_w:
            rr, vv, ss = port_perf(sw, mr.values, cov.values)
            sim_r.append(rr * 100)
            sim_v.append(vv * 100)
            sim_s.append(ss)

        fig_ef = go.Figure()
        if ef["volatility"]:
            fig_ef.add_trace(go.Scatter(
                x=ef["volatility"], y=ef["returns"],
                mode="lines", name="Efficient Frontier",
                line=dict(color="#06b6d4", width=3)))

        fig_ef.add_trace(go.Scatter(
            x=sim_v, y=sim_r, mode="markers", name="Simulated Portfolios",
            marker=dict(size=4, color=sim_s, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Sharpe"),
                        opacity=0.5)))

        stock_vols = [safe_std(rets[n], annualise=True) * 100 for n in vn]
        stock_rets = [safe_mean(rets[n], annualise=True) * 100 for n in vn]
        fig_ef.add_trace(go.Scatter(
            x=stock_vols, y=stock_rets,
            mode="markers+text",
            text=[n[:14] for n in vn],
            textposition="top center",
            name="Individual Stocks",
            marker=dict(size=10, color="#f97316", symbol="diamond")))

        fig_ef.add_trace(go.Scatter(
            x=[opt_v * 100], y=[opt_r * 100],
            mode="markers+text",
            text=["Optimal"],
            textposition="top right",
            name="Optimal Portfolio",
            marker=dict(size=18, color="#facc15", symbol="star")))

        fig_ef.update_layout(
            title="Efficient Frontier — Risk vs Return",
            xaxis_title="Annual Volatility (%)",
            yaxis_title="Expected Annual Return (%)",
            paper_bgcolor="rgba(2,11,24,0)",
            plot_bgcolor="rgba(5,24,48,0.6)",
            hovermode="closest",
            font=dict(color="#f0deb4"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0deb4")),
        )
        fig_ef.update_xaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)")
        fig_ef.update_yaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)")
        st.plotly_chart(fig_ef, use_container_width=True)

        st.markdown("""
**How to read this chart**

- Blue curve — the efficient frontier (best return for each level of risk)
- Coloured dots — 2000 randomly generated portfolios (colour = Sharpe ratio)
- Orange diamonds — individual stocks
- Gold star — your optimal portfolio
        """)
    except Exception as e:
        st.error(f"Efficient frontier calculation failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — RISK AND VOLATILITY
# ──────────────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="section-hdr">Risk and Volatility Analysis</div>',
                unsafe_allow_html=True)

    # ── Rolling volatility — precompute ALL windows, pick by slider ──────────
    # This avoids re-fetching data when the slider changes.
    n_obs       = len(rets)
    max_window  = max(5, min(60, n_obs // 4))
    default_win = min(21, max_window)

    # Precompute returns matrix as numpy for speed
    rets_np  = rets[vn].values.astype(float)          # shape (n_obs, n_stocks)
    port_np  = pd_.values.astype(float)               # shape (n_obs,)
    dates    = rets.index

    win = st.select_slider(
        "Rolling volatility window",
        options=list(range(5, max_window + 1)),
        value=default_win,
        help=f"Drag to change window size. Max {max_window} days based on your data.",
    )

    # Rolling volatility — fully guarded, computed purely with numpy
    try:
        colors    = px.colors.qualitative.Vivid
        fig_vol   = go.Figure()
        any_trace = False
        actual_win = int(win)

        for i, name in enumerate(vn):
            col = rets_np[:, i]
            vol_vals = []
            for j in range(len(col)):
                start = max(0, j - actual_win + 1)
                window_data = col[start:j+1]
                window_data = window_data[np.isfinite(window_data)]
                if len(window_data) >= max(2, actual_win // 3):
                    vol_vals.append(float(np.std(window_data, ddof=1)) * np.sqrt(TRADING_DAYS) * 100)
                else:
                    vol_vals.append(np.nan)
            vol_series = pd.Series(vol_vals, index=dates)
            clean = vol_series.dropna()
            if len(clean) >= 2:
                fig_vol.add_trace(go.Scatter(
                    x=clean.index, y=clean.values,
                    name=name[:16], mode="lines",
                    line=dict(width=1.5, color=colors[i % len(colors)])))
                any_trace = True

        # Portfolio line
        pvol_vals = []
        for j in range(len(port_np)):
            start = max(0, j - actual_win + 1)
            wd = port_np[start:j+1]
            wd = wd[np.isfinite(wd)]
            if len(wd) >= max(2, actual_win // 3):
                pvol_vals.append(float(np.std(wd, ddof=1)) * np.sqrt(TRADING_DAYS) * 100)
            else:
                pvol_vals.append(np.nan)
        pvol_series = pd.Series(pvol_vals, index=dates).dropna()
        if len(pvol_series) >= 2:
            fig_vol.add_trace(go.Scatter(
                x=pvol_series.index, y=pvol_series.values,
                name="PORTFOLIO", mode="lines",
                line=dict(width=3, color="#fdf6e3", dash="dash")))
            any_trace = True

        if any_trace:
            fig_vol.update_layout(
                title=f"{actual_win}-Day Rolling Annualised Volatility (%)",
                paper_bgcolor="rgba(2,11,24,0)",
                plot_bgcolor="rgba(5,24,48,0.6)",
                hovermode="x unified",
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f0deb4")),
                font=dict(color="#f0deb4"),
            )
            fig_vol.update_xaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)",
                                  color="#f5e6c8")
            fig_vol.update_yaxes(showgrid=True, gridcolor="rgba(56,189,248,0.12)",
                                  color="#f5e6c8")
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Increase the data horizon to see rolling volatility at this window size.")
    except Exception as e:
        st.warning(f"Rolling volatility chart: {e}")

    # Risk-return scatter
    try:
        sdf = pd.DataFrame({
            "Stock": vn,
            "Annual Return (%)":    [safe_mean(rets[n], annualise=True) * 100 for n in vn],
            "Annual Volatility (%)": [safe_std(rets[n], annualise=True) * 100 for n in vn],
            "Sharpe": [
                safe_float(
                    (safe_mean(rets[n], annualise=True) - RISK_FREE_RATE) /
                    max(safe_std(rets[n], annualise=True), 1e-10)
                )
                for n in vn
            ],
            "Weight (%)": (w * 100).round(2),
        }).dropna()

        if not sdf.empty:
            sdf["Weight (%)"] = sdf["Weight (%)"].clip(lower=0.1)
            fig_rr = px.scatter(
                sdf, x="Annual Volatility (%)", y="Annual Return (%)",
                text="Stock", size="Weight (%)", color="Sharpe",
                color_continuous_scale="RdYlGn", size_max=40,
                title="Risk-Return Map (bubble size = portfolio weight)"
            )
            fig_rr.update_traces(textposition="top center")
            fig_rr.update_layout(
                paper_bgcolor="rgba(2,11,24,0)",
                plot_bgcolor="rgba(5,24,48,0.6)",
                font=dict(color="#f0deb4"),
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_rr, use_container_width=True)
    except Exception as e:
        st.warning(f"Risk-return chart could not be rendered: {e}")

    # Drawdown
    try:
        cum2  = (1 + pd_).cumprod()
        rmx2  = cum2.cummax()
        dds   = ((cum2 - rmx2) / rmx2 * 100).dropna()
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dds.index, y=dds.values, fill="tozeroy",
            line=dict(color="#ef4444", width=1.5),
            fillcolor="rgba(239,68,68,0.2)", name="Drawdown"))
        fig_dd.update_layout(
            title="Portfolio Drawdown (%)",
            paper_bgcolor="rgba(2,11,24,0)",
            plot_bgcolor="rgba(5,24,48,0.6)",
            font=dict(color="#f0deb4"))
        st.plotly_chart(fig_dd, use_container_width=True)
    except Exception as e:
        st.warning(f"Drawdown chart could not be rendered: {e}")

    # Return distribution
    try:
        clean_ret = pd_.dropna() * 100
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=clean_ret, nbinsx=60,
            marker_color="#7c6cf8", opacity=0.8, name="Daily Returns"))
        fig_hist.update_layout(
            title="Daily Return Distribution",
            paper_bgcolor="rgba(2,11,24,0)",
            plot_bgcolor="rgba(5,24,48,0.6)",
            font=dict(color="#f0deb4"))
        st.plotly_chart(fig_hist, use_container_width=True)
    except Exception as e:
        st.warning(f"Distribution chart could not be rendered: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — CORRELATION
# ──────────────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="section-hdr">Correlation Heatmap</div>',
                unsafe_allow_html=True)
    try:
        corr = rets[vn].corr()
        sn   = [n[:16] for n in vn]
        corr.index   = sn
        corr.columns = sn

        fig_cr = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu_r", zmin=-1, zmax=1,
            text=corr.round(2).values, texttemplate="%{text}",
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        ))
        fig_cr.update_layout(
            title="Pairwise Correlation Matrix",
            paper_bgcolor="rgba(2,11,24,0)",
            plot_bgcolor="rgba(5,24,48,0.6)",
            height=max(400, len(vn) * 40),
            font=dict(color="#f0deb4"),
        )
        st.plotly_chart(fig_cr, use_container_width=True)
    except Exception as e:
        st.warning(f"Correlation heatmap could not be rendered: {e}")

    st.markdown("""
| Correlation Range | Meaning |
|---|---|
| +0.8 to +1.0 | Very high — stocks move almost identically |
| +0.4 to +0.8 | Moderate positive — partially correlated |
| −0.2 to +0.4 | Low — good diversification benefit |
| Below −0.2    | Negative — excellent natural hedge |
    """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ALL RATIOS
# ──────────────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="section-hdr">All Ratios and Interpretation</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Return Metrics**")
        card("Annualised Return",
             fmt(rat["ann_return"], pct=True),
             f"{sig(rat['ann_return'], ga=0.10, bl=0.05)} Annual portfolio growth rate. Nifty 50 average is 12–14%.")
        card("Annualised Excess Return",
             fmt(rat["ann_excess"], pct=True),
             f"{sig(rat['ann_excess'], ga=0.02, bl=0)} Return above the risk-free rate. Reward for taking equity risk.")
        card("Annualised Outperformance",
             fmt(rat["ann_outperf"], pct=True),
             f"{sig(rat['ann_outperf'], ga=0, bl=-0.02)} Beats or lags Nifty 50 by this much per year.")
        card("Annualised Abnormal Return",
             fmt(rat["ann_abnormal"], pct=True),
             f"{sig(rat['ann_abnormal'], ga=0, bl=-0.02)} Return unexplained by market beta.")
        card("Expected Return",
             fmt(rat["exp_return"], pct=True),
             "Arithmetic annual average of daily returns.")
        card("Winning Days Ratio",
             fmt(rat["win_days"], pct=True),
             f"{sig(rat['win_days'], ga=0.52, bl=0.45)} Fraction of green days. Above 50% is healthy.")
        card("Annual Risk-Free Rate",
             fmt(rat["rf"], pct=True),
             "India 10-Year G-Sec yield used as the hurdle rate in all ratio calculations.")

    with c2:
        st.markdown("**Risk Metrics**")
        card("Annualised Variance",
             fmt(rat["ann_var"], pct=True),
             "Squared return dispersion per year.")
        card("Annualised Risk (Volatility)",
             fmt(rat["ann_vol"], pct=True),
             "Total risk. Nifty 50 is typically 15–18% per year. Lower is safer.")
        card("Beta",
             fmt(rat["beta"]),
             ("Defensive — moves less than the market." if rat["beta"] and rat["beta"] < 0.9
              else "Aggressive — amplifies market swings." if rat["beta"] and rat["beta"] >= 1.2
              else "Market-like — similar swings to the benchmark."))
        card("Systematic Risk",
             fmt(rat["sys_risk"], pct=True),
             "Market-driven risk that cannot be diversified away.")
        card("Unsystematic Risk",
             fmt(rat["unsys_risk"], pct=True),
             "Stock-specific risk. Reduce by adding more uncorrelated stocks.")
        card("Tracking Error",
             fmt(rat["te"], pct=True),
             "Deviation from Nifty 50. Low means index-like, high means active bets.")
        card("Value at Risk — Annualised (95%)",
             fmt(rat["var_ann"], pct=True),
             "Worst expected annual loss in 95% of scenarios.")
        card("Conditional VaR",
             fmt(rat["cvar_95"], pct=True),
             "Average daily loss on extreme bad days beyond VaR.")
        card("Downside Deviation",
             fmt(rat["downside_dev"], pct=True),
             "Annualised std dev of negative-return days only.")
        card("Semideviation",
             fmt(rat["semidev"], pct=True),
             "Std dev of returns below the mean.")
        card("Max Drawdown",
             fmt(rat["max_dd"], pct=True),
             f"{sig(rat['max_dd'], ga=-0.10, bl=-0.30)} Largest peak-to-trough loss. Closer to 0 is better.")
        card("Average Drawdown",
             fmt(rat["avg_dd"], pct=True),
             "Typical severity of losing streaks across the full period.")

    with c3:
        st.markdown("**Performance Ratios**")
        card("Sharpe Ratio",
             fmt(rat["sharpe"]),
             f"{sig(rat['sharpe'], ga=1.0, bl=0.5)} Return per unit of total risk. Above 1 is good, above 2 is excellent.")
        card("Sortino Ratio",
             fmt(rat["sortino"]),
             f"{sig(rat['sortino'], ga=1.0, bl=0.5)} Return per unit of downside risk only.")
        card("Treynor Ratio",
             fmt(rat["treynor"]),
             f"{sig(rat['treynor'], ga=0.08, bl=0)} Return per unit of market risk (beta).")
        card("Calmar Ratio",
             fmt(rat["calmar"]),
             f"{sig(rat['calmar'], ga=1.0, bl=0.5)} Annual return divided by max drawdown.")
        card("Sterling Ratio",
             fmt(rat["sterling"]),
             f"{sig(rat['sterling'], ga=1.0, bl=0.5)} Annual return divided by average drawdown.")
        card("Jensen's Alpha",
             fmt(rat["alpha"], pct=True),
             f"{sig(rat['alpha'], ga=0.02, bl=-0.02)} CAPM-adjusted excess return. Positive means genuine edge.")
        card("Information Ratio",
             fmt(rat["info_ratio"]),
             f"{sig(rat['info_ratio'], ga=0.5, bl=0)} Active return divided by tracking error.")
        card("Omega Ratio",
             fmt(rat["omega"]),
             f"{sig(rat['omega'], ga=1.5, bl=1.0)} Gains over losses using full return distribution.")
        card("Tail Ratio",
             fmt(rat["tail_ratio"]),
             f"{sig(rat['tail_ratio'], ga=1.0, bl=0.8)} Upside tail versus downside tail magnitude.")
        card("Common Sense Ratio",
             fmt(rat["common_sense"]),
             f"{sig(rat['common_sense'], ga=1.5, bl=1.0)} Omega multiplied by Tail ratio.")
        card("Idiosyncratic Volatility",
             fmt(rat["idio_vol"], pct=True),
             "Non-market component of volatility. Reduces with better diversification.")
        card("Appraisal Ratio",
             fmt(rat["appraisal"]),
             f"{sig(rat['appraisal'], ga=0.5, bl=0)} Alpha per unit of unsystematic risk.")

        st.markdown("**Statistical Properties**")
        card("Skewness",
             fmt(rat["skewness"]),
             f"{sig(rat['skewness'], ga=0, bl=-0.5)} Positive means occasional large gains. Negative means tail risk on the downside.")
        card("Excess Kurtosis",
             fmt(rat["ex_kurtosis"]),
             "Fat-tails indicator. High positive means extreme events happen more than expected.")
        h_val = rat.get("hurst")
        h_label = (
            "Trending returns (H above 0.5)" if h_val and h_val > 0.55
            else "Mean-reverting (H below 0.5)" if h_val and h_val < 0.45
            else "Random walk (H near 0.5)"
        )
        card("Hurst Exponent", fmt(h_val), h_label)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — LEARN
# ──────────────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="section-hdr">Learn — Complete Ratio Reference</div>',
                unsafe_allow_html=True)
    st.caption(
        "All 34 ratios used in this tool — with formula, plain-English meaning, "
        "and guidance on how to interpret each value."
    )

    # Render all ratio cards directly — no search bar, no filtering
    cards_html = ""
    for title, formula, desc in GLOSSARY:
        cards_html += f"""
        <div class="learn-card">
            <div class="learn-title">{title}</div>
            <div class="learn-formula">Formula: {formula}</div>
            <div class="learn-desc">{desc}</div>
        </div>"""
    st.markdown(cards_html, unsafe_allow_html=True)
