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

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bhardwaj Solutions — Portfolio Optimisation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }
    .brand-title {
        text-align: center; font-size: 36px; font-weight: 800;
        background: linear-gradient(90deg, #7c6cf8, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 2px; letter-spacing: 1px;
    }
    .brand-sub {
        text-align: center; font-size: 13px; color: #888;
        margin-bottom: 18px; letter-spacing: 3px; text-transform: uppercase;
    }
    .metric-card {
        background: #1a1a2e; border-radius: 12px; padding: 14px 18px;
        margin-bottom: 10px; border-left: 4px solid #7c6cf8;
    }
    .metric-title { font-size: 11px; color: #aaa; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 22px; font-weight: 700; color: #fff; margin: 4px 0; }
    .metric-interp { font-size: 12px; color: #ccc; margin-top: 4px; line-height: 1.5; }
    .learn-card {
        background: #111827; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 8px; border-left: 3px solid #06b6d4;
    }
    .learn-title { font-size: 15px; font-weight: 700; color: #06b6d4; margin-bottom: 4px; }
    .learn-formula {
        font-size: 12px; color: #fbbf24; font-family: monospace;
        background: #0f0f1a; padding: 4px 10px; border-radius: 4px;
        margin: 6px 0; display: inline-block;
    }
    .learn-desc { font-size: 13px; color: #ccc; line-height: 1.6; }
    .section-hdr {
        font-size: 20px; font-weight: 700; margin: 18px 0 10px 0;
        background: linear-gradient(90deg, #7c6cf8, #06b6d4);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stButton > button {
        background: linear-gradient(90deg, #7c6cf8, #06b6d4);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; font-size: 15px;
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

RISK_FREE_RATE = 0.068

ALL_OPTIONS = (
    [f"[Nifty 50] {k}" for k in NIFTY50] +
    [f"[Bank Nifty] {k}" for k in BANK_NIFTY] +
    [f"[Midcap 150] {k}" for k in MIDCAP150]
)

def get_ticker(name):
    if name.startswith("[Nifty 50] "):    return NIFTY50.get(name[11:])
    if name.startswith("[Bank Nifty] "): return BANK_NIFTY.get(name[13:])
    if name.startswith("[Midcap 150] "): return MIDCAP150.get(name[13:])
    return None

def short(name):
    for prefix in ["[Nifty 50] ","[Bank Nifty] ","[Midcap 150] "]:
        if name.startswith(prefix): return name[len(prefix):]
    return name

# ── PERIOD OPTIONS ────────────────────────────────────────────────────────────
PERIOD_OPTIONS, PERIOD_LABELS = [], []
for m in [3, 6, 9]:
    PERIOD_OPTIONS.append(f"{m}mo"); PERIOD_LABELS.append(f"{m} Months")
for y in range(1, 26):
    PERIOD_OPTIONS.append(f"{y}y"); PERIOD_LABELS.append(f"{y} Year{'s' if y>1 else ''}")

# ── CORE FUNCTIONS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, period):
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(raw.columns, pd.MultiIndex) and len(tickers)==1:
        prices.columns = tickers
    return prices.dropna(axis=1, how="all").ffill()

def compute_returns(prices):   return prices.pct_change().dropna()

def port_perf(w, mr, cov, rf=RISK_FREE_RATE):
    r = np.dot(w, mr)*252
    v = np.sqrt(w @ cov @ w)*np.sqrt(252)
    return r, v, (r-rf)/v if v!=0 else 0

def _neg_sharpe(w, mr, cov):
    r, v, s = port_perf(w, mr, cov)
    return -s

def _min_vol(w, mr, cov):
    r, v, s = port_perf(w, mr, cov)
    return v

def _port_ret(w, mr, cov):
    return port_perf(w, mr, cov)[0]

def _sum_to_one(w):
    return np.sum(w) - 1.0

def optimize(mr, cov, obj="sharpe"):
    n   = len(mr)
    b   = tuple((0.0, 1.0) for _ in range(n))
    c   = {"type": "eq", "fun": _sum_to_one}
    f   = _neg_sharpe if obj == "sharpe" else _min_vol
    x0  = np.array([1.0 / n] * n)
    return minimize(f, x0, args=(mr, cov), method="SLSQP",
                    bounds=b, constraints=c, options={"maxiter": 1000})

def ef_points(mr, cov, n=150):
    res  = {"returns": [], "volatility": [], "sharpe": []}
    n_   = len(mr)
    b    = tuple((0.0, 1.0) for _ in range(n_))
    x0   = np.array([1.0 / n_] * n_)
    for target in np.linspace(float(mr.min()) * 252, float(mr.max()) * 252, n):
        cs = [
            {"type": "eq", "fun": _sum_to_one},
            {"type": "eq", "fun": _port_ret,
             "args": (mr, cov),
             "jac": None},
        ]
        # Override the return constraint with a closure-free version
        _t = float(target)
        def _ret_constraint(w, _mr=mr, _cov=cov, _t=_t):
            return port_perf(w, _mr, _cov)[0] - _t
        cs[1] = {"type": "eq", "fun": _ret_constraint}
        r_ = minimize(_min_vol, x0.copy(), args=(mr, cov),
                      method="SLSQP", bounds=b, constraints=cs,
                      options={"maxiter": 500})
        if r_.success:
            rv, vv, sv = port_perf(r_.x, mr, cov)
            res["returns"].append(rv * 100)
            res["volatility"].append(vv * 100)
            res["sharpe"].append(sv)
    return res

def hurst(ts, max_lag=20):
    try:
        ts=np.array(ts,dtype=float); lags=range(2,min(max_lag,len(ts)//2))
        tau=[np.std(ts[l:]-ts[:-l]) for l in lags]
        if len(tau)<2 or any(t==0 for t in tau): return np.nan
        return np.polyfit(np.log(list(lags)),np.log(tau),1)[0]
    except: return np.nan

def compute_ratios(pr, br=None):
    r=pr.dropna()
    ann_ret=r.mean()*252; ann_vol=r.std()*np.sqrt(252); ann_var=r.var()*252; rf=RISK_FREE_RATE
    neg_r=r[r<0]; pos_r=r[r>0]
    downside=neg_r.std()*np.sqrt(252) if len(neg_r)>1 else np.nan
    semidev=r[r<r.mean()].std()*np.sqrt(252) if len(r[r<r.mean()])>1 else np.nan
    cum=(1+r).cumprod(); rmx=cum.cummax(); dd=(cum-rmx)/rmx
    max_dd=dd.min(); avg_dd=dd[dd<0].mean() if (dd<0).any() else 0
    var95=np.percentile(r,5); cvar95=r[r<=var95].mean(); var_ann=var95*np.sqrt(252)
    sharpe=(ann_ret-rf)/ann_vol if ann_vol!=0 else np.nan
    sortino=(ann_ret-rf)/downside if downside and not np.isnan(downside) else np.nan
    calmar=ann_ret/abs(max_dd) if max_dd!=0 else np.nan
    sterling=ann_ret/abs(avg_dd) if avg_dd!=0 else np.nan
    omega=pos_r.sum()/abs(neg_r.sum()) if len(neg_r)>0 and neg_r.sum()!=0 else np.nan
    tail=abs(np.percentile(r,95))/abs(np.percentile(r,5)) if np.percentile(r,5)!=0 else np.nan
    common=omega*tail if not np.isnan(omega) and not np.isnan(tail) else np.nan
    sk=float(skew(r)); ek=float(kurtosis(r,fisher=True))
    win=float((r>0).mean()); h=hurst(r.values)

    beta=alpha=treynor=info_r=te=idio=appr=sys_r=unsys_r=ann_exc=ann_out=ann_abn=None
    if br is not None and len(br)>10:
        try:
            ps=pd.Series(r.values.flatten(),index=pd.to_datetime(r.index))
            bs=pd.Series(br.values.flatten(),index=pd.to_datetime(br.index))
            pa,ba=ps.align(bs,join="inner"); pa=pa.dropna(); ba=ba.dropna()
            pa,ba=pa.align(ba,join="inner")
            if len(pa)>10:
                pv=np.array(pa,dtype=float); bv=np.array(ba,dtype=float)
                cm=np.cov(pv,bv); vb=np.var(bv,ddof=1); br_ann=bv.mean()*252
                if vb!=0:
                    beta=cm[0,1]/vb; alpha=(ann_ret-rf)-beta*(br_ann-rf)
                    treynor=(ann_ret-rf)/beta if beta!=0 else None
                    sys_r=beta**2*vb*252; unsys_r=ann_var-sys_r
                    d=pv-bv; te=d.std()*np.sqrt(252)
                    info_r=(ann_ret-br_ann)/te if te!=0 else None
                    idio=np.sqrt(max(unsys_r,0)) if unsys_r else None
                    appr=alpha/idio if alpha and idio and idio!=0 else None
                    ann_exc=ann_ret-rf; ann_out=ann_ret-br_ann; ann_abn=alpha
        except: pass

    return dict(ann_return=ann_ret,ann_excess=ann_exc,ann_outperf=ann_out,ann_abnormal=ann_abn,
                win_days=win,rf=rf,ann_var=ann_var,ann_vol=ann_vol,beta=beta,
                sys_risk=sys_r,unsys_risk=unsys_r,te=te,var_ann=var_ann,cvar_95=cvar95,
                downside_dev=downside,semidev=semidev,max_dd=max_dd,avg_dd=avg_dd,
                skewness=sk,ex_kurtosis=ek,tail_ratio=tail,omega=omega,common_sense=common,
                exp_return=float(r.mean()*252),alpha=alpha,sharpe=sharpe,treynor=treynor,
                sortino=sortino,info_ratio=info_r,idio_vol=idio,appraisal=appr,
                sterling=sterling,calmar=calmar,hurst=h)

def fmt(v, pct=False, dec=3):
    if v is None or (isinstance(v,float) and np.isnan(v)): return "N/A"
    return f"{v*100:.2f}%" if pct else f"{v:.{dec}f}"

def sig(v, ga=None, bl=None):
    if v is None or (isinstance(v,float) and np.isnan(v)): return ""
    if ga is not None and v>ga: return "🟢"
    if bl is not None and v<bl: return "🔴"
    return "🟡"

def card(title, val, interp):
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-interp">{interp}</div>
    </div>""", unsafe_allow_html=True)

# ── GLOSSARY ──────────────────────────────────────────────────────────────────
GLOSSARY = [
    ("Annualised Return","Mean Daily Return × 252",
     "The compounded annual growth rate of the portfolio. Compare against Nifty 50 (≈12–14%). Higher is better."),
    ("Annualised Excess Return","Port. Return − Risk-Free Rate",
     "Extra return earned above the risk-free rate (G-Sec yield). This is the reward for taking equity risk over safe assets."),
    ("Annualised Outperformance","Port. Return − Benchmark Return",
     "How much the portfolio beats (or lags) the Nifty 50 annually. Positive = you are beating the index."),
    ("Annualised Abnormal Return","Jensen's Alpha (annualised)",
     "Return unexplained by market movements (beta). Positive means the portfolio earns more than CAPM predicts."),
    ("Winning Days Ratio","Days with Positive Return / Total Days",
     "Fraction of trading days the portfolio ended in the green. Above 50% is generally healthy for a diversified portfolio."),
    ("Annual Risk-Free Rate","India 10-Year G-Sec Yield ≈ 6.8%",
     "The baseline return available with zero risk. Used as the hurdle rate in Sharpe, Sortino, Treynor, and Alpha calculations."),
    ("Annualised Variance","Daily Variance × 252",
     "The squared measure of return dispersion, scaled to one year. Used to decompose systematic vs unsystematic risk components."),
    ("Annualised Risk (Volatility)","Daily Std Dev × √252",
     "Standard deviation of returns annualised. The most common measure of total portfolio risk. Nifty 50 ≈ 15–18% pa."),
    ("Beta","Cov(Portfolio, Market) / Var(Market)",
     "Measures sensitivity to market moves. β=1 → moves with market exactly. β>1 → amplifies swings (aggressive). β<1 → defensive. β<0 → inverse."),
    ("Systematic Risk","β² × Market Variance × 252",
     "The portion of total portfolio risk driven by broad market movements. Cannot be eliminated through diversification."),
    ("Unsystematic Risk","Total Variance − Systematic Risk",
     "Portfolio-specific risk that CAN be reduced by diversifying across uncorrelated assets. High value = underdiversified."),
    ("Tracking Error","Std Dev(Port Returns − Bench Returns) × √252",
     "How closely the portfolio mirrors the benchmark. Low TE = index-like behaviour. High TE = significant active bets vs index."),
    ("Value at Risk — Annualised (95%)","5th Percentile of Daily Returns × √252",
     "The worst expected annual loss on 95% of scenarios. E.g. VaR=−15% means 95% of the time you lose no more than 15% pa."),
    ("Conditional VaR (CVaR / Expected Shortfall)","Average of returns below the VaR threshold",
     "The average loss on those extreme bad days beyond VaR. A tail-risk measure — tells you how bad the worst-case really is."),
    ("Downside Deviation","Std Dev of Negative Returns × √252",
     "Like volatility but only penalises losses, ignoring upside moves. Used in Sortino ratio. Lower = better downside protection."),
    ("Semideviation","Std Dev of Returns Below Mean × √252",
     "Deviation of returns that fall below the average return. Similar to downside deviation but uses the mean as the threshold."),
    ("Max Drawdown","(Trough Value − Peak Value) / Peak Value",
     "The largest peak-to-trough percentage loss over the entire period. Closer to 0% = more resilient. E.g. −30% means ₹1L fell to ₹70K."),
    ("Average Drawdown","Mean of all drawdown periods",
     "Typical severity of losing streaks over the full history. Gives context on how common bad periods are vs the Max Drawdown."),
    ("Skewness","Third standardised moment of returns",
     "Positive skew = rare large gains with frequent small losses (good). Negative skew = rare large losses with frequent small gains (risky)."),
    ("Excess Kurtosis","Fourth moment − 3 (Fisher definition)",
     "Fat-tails indicator. Positive kurtosis means extreme events happen more often than a normal distribution would predict. Higher = more crash risk."),
    ("Tail Ratio","|95th percentile| / |5th percentile|",
     "Ratio of the upside tail to the downside tail. >1 means big gains in good times outsize big losses in bad times. Higher is better."),
    ("Omega Ratio","Sum of Gains / |Sum of Losses|",
     "Considers the entire return distribution. >1 = portfolio gains more on up days than it loses on down days. >1.5 is generally considered good."),
    ("Common Sense Ratio","Omega Ratio × Tail Ratio",
     "A composite of the Omega and Tail ratios capturing both frequency and magnitude of wins vs losses. Higher is better."),
    ("Expected Return","Arithmetic Mean of Daily Returns × 252",
     "Simple annualised average of daily returns — your baseline expectation for a normal year."),
    ("Jensen's Alpha","(Port. Return − Rf) − β × (Market Return − Rf)",
     "Excess return after adjusting for market risk (beta). Positive alpha = genuine skill or edge. Negative = destroys value vs CAPM expectations."),
    ("Sharpe Ratio","(Return − Rf) / Total Volatility",
     "Return per unit of total risk. The most widely used risk-adjusted measure. >1 = good, >2 = excellent, <0 = losing to risk-free assets."),
    ("Treynor Ratio","(Return − Rf) / Beta",
     "Return per unit of market risk only. Unlike Sharpe it ignores unsystematic risk. Best used to compare well-diversified portfolios."),
    ("Sortino Ratio","(Return − Rf) / Downside Deviation",
     "Like Sharpe but only penalises downside volatility, not upside gains. Fairer for asymmetric strategies. >1 = good, >2 = excellent."),
    ("Information Ratio","(Port. Return − Bench Return) / Tracking Error",
     "Active return per unit of active risk. Measures consistency of outperformance over the benchmark. >0.5 = good, >1.0 = excellent."),
    ("Idiosyncratic Volatility","√(Unsystematic Risk)",
     "The stock-specific, non-market component of total volatility. High value = concentrated single-stock bets. Reduces with diversification."),
    ("Appraisal Ratio","Alpha / Idiosyncratic Volatility",
     "Alpha earned per unit of diversifiable risk taken. Measures the quality of active stock-picking decisions. Higher = better use of the risk budget."),
    ("Sterling Ratio","Annual Return / |Average Drawdown|",
     "Like Calmar but uses average drawdown instead of maximum, making it less sensitive to a single extreme event. Higher is better."),
    ("Calmar Ratio","Annual Return / |Max Drawdown|",
     "Return relative to worst-case loss. A Calmar of 1 means the portfolio earns back its max drawdown in exactly one year. >1 is healthy."),
    ("Hurst Exponent","Estimated from rescaled range (R/S) analysis",
     "H > 0.5 = trending / momentum-driven returns. H = 0.5 = random walk (no memory). H < 0.5 = mean-reverting. Useful for strategy design."),
]

# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="brand-title">Bhardwaj Solutions</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">Portfolio Optimisation Tool</div>', unsafe_allow_html=True)

# ── SEARCH BAR ────────────────────────────────────────────────────────────────
_, sc, _ = st.columns([1, 6, 1])
with sc:
    selected_names = st.multiselect(
        "🔍  Search & select stocks  —  Nifty 50  ·  Bank Nifty  ·  Nifty Midcap 150",
        options=ALL_OPTIONS,
        default=["[Nifty 50] Reliance Industries","[Nifty 50] TCS",
                 "[Nifty 50] Infosys","[Nifty 50] HDFC Bank",
                 "[Nifty 50] ICICI Bank","[Nifty 50] Axis Bank"],
        placeholder="Type a company name…",
    )

# ── AMOUNT + HORIZON ──────────────────────────────────────────────────────────
_, ca, cb, _ = st.columns([1, 2, 2, 1])
with ca:
    st.markdown("**💰 Investment Amount (₹)**")
    investment = st.number_input(
        "inv", label_visibility="collapsed",
        min_value=1, max_value=100_000_000,
        value=1_000_000, step=1, format="%d",
        help="Range: ₹1 to ₹10 Crore. Enter any whole number — no zeros only."
    )
    if investment > 0:
        st.caption(f"₹ {investment:,}  =  ₹ {investment/100_000:.2f} L  =  ₹ {investment/10_000_000:.4f} Cr")
    else:
        st.error("Amount must be greater than ₹0")

with cb:
    st.markdown("**📅 Investment Horizon**")
    pidx = st.selectbox(
        "horizon", label_visibility="collapsed",
        options=list(range(len(PERIOD_OPTIONS))),
        index=6,
        format_func=lambda i: PERIOD_LABELS[i],
        help="3 months minimum · 25 years maximum"
    )
    period = PERIOD_OPTIONS[pidx]

# ── OPT GOAL + RUN ────────────────────────────────────────────────────────────
_, cg, ch, _ = st.columns([1, 2, 2, 1])
with cg:
    opt_obj = st.radio("🎯 **Optimisation Goal**",
                       ["Max Sharpe Ratio","Min Volatility"], horizontal=True)
with ch:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀  Run Optimisation", use_container_width=True, type="primary")

st.markdown("---")

# ── GUARD ─────────────────────────────────────────────────────────────────────
if not run:
    st.info("👆 Select stocks, set your amount & horizon, then click **Run Optimisation**.")
    st.stop()
if investment <= 0:
    st.error("Investment must be > ₹0"); st.stop()
if len(selected_names) < 2:
    st.error("Select at least 2 stocks."); st.stop()

tickers = [t for t in [get_ticker(n) for n in selected_names] if t]
if len(tickers) < 2:
    st.error("Could not resolve tickers."); st.stop()

# ── DATA ──────────────────────────────────────────────────────────────────────
with st.spinner("📡 Fetching data…"):
    prices = fetch_prices(tickers, period)
if prices.empty or prices.shape[1] < 2:
    st.error("Not enough data. Try a longer horizon."); st.stop()

t2n = {get_ticker(n): short(n) for n in selected_names}
prices.columns = [t2n.get(c, c) for c in prices.columns]
vn = list(prices.columns)
rets = compute_returns(prices)
mr = rets.mean(); cov = rets.cov()

with st.spinner("⚙️ Optimising…"):
    obj = "sharpe" if "Sharpe" in opt_obj else "min_vol"
    res = optimize(mr, cov, obj)
    w   = res.x
    or_, ov, os_ = port_perf(w, mr, cov)

with st.spinner("📡 Benchmark…"):
    braw = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
    bp   = braw["Close"].ffill() if "Close" in braw.columns else None
    br   = bp.pct_change().dropna() if bp is not None else None

pd_ = rets[vn].dot(w)
rat = compute_ratios(pd_, br)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "🏆 Portfolio","📉 Efficient Frontier",
    "📊 Risk & Volatility","🔗 Correlation",
    "📐 All Ratios","📚 Learn"
])

# ─ TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    st.markdown('<div class="section-hdr">Optimal Portfolio Allocation</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Expected Annual Return", f"{or_*100:.2f}%")
    c2.metric("Annual Volatility",      f"{ov*100:.2f}%")
    c3.metric("Sharpe Ratio",           f"{os_:.3f}")
    c4.metric("Investment",             f"₹{investment:,}")
    st.markdown("---")
    wdf = pd.DataFrame({"Stock":vn,"Weight (%)":(w*100).round(2),
                         "Amount (₹)":(w*investment).round(0).astype(int)}
                       ).sort_values("Weight (%)",ascending=False).reset_index(drop=True)
    wdf = wdf[wdf["Weight (%)"]>0.01]
    st.dataframe(wdf.style.bar(subset=["Weight (%)"],color="#7c6cf8"), use_container_width=True)
    fig_p = px.pie(wdf[wdf["Weight (%)"]>0.5], names="Stock", values="Weight (%)",
                   color_discrete_sequence=px.colors.qualitative.Vivid, title="Weight Distribution")
    fig_p.update_traces(textposition="inside", textinfo="percent+label")
    fig_p.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_p, use_container_width=True)
    st.subheader("Cumulative Return vs Nifty 50")
    cp = (1+pd_).cumprod()*100-100
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=cp.index,y=cp.values,name="Portfolio",
                               line=dict(color="#7c6cf8",width=2.5)))
    if br is not None:
        cb2=(1+br).cumprod()*100-100; cb2=cb2.reindex(cp.index,method="ffill")
        fig_c.add_trace(go.Scatter(x=cb2.index,y=cb2.values,name="Nifty 50",
                                   line=dict(color="#f97316",width=1.8,dash="dot")))
    fig_c.update_layout(title="Cumulative Return (%)", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    fig_c.update_xaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
    fig_c.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig_c, use_container_width=True)

# ─ TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    st.markdown('<div class="section-hdr">Efficient Frontier</div>', unsafe_allow_html=True)
    with st.spinner("Calculating…"): ef = ef_points(mr, cov)
    nsim=3000; sw=np.random.dirichlet(np.ones(len(vn)),nsim)
    sr,sv,ss=[],[],[]
    for ww in sw:
        rr,vv,sss=port_perf(ww,mr,cov); sr.append(rr*100); sv.append(vv*100); ss.append(sss)
    fig_e = go.Figure()
    if ef["volatility"]:
        fig_e.add_trace(go.Scatter(x=ef["volatility"],y=ef["returns"],mode="lines",
            name="Frontier",line=dict(color="#06b6d4",width=3)))
    fig_e.add_trace(go.Scatter(x=sv,y=sr,mode="markers",name="Simulated",
        marker=dict(size=4,color=ss,colorscale="Viridis",showscale=True,
                    colorbar=dict(title="Sharpe"),opacity=0.5)))
    fig_e.add_trace(go.Scatter(
        x=[rets[n].std()*np.sqrt(252)*100 for n in vn],
        y=[mr[n]*252*100 for n in vn],
        mode="markers+text",text=[n[:12] for n in vn],textposition="top center",
        name="Stocks",marker=dict(size=10,color="#f97316",symbol="diamond")))
    fig_e.add_trace(go.Scatter(x=[ov*100],y=[or_*100],mode="markers+text",
        text=["⭐ Optimal"],textposition="top right",name="Optimal",
        marker=dict(size=18,color="#facc15",symbol="star")))
    fig_e.update_layout(title="Efficient Frontier — Risk vs Return",
        xaxis_title="Annual Volatility (%)",yaxis_title="Annual Return (%)",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",hovermode="closest")
    fig_e.update_xaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
    fig_e.update_yaxes(showgrid=True,gridcolor="rgba(255,255,255,0.07)")
    st.plotly_chart(fig_e, use_container_width=True)

# ─ TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    st.markdown('<div class="section-hdr">Risk & Volatility Analysis</div>', unsafe_allow_html=True)
    win = st.slider("Rolling window (days)",10,60,21)
    rv  = rets[vn].rolling(win).std()*np.sqrt(252)*100
    clrs= px.colors.qualitative.Vivid
    fig_v = go.Figure()
    for i,n in enumerate(vn):
        fig_v.add_trace(go.Scatter(x=rv.index,y=rv[n],name=n[:14],
                                   mode="lines",line=dict(width=1.5,color=clrs[i%len(clrs)])))
    prv=pd_.rolling(win).std()*np.sqrt(252)*100
    fig_v.add_trace(go.Scatter(x=prv.index,y=prv.values,name="PORTFOLIO",
                               mode="lines",line=dict(width=3,color="white",dash="dash")))
    fig_v.update_layout(title=f"{win}-Day Rolling Volatility (%)",
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",hovermode="x unified")
    st.plotly_chart(fig_v, use_container_width=True)

    sdf=pd.DataFrame({"Stock":vn,
        "Annual Return (%)":[mr[n]*252*100 for n in vn],
        "Annual Volatility (%)":[rets[n].std()*np.sqrt(252)*100 for n in vn],
        "Sharpe":[(mr[n]*252-RISK_FREE_RATE)/(rets[n].std()*np.sqrt(252)) for n in vn],
        "Weight (%)":(w*100).round(2)})
    fig_rr=px.scatter(sdf,x="Annual Volatility (%)",y="Annual Return (%)",
        text="Stock",size="Weight (%)",color="Sharpe",color_continuous_scale="RdYlGn",
        size_max=40,title="Risk-Return Map (bubble = weight)")
    fig_rr.update_traces(textposition="top center")
    fig_rr.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rr, use_container_width=True)

    cum2=(1+pd_).cumprod(); rmx2=cum2.cummax(); dds=(cum2-rmx2)/rmx2*100
    fig_dd=go.Figure()
    fig_dd.add_trace(go.Scatter(x=dds.index,y=dds.values,fill="tozeroy",
        line=dict(color="#ef4444",width=1.5),fillcolor="rgba(239,68,68,0.2)",name="DD"))
    fig_dd.update_layout(title="Portfolio Drawdown (%)",paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_dd, use_container_width=True)

    fig_h=go.Figure()
    fig_h.add_trace(go.Histogram(x=pd_*100,nbinsx=60,
        marker_color="#7c6cf8",opacity=0.8,name="Daily Returns"))
    fig_h.update_layout(title="Daily Return Distribution",paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, use_container_width=True)

# ─ TAB 4 ─────────────────────────────────────────────────────────────────────
with t4:
    st.markdown('<div class="section-hdr">Correlation Heatmap</div>', unsafe_allow_html=True)
    corr=rets[vn].corr(); sn=[n[:14] for n in vn]
    corr.index=sn; corr.columns=sn
    fig_cr=go.Figure(data=go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
        colorscale="RdBu_r",zmin=-1,zmax=1,text=corr.round(2).values,texttemplate="%{text}"))
    fig_cr.update_layout(title="Pairwise Correlation Matrix",paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)",height=max(400,len(vn)*38))
    st.plotly_chart(fig_cr, use_container_width=True)
    st.markdown("""| Range | Meaning |\n|---|---|\n| +0.8 to +1.0 | Near identical movement |\n| +0.4 to +0.8 | Moderate positive |\n| −0.2 to +0.4 | Low — good diversification |\n| Below −0.2 | Negative — excellent hedge |""")

# ─ TAB 5 ─────────────────────────────────────────────────────────────────────
with t5:
    st.markdown('<div class="section-hdr">All Ratios & Interpretation</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**Return Metrics**")
        card("Annualised Return",fmt(rat["ann_return"],pct=True),
             f"{sig(rat['ann_return'],ga=0.10,bl=0.05)} Annual portfolio growth rate.")
        card("Annualised Excess Return",fmt(rat["ann_excess"],pct=True),
             f"{sig(rat['ann_excess'],ga=0.02,bl=0)} Return above risk-free rate — reward for investing.")
        card("Annualised Outperformance",fmt(rat["ann_outperf"],pct=True),
             f"{sig(rat['ann_outperf'],ga=0,bl=-0.02)} Beats (+) or lags (−) Nifty 50.")
        card("Annualised Abnormal Return",fmt(rat["ann_abnormal"],pct=True),
             f"{sig(rat['ann_abnormal'],ga=0,bl=-0.02)} Return unexplained by market beta (Jensen's Alpha).")
        card("Expected Return",fmt(rat["exp_return"],pct=True),
             "Arithmetic annual average of daily returns.")
        card("Winning Days Ratio",fmt(rat["win_days"],pct=True),
             f"{sig(rat['win_days'],ga=0.52,bl=0.45)} Fraction of green days. >50% is healthy.")
        card("Annual Risk-Free Rate",fmt(rat["rf"],pct=True),
             "India 10-Year G-Sec yield — the hurdle rate used in all ratio calculations.")

    with c2:
        st.markdown("**Risk Metrics**")
        card("Annualised Variance",fmt(rat["ann_var"],pct=True),
             "Squared return dispersion per year.")
        card("Annualised Risk (Volatility)",fmt(rat["ann_vol"],pct=True),
             f"{sig(rat['ann_vol'],ga=None,bl=None)} Total risk. Nifty 50 ≈ 15–18%.")
        card("Beta",fmt(rat["beta"]),
             f"{'🟢 Defensive (<1)' if rat['beta'] and rat['beta']<0.9 else '🟡 Market-like' if rat['beta'] and rat['beta']<1.2 else '🔴 Aggressive (>1.2)'} Market sensitivity.")
        card("Systematic Risk",fmt(rat["sys_risk"],pct=True),
             "Market-driven risk — cannot be diversified away.")
        card("Unsystematic Risk",fmt(rat["unsys_risk"],pct=True),
             "Stock-specific risk — reduce by diversifying more.")
        card("Tracking Error",fmt(rat["te"],pct=True),
             f"{sig(rat['te'],ga=None,bl=None)} Deviation from Nifty 50 benchmark.")
        card("VaR — Annualised (95%)",fmt(rat["var_ann"],pct=True),
             "Worst expected annual loss in 95% of scenarios.")
        card("Conditional VaR (CVaR)",fmt(rat["cvar_95"],pct=True),
             "Average daily loss on extreme bad days (tail risk).")
        card("Downside Deviation",fmt(rat["downside_dev"],pct=True),
             "Annualised std dev of negative-return days only.")
        card("Semideviation",fmt(rat["semidev"],pct=True),
             "Std dev of returns below the mean.")
        card("Max Drawdown",fmt(rat["max_dd"],pct=True),
             f"{sig(rat['max_dd'],ga=-0.10,bl=-0.30)} Largest peak-to-trough loss.")
        card("Average Drawdown",fmt(rat["avg_dd"],pct=True),
             "Typical severity of losing streaks.")

    with c3:
        st.markdown("**Performance Ratios**")
        card("Sharpe Ratio",fmt(rat["sharpe"]),
             f"{sig(rat['sharpe'],ga=1.0,bl=0.5)} Return / total risk. >1=good, >2=excellent.")
        card("Sortino Ratio",fmt(rat["sortino"]),
             f"{sig(rat['sortino'],ga=1.0,bl=0.5)} Return / downside risk only.")
        card("Treynor Ratio",fmt(rat["treynor"]),
             f"{sig(rat['treynor'],ga=0.08,bl=0)} Return / market risk (beta).")
        card("Calmar Ratio",fmt(rat["calmar"]),
             f"{sig(rat['calmar'],ga=1.0,bl=0.5)} Return / max drawdown.")
        card("Sterling Ratio",fmt(rat["sterling"]),
             f"{sig(rat['sterling'],ga=1.0,bl=0.5)} Return / average drawdown.")
        card("Jensen's Alpha",fmt(rat["alpha"],pct=True),
             f"{sig(rat['alpha'],ga=0.02,bl=-0.02)} CAPM-adjusted excess return.")
        card("Information Ratio",fmt(rat["info_ratio"]),
             f"{sig(rat['info_ratio'],ga=0.5,bl=0)} Active return / tracking error.")
        card("Omega Ratio",fmt(rat["omega"]),
             f"{sig(rat['omega'],ga=1.5,bl=1.0)} Gains / losses (full distribution). >1.5=good.")
        card("Tail Ratio",fmt(rat["tail_ratio"]),
             f"{sig(rat['tail_ratio'],ga=1.0,bl=0.8)} Upside tail vs downside tail.")
        card("Common Sense Ratio",fmt(rat["common_sense"]),
             f"{sig(rat['common_sense'],ga=1.5,bl=1.0)} Omega × Tail ratio composite.")
        card("Idiosyncratic Volatility",fmt(rat["idio_vol"],pct=True),
             "Non-market volatility. Reduce via diversification.")
        card("Appraisal Ratio",fmt(rat["appraisal"]),
             f"{sig(rat['appraisal'],ga=0.5,bl=0)} Alpha per unit of unsystematic risk.")
        st.markdown("**Statistical Properties**")
        card("Skewness",fmt(rat["skewness"]),
             f"{sig(rat['skewness'],ga=0,bl=-0.5)} Positive = occasional large gains (good).")
        card("Excess Kurtosis",fmt(rat["ex_kurtosis"]),
             "Fat-tails indicator. High positive = more crash risk.")
        card("Hurst Exponent",fmt(rat["hurst"]),
             f"{'🟢 Trending (H>0.5)' if rat['hurst'] and rat['hurst']>0.55 else '🔵 Mean-reverting (H<0.5)' if rat['hurst'] and rat['hurst']<0.45 else '🟡 Random Walk (H≈0.5)'}")

# ─ TAB 6: LEARN ──────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="section-hdr">📚 Learn — Complete Ratio Reference</div>', unsafe_allow_html=True)
    st.caption("Every ratio used in this tool — formula, meaning, and how to interpret values.")
    sl = st.text_input("🔍 Search…", placeholder="e.g. Sharpe, Beta, Drawdown, Alpha…")
    shown = [(t,f,d) for t,f,d in GLOSSARY
             if not sl or sl.lower() in t.lower() or sl.lower() in d.lower()]
    if not shown:
        st.warning("No match found.")
    for (title, formula, desc) in shown:
        st.markdown(f"""<div class="learn-card">
            <div class="learn-title">{title}</div>
            <div class="learn-formula">Formula: {formula}</div>
            <div class="learn-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)
