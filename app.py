import matplotlib
matplotlib.use('Agg')

import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import re
import numpy as np

# ─────────────────────────────────────────────
# 1. PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kerala Election Atlas",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# 1b. PASSWORD PROTECTION
# ─────────────────────────────────────────────
# Set APP_PASSWORD in st.secrets to enable.
# Auth token stored in URL query params (persists in browser tab).
# Clear by removing ?auth=... from URL or clicking Sign Out.

def _check_password():
    secret_pw = st.secrets.get("APP_PASSWORD", None)
    if not secret_pw:
        return True  # no password configured — open access

    import hashlib
    token = hashlib.sha256(secret_pw.encode()).hexdigest()[:16]

    # Check if already authenticated via query param
    params = st.query_params
    if params.get("auth") == token:
        return True

    # Show login screen
    st.markdown("""
    <div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0b1120;">
      <div style="background:linear-gradient(135deg,#0d1b2a,#1a2c45);border:1px solid #2a4060;
                  border-radius:16px;padding:2.5rem 3rem;max-width:400px;width:100%;text-align:center;
                  box-shadow:0 20px 60px rgba(0,0,0,0.5);">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">🗳️</div>
        <div style="font-family:'Playfair Display',serif;font-size:1.6rem;font-weight:900;
                    color:#c9a84c;margin-bottom:0.3rem;">Kerala Election Atlas</div>
        <div style="font-size:0.75rem;color:#8fa3c0;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:1.8rem;">Private Access</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Overlay the input on top
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        entered = st.text_input("Password", type="password", key="_pw_input",
                                placeholder="Enter access password",
                                label_visibility="collapsed")
        if st.button("Unlock →", use_container_width=True, key="_pw_btn", type="primary"):
            if entered == secret_pw:
                st.query_params["auth"] = token
                st.rerun()
            else:
                st.error("Incorrect password.")
    st.stop()

_check_password()


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b1120;
    color: #e8e4da;
}
.atlas-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2c45 60%, #0d1b2a 100%);
    border-bottom: 2px solid #c9a84c;
    padding: 1.6rem 2rem 1.2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    position: relative; overflow: hidden;
}
.atlas-header::before {
    content:''; position:absolute; top:-40px; right:-40px;
    width:200px; height:200px; border-radius:50%;
    background:radial-gradient(circle,rgba(201,168,76,0.12) 0%,transparent 70%);
}
.atlas-title {
    font-family:'Playfair Display',serif; font-size:2.4rem; font-weight:900;
    color:#c9a84c; letter-spacing:-0.5px; margin:0; line-height:1;
}
.atlas-subtitle {
    font-size:0.78rem; color:#8fa3c0; letter-spacing:2.5px;
    text-transform:uppercase; margin-top:0.3rem;
}
.metric-row { display:flex; gap:1rem; margin-bottom:1.2rem; flex-wrap:wrap; }
.metric-card {
    background:linear-gradient(135deg,#142236 0%,#1c3050 100%);
    border:1px solid #2a4060; border-left:3px solid #c9a84c;
    border-radius:10px; padding:1rem 1.4rem; flex:1; min-width:140px;
}
.metric-card .mc-label { font-size:0.68rem; letter-spacing:1.5px; text-transform:uppercase; color:#7a9ab8; margin-bottom:0.2rem; }
.metric-card .mc-value { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; color:#e8e4da; line-height:1; }
.metric-card .mc-sub { font-size:0.72rem; color:#5a7a98; margin-top:0.15rem; }
.section-title {
    font-family:'Playfair Display',serif; font-size:1.25rem; font-weight:700; color:#c9a84c;
    border-bottom:1px solid #2a3d55; padding-bottom:0.4rem; margin:1.4rem 0 0.9rem;
}
[data-testid="stSidebar"] { background:#0d1b2a !important; border-right:1px solid #1e3250; }
.stButton > button {
    background:linear-gradient(135deg,#1a3050 0%,#243f60 100%) !important;
    border:1px solid #2a4060 !important; color:#c8d8e8 !important;
    font-family:'DM Sans',sans-serif !important; font-weight:500 !important;
    border-radius:8px !important; transition:all 0.2s ease; font-size:0.8rem !important;
    padding:0.4rem 0.3rem !important;
}
.stButton > button:hover { background:#c9a84c !important; color:#0b1120 !important; border-color:#c9a84c !important; }
.stSelectbox > div > div, .stMultiSelect > div > div {
    background:#0f1e30 !important; border:1px solid #2a4060 !important;
    color:#e8e4da !important; border-radius:8px !important;
}
.streamlit-expanderHeader {
    background:#0f1e30 !important; color:#c9a84c !important;
    border:1px solid #2a4060 !important; border-radius:8px !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background:#0f1e30 !important; border:1px solid #2a4060 !important;
    color:#e8e4da !important; border-radius:8px !important;
}
hr { border-color:#1e3250 !important; }
[data-testid="stChatMessage"] {
    background:#0f1e30 !important; border:1px solid #1e3250 !important; border-radius:10px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Matplotlib dark theme
# ─────────────────────────────────────────────
DARK_BG   = "#0b1120"; CARD_BG = "#0f1e30"; GOLD = "#c9a84c"
MUTED     = "#8fa3c0";  TEXT_MAIN = "#e8e4da"
A1 = "#e05c4b"; A2 = "#4b9ce8"; A3 = "#6bcb77"
BLOC_COLORS = {"LDF": A1, "UDF": A2, "NDA": "#f0a500", "Other": "#888"}
PAL = [GOLD, A1, A2, A3, "#b07aff", "#ff9f7a", "#7af0d8", "#ffde7a", "#c080ff", "#80d4c0"]

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
        "axes.edgecolor": "#2a4060", "axes.labelcolor": MUTED,
        "axes.titlecolor": GOLD, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "grid.color": "#1e3250", "grid.linestyle": "--", "grid.alpha": 0.5,
        "legend.facecolor": "#0d1b2a", "legend.edgecolor": "#2a4060",
        "legend.fontsize": 8, "legend.labelcolor": TEXT_MAIN,
        "text.color": TEXT_MAIN, "font.family": "DejaVu Sans",
        "lines.linewidth": 2.2, "patch.linewidth": 0,
    })
set_plot_style()

# ─────────────────────────────────────────────
# 2. API KEY
# ─────────────────────────────────────────────
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    api_key = None

# ─────────────────────────────────────────────
# 3. PARTY / BLOC DEFINITIONS
# ─────────────────────────────────────────────
INDIVIDUAL_PARTIES = ["CPI","CPM","INC","ML","KCM","KCJ"]
PARTY_FAMILIES = {
    "Kerala Congress Family": ["KC","KCM","KCJ","KCJB","KCB","KCS","KCST","KJS","KCAMG","KCD"],
    "Muslim League Family":   ["ML","AIML","INL"],
    "Congress Family":        ["INC","INCO","INCA","CS"],
    "Socialist Family":       ["PSP","RSP","SSP","ISP","KTP","CS","KSP","LKD","LJD","BLD","DSP","ICS","JDU","NCP","RSPB"],
    "CPM Breakaway":          ["CMP","JSS","RMP"],
}
BLOCS = {
    "LDF": ["CPM","CPI","NCP","JDU","RSP","KTP","CMP","JSS","RMP","LDF"],
    "UDF": ["INC","ML","AIML","INL","KC","KCM","KCJ","KCJB","KCB","KCS","KCST","KJS","KCAMG","KCD","INCO","INCA","CS","UDF"],
    "NDA": ["BJP","BDJS","KCP","BDP","NDA"],
}

# ─────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────
def smart_col(df, name):
    """Returns the COLUMN NAME (string) that best matches `name`."""
    if name in df.columns: return name
    for c in df.columns:
        if c.strip().lower() == name.strip().lower(): return c
    m = difflib.get_close_matches(name, df.columns, n=1, cutoff=0.45)
    return m[0] if m else name

def smart_get(df, name):
    """Returns the actual column SERIES. Use this in metric code, not smart_lookup.
    Example:  pd.to_numeric(smart_get(df, 'Win Vote'), errors='coerce') + 1
    """
    return df[smart_col(df, name)]

# Alias kept for backward compat — but AI is now instructed to use smart_get
smart_lookup = smart_col

def assign_family(p):
    for f, ms in PARTY_FAMILIES.items():
        if p in ms: return f
    return "Other"

def assign_bloc(v):
    if pd.isna(v): return "Other"
    v = str(v).strip()
    for b, ms in BLOCS.items():
        if v in ms: return b
    return "Other"

def n(s): return pd.to_numeric(s, errors='coerce')

def fmt_year(y):
    try: return str(int(float(y)))
    except: return str(y)

def mc(label, value, sub=""):
    return (f'<div class="metric-card"><div class="mc-label">{label}</div>'
            f'<div class="mc-value">{value}</div>'
            + (f'<div class="mc-sub">{sub}</div>' if sub else '') + '</div>')

def margin_cat(margin, tv):
    if tv == 0: return "Unknown"
    p = margin/tv*100
    if p > 20: return "Brute"
    elif p > 10: return "Comfortable"
    elif p > 5: return "Narrow"
    else: return "Very Thin"

# ─────────────────────────────────────────────
# 5. STATISTICS
# ─────────────────────────────────────────────
def gallagher(vp, sp):
    d = np.array(vp)-np.array(sp); return np.sqrt(0.5*np.sum(d**2))
def loosemore(vp, sp):
    return 0.5*np.sum(np.abs(np.array(vp)-np.array(sp)))
def enep(vs):
    p = np.array(vs)/100; p=p[p>0]; return 1/np.sum(p**2) if len(p) else np.nan
def enpp(ss):
    p = np.array(ss)/100; p=p[p>0]; return 1/np.sum(p**2) if len(p) else np.nan
def pedersen(d1, d2):
    pts = set(d1)|set(d2); return 0.5*sum(abs(d2.get(p,0)-d1.get(p,0)) for p in pts)
def hhi(vs):
    return np.sum((np.array(vs)/100)**2)
def frac(vs):
    p=np.array(vs)/100; return 1-np.sum(p**2)
def entropy(vs):
    p=np.array(vs)/100; p=p[p>0]; return -np.sum(p*np.log(p))

# ─────────────────────────────────────────────
# 6. DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            ext = file.name.split('.')[-1].lower()
            def proc(df, src):
                df.columns = [c.strip() for c in df.columns]
                rmap = {
                    " Elecors":"Electors","Elecors":"Electors",
                    "Votes polled ":"Votes Polled","Votes polled":"Votes Polled",
                    "Margin Win Vote-Run Vote":"Margin",
                    "Others Vote {PollVote-(Win vote+ Run Vote )}":"Others Vote",
                    "NDA/ BJP vote":"NDA BJP Vote",
                    "Run Alliance ":"Run Alliance",
                    "W Alliance":"Win Alliance",
                    "Type of Cons":"Category",
                }
                df.rename(columns=rmap, inplace=True)
                if 'Cons No.' in df.columns:
                    df['Cons No.'] = df['Cons No.'].astype(str)
                if 'Year' not in df.columns:
                    try:
                        m = re.search(r'\d{4}', str(src))
                        df['Year'] = int(m.group(0)) if m else src
                    except: df['Year'] = src
                try: df['Year'] = df['Year'].apply(lambda y: int(float(y)) if pd.notna(y) else y)
                except: pass
                return df
            if ext in ['xlsx','xls']:
                for sheet, df in pd.read_excel(file, sheet_name=None).items():
                    all_dfs.append(proc(df, sheet))
            elif ext == 'csv':
                all_dfs.append(proc(pd.read_csv(file), file.name))
        except Exception as e:
            st.error(f"Error: {file.name}: {e}")
    if not all_dfs: return None
    df = pd.concat(all_dfs, ignore_index=True)
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    wc = smart_col(df, "Win Party")
    if wc in df.columns:
        df["Party Family"] = df[wc].astype(str).apply(assign_family)
    wa = smart_col(df, "Win Alliance")
    if wa in df.columns:
        df["Bloc"] = df[wa].astype(str).apply(assign_bloc)
    elif wc in df.columns:
        df["Bloc"] = df[wc].astype(str).apply(assign_bloc)
    return df

# ─────────────────────────────────────────────
# 7. GEMINI
# ─────────────────────────────────────────────
def gen_metric_code(df, name, desc):
    if not api_key: return "# No API key"
    # Send only column names + numeric dtypes — skip object cols to save tokens
    num_cols = {c: str(df[c].dtype) for c in df.columns if df[c].dtype in ['float64','int64','int32','float32']}
    all_cols = list(df.columns)
    p = (f"ONE-LINE pandas assignment. df['{name}']=...\n"
         f"Cols:{all_cols}\nNumeric:{num_cols}\nLogic:{desc}\n"
         f"Rules:smart_get(df,'Col') returns Series. pd.to_numeric(x,errors='coerce') for math.\n"
         f"Ex:df['m']=pd.to_numeric(smart_get(df,'Win Vote'),errors='coerce')/pd.to_numeric(smart_get(df,'Votes Polled'),errors='coerce')*100\n"
         f"Output:single assignment line only,no comments,no imports,no markdown")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash',
        generation_config={"temperature":0,"max_output_tokens":120,"candidate_count":1})
    code = ""
    for attempt in range(3):
        try:
            code = model.generate_content(p).text.replace("```python","").replace("```","").strip()
            # strip any accidental multi-line output — take only the assignment line
            code = next((l for l in code.splitlines() if l.strip().startswith(f"df['{name}']")), code)
            test_df = df.head(5).copy()
            exec(code, {"df":test_df,"pd":pd,"np":np,"smart_get":smart_get,"smart_lookup":smart_col})
            if name in test_df.columns and test_df[name].notna().any():
                return code
            p += f"\nFAIL:all-None. Code:{code}\nRemember smart_get returns Series not name."
        except Exception as e:
            p += f"\nERR:{e} Code:{code}\nFix."
    return code

def query_ai(query, df):
    if not api_key: return None, None, "No API key."
    genai.configure(api_key=api_key)
    try: sample = df.sample(n=5).to_markdown()
    except: sample = df.head(5).to_markdown()
    p = f"""Expert Kerala Election Analyst. DataFrame `df` available.
Use smart_lookup(df,'col') for columns. For plots: fig=. For text: answer=.
Columns: {list(df.columns)}\nSample:\n{sample}\nUSER: {query}"""
    model = genai.GenerativeModel('gemini-2.0-flash')
    for _ in range(3):
        try:
            code = model.generate_content(p).text.replace("```python","").replace("```","").strip()
            g = {"df":df,"plt":plt,"sns":sns,"pd":pd,"np":np,"smart_lookup":smart_col,"fig":None,"answer":None}
            exec(code, g)
            return g.get('fig'), g.get('answer'), code
        except Exception as e:
            p += f"\nError: {e}. Fix."
    return None, None, "Failed."

# ─────────────────────────────────────────────
# 8. PAGE RENDERERS
# ─────────────────────────────────────────────

# ── HD Plot Export helper ──────────────────────────────
def save_fig_hd(fig, name="plot"):
    """Return a bytes buffer of the figure at 200dpi for download."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def hd_download(fig, label="📥 Download HD", key="dl"):
    """Inline HD download button below a chart."""
    data = save_fig_hd(fig)
    st.download_button(label, data, file_name=f"{key}.png",
                       mime="image/png", key=key)

# ── Plot store for batch export ────────────────────────
if "plot_store" not in st.session_state:
    st.session_state.plot_store = {}          # {label: png_bytes}

def store_plot(fig, label):
    """Register a figure into the plot store for batch download."""
    st.session_state.plot_store[label] = save_fig_hd(fig, label)


# ─────────────────────────────────────────────
# 8. PAGE RENDERERS
# ─────────────────────────────────────────────

def page_overview(df):
    wc = smart_col(df,"Win Party"); tv = smart_col(df,"Votes Polled"); el = smart_col(df,"Electors")
    mc_col = smart_col(df,"Constituency Name")
    years = sorted(df['Year'].unique())
    tot_v  = n(df[tv]).sum() if tv in df.columns else 0
    tot_el = n(df[el]).sum() if el in df.columns else 0
    turnout = tot_v/tot_el*100 if tot_el>0 else 0
    np_    = df[wc].nunique() if wc in df.columns else "—"
    tot_s  = df[mc_col].nunique() if mc_col in df.columns else "—"

    st.markdown('<div class="metric-row">'
        + mc("Elections", str(len(years)), f"{fmt_year(min(years))} – {fmt_year(max(years))}")
        + mc("Constituencies", str(tot_s), "unique seats")
        + mc("Total Votes", f"{tot_v/1e6:.1f}M", "across all elections")
        + mc("Avg Turnout", f"{turnout:.1f}%", "votes / electors")
        + mc("Parties Won Seats", str(np_), "distinct parties")
        + '</div>', unsafe_allow_html=True)

    if wc not in df.columns: return
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-title">Seats Won by Bloc — All Elections</div>', unsafe_allow_html=True)
        if 'Bloc' in df.columns:
            by = df.groupby(['Year','Bloc']).size().unstack(fill_value=0)
            fig, ax = plt.subplots(figsize=(9,4))
            bottom = np.zeros(len(by))
            for b in ['LDF','UDF','NDA','Other']:
                if b in by.columns:
                    v = by[b].values
                    ax.bar(by.index.astype(str), v, bottom=bottom, color=BLOC_COLORS.get(b,"#888"), label=b, width=0.65)
                    bottom += v
            ax.set_xlabel("Year"); ax.set_ylabel("Seats"); ax.legend(); ax.grid(axis='y',alpha=0.3)
            plt.tight_layout(); st.pyplot(fig)
            store_plot(fig, "bloc_seats_overview"); plt.close()

    with col2:
        st.markdown('<div class="section-title">Top 10 Parties by Total Seats</div>', unsafe_allow_html=True)
        tp = df[wc].value_counts().head(10).reset_index()
        tp.columns = ["Party","Seats"]
        fig2, ax2 = plt.subplots(figsize=(5,4))
        colors = [GOLD if i==0 else A2 if i==1 else A1 if i==2 else MUTED for i in range(len(tp))]
        bars = ax2.barh(tp["Party"][::-1], tp["Seats"][::-1], color=colors[::-1])
        ax2.bar_label(bars, fmt='%d', padding=3, color=TEXT_MAIN, fontsize=8)
        ax2.set_xlabel("Total Seats")
        plt.tight_layout(); st.pyplot(fig2)
        store_plot(fig2, "top_parties_seats"); plt.close()

    if tv in df.columns and el in df.columns:
        st.markdown('<div class="section-title">Voter Turnout Trend (%)</div>', unsafe_allow_html=True)
        tr = df.groupby('Year').apply(
            lambda g: n(g[tv]).sum()/n(g[el]).sum()*100 if n(g[el]).sum()>0 else np.nan
        ).reset_index(name="Turnout %")
        fig3, ax3 = plt.subplots(figsize=(12,3))
        ax3.fill_between(tr['Year'].astype(str), tr['Turnout %'], alpha=0.2, color=GOLD)
        ax3.plot(tr['Year'].astype(str), tr['Turnout %'], marker='o', color=GOLD)
        for _, row in tr.iterrows():
            if pd.notna(row['Turnout %']):
                ax3.annotate(f"{row['Turnout %']:.1f}%", (str(row['Year']),row['Turnout %']),
                             textcoords="offset points",xytext=(0,6),ha='center',fontsize=7.5,color=GOLD)
        ax3.set_ylim(50,90); ax3.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig3)
        store_plot(fig3, "turnout_trend"); plt.close()

    # Heatmap
    st.markdown('<div class="section-title">Party Vote Share Heatmap (Top Parties)</div>', unsafe_allow_html=True)
    if tv in df.columns:
        top_parties_list = df[wc].value_counts().head(8).index.tolist()
        heat_rows = []
        for yr, grp in df.groupby('Year'):
            tv_sum = n(grp[tv]).sum()
            for p in top_parties_list:
                pv = n(grp[grp[wc]==p][tv]).sum()
                heat_rows.append({"Year": yr, "Party": p, "Vote %": pv/tv_sum*100 if tv_sum>0 else 0})
        hdf = pd.DataFrame(heat_rows).pivot(index='Party', columns='Year', values='Vote %').fillna(0)
        hdf.columns = [str(c) for c in hdf.columns]
        fig4, ax4 = plt.subplots(figsize=(14,4))
        sns.heatmap(hdf, cmap='YlOrBr', annot=True, fmt='.1f', linewidths=0.3,
                    linecolor='#1e3250', ax=ax4, cbar_kws={'shrink':0.6})
        ax4.set_title("Vote % by Party and Election Year")
        ax4.tick_params(axis='x', rotation=45)
        plt.tight_layout(); st.pyplot(fig4)
        store_plot(fig4, "vote_share_heatmap"); plt.close()


def page_dashboard(df):
    """Interactive chart builder — any metric, any chart type, any grouping."""
    st.markdown('<div class="section-title">📊 Interactive Dashboard</div>', unsafe_allow_html=True)

    wc  = smart_col(df,"Win Party");  tv  = smart_col(df,"Votes Polled")
    el  = smart_col(df,"Electors");   mg  = smart_col(df,"Margin")
    cc  = smart_col(df,"Constituency Name")

    # Build metric list: numeric columns + custom metrics
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    base_metrics = [c for c in [tv, el, mg, "Win Vote", "Run vote", "NDA BJP Vote", "Others Vote"]
                    if c in df.columns]
    custom_cols  = [c for c in df.columns if c not in [tv,el,mg,"Year","Cons No.","Cons No."]
                    and df[c].dtype in ['float64','int64'] and c not in base_metrics]
    metric_opts  = list(dict.fromkeys(base_metrics + custom_cols))

    cat_opts = ["Win Party","Bloc","Party Family","Win Alliance","Category","Constituency Name","Year"]
    cat_opts = [smart_col(df,c) for c in cat_opts if smart_col(df,c) in df.columns]

    # ── Controls ───────────────────────────────
    c1,c2,c3,c4,c5 = st.columns([2,2,2,2,2])
    with c1:
        chart_type = st.selectbox("Chart Type", ["📈 Line","📊 Bar","🥧 Pie","📦 Box","🔥 Heatmap"], key="db_chart")
    with c2:
        metric = st.selectbox("Metric / Y-axis", metric_opts, key="db_metric") if metric_opts else None
    with c3:
        split_by = st.selectbox("Group / Split by", cat_opts, key="db_split") if cat_opts else None
    with c4:
        agg = st.selectbox("Aggregation", ["Sum","Mean","Max","Count"], key="db_agg")
    with c5:
        top_n = st.slider("Top N categories", 3, 20, 10, key="db_topn")

    if not metric:
        st.info("No numeric columns detected yet."); return

    agg_map = {"Sum":"sum","Mean":"mean","Max":"max","Count":"count"}
    af = agg_map[agg]

    df2 = df.copy()
    df2[metric] = pd.to_numeric(df2[metric], errors='coerce')

    fig = None

    try:
        # ── LINE ──────────────────────────────
        if "Line" in chart_type:
            grp = ["Year"] + ([split_by] if split_by and split_by != "Year" else [])
            data = df2.groupby(grp)[metric].agg(af).reset_index()
            fig, ax = plt.subplots(figsize=(12,4))
            if split_by and split_by != "Year":
                top_cats = df2[split_by].value_counts().head(top_n).index
                for i,cat in enumerate(top_cats):
                    sub = data[data[split_by]==cat].sort_values("Year")
                    ax.plot(sub["Year"].astype(str), sub[metric], marker='o',
                            color=PAL[i%len(PAL)], label=str(cat), linewidth=2)
                ax.legend(fontsize=7, ncol=2)
            else:
                data = data.sort_values("Year")
                ax.fill_between(data["Year"].astype(str), data[metric], alpha=0.15, color=GOLD)
                ax.plot(data["Year"].astype(str), data[metric], marker='o', color=GOLD, linewidth=2.5)
            ax.set_title(f"{agg} of {metric} over Time"); ax.set_xlabel("Year")

        # ── BAR ──────────────────────────────
        elif "Bar" in chart_type:
            if not split_by: st.warning("Select a Group category for bar charts."); return
            data = df2.groupby(split_by)[metric].agg(af).reset_index()
            data = data.dropna().sort_values(metric, ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(10,max(4, len(data)*0.4)))
            colors = [BLOC_COLORS.get(assign_bloc(p), PAL[i%len(PAL)]) for i,p in enumerate(data[split_by])]
            bars = ax.barh(data[split_by][::-1], data[metric][::-1], color=colors[::-1])
            ax.bar_label(bars, fmt='%.0f', padding=3, color=TEXT_MAIN, fontsize=8)
            ax.set_title(f"Top {top_n} {split_by} by {agg} {metric}")
            ax.set_xlabel(f"{agg} {metric}")

        # ── PIE ──────────────────────────────
        elif "Pie" in chart_type:
            if not split_by: st.warning("Select a Group category for pie charts."); return
            data = df2.groupby(split_by)[metric].agg(af).reset_index()
            data = data.dropna().sort_values(metric, ascending=False).head(top_n)
            fig, ax = plt.subplots(figsize=(8,6))
            wedge_colors = [BLOC_COLORS.get(assign_bloc(p), PAL[i%len(PAL)]) for i,p in enumerate(data[split_by])]
            wedges, texts, autotexts = ax.pie(
                data[metric], labels=data[split_by], autopct='%1.1f%%',
                startangle=140, colors=wedge_colors,
                wedgeprops={'edgecolor':'#0b1120','linewidth':1.5}
            )
            for t in autotexts: t.set_color(DARK_BG); t.set_fontsize(8)
            for t in texts:     t.set_color(TEXT_MAIN); t.set_fontsize(8)
            ax.set_title(f"{metric} share by {split_by} (Top {top_n})")

        # ── BOX ──────────────────────────────
        elif "Box" in chart_type:
            if split_by:
                top_cats = df2[split_by].value_counts().head(top_n).index
                sub = df2[df2[split_by].isin(top_cats)]
                fig, ax = plt.subplots(figsize=(12,5))
                sns.boxplot(data=sub, x=split_by, y=metric, ax=ax,
                            palette={c: PAL[i%len(PAL)] for i,c in enumerate(top_cats)},
                            order=top_cats)
                plt.xticks(rotation=35, ha='right')
            else:
                fig, ax = plt.subplots(figsize=(5,5))
                sns.boxplot(y=df2[metric], ax=ax, color=GOLD)
            ax.set_title(f"Distribution of {metric}")

        # ── HEATMAP ──────────────────────────
        elif "Heatmap" in chart_type:
            if not split_by: st.warning("Select a Group for heatmap."); return
            top_cats = df2[split_by].value_counts().head(top_n).index
            sub = df2[df2[split_by].isin(top_cats)]
            pivot = sub.groupby(["Year", split_by])[metric].agg(af).unstack(fill_value=0)
            pivot.columns = [str(c) for c in pivot.columns]
            fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)*0.9), max(4, len(pivot)*0.5)))
            sns.heatmap(pivot.T, cmap='YlOrBr', annot=True, fmt='.0f',
                        linewidths=0.3, linecolor='#1e3250', ax=ax)
            ax.set_title(f"{agg} {metric} by Year and {split_by}")
            ax.tick_params(axis='x', rotation=45)

        if fig:
            plt.tight_layout()
            st.pyplot(fig)
            plot_key = f"dashboard_{chart_type.split()[1].lower()}_{metric}_{split_by}"
            store_plot(fig, plot_key)
            hd_download(fig, "📥 Download this chart (HD)", key=f"dldash_{plot_key[:30]}")
            plt.close()

    except Exception as e:
        st.error(f"Chart error: {e}")

    # ── Custom metric columns visible as table ──
    custom_metric_cols = list(st.session_state.custom_metrics.keys())
    if custom_metric_cols:
        st.markdown('<div class="section-title">📐 Custom Metric Preview</div>', unsafe_allow_html=True)
        show_cols = ["Year", smart_col(df,"Constituency Name")] + \
                    [c for c in custom_metric_cols if c in df.columns]
        show_cols = [c for c in show_cols if c in df.columns]
        if show_cols:
            st.dataframe(df[show_cols].head(30), use_container_width=True, hide_index=True)
        else:
            st.info("Custom metrics not yet visible — save them in the Custom Metrics tab first.")

def page_parties(df):
    wc = smart_col(df,"Win Party"); tv = smart_col(df,"Votes Polled")
    mc_c = smart_col(df,"Constituency Name"); mg = smart_col(df,"Margin")
    if wc not in df.columns: st.warning("Win Party not found"); return

    all_p = sorted(df[wc].dropna().unique())
    default = [p for p in INDIVIDUAL_PARTIES if p in all_p]
    sel = st.multiselect("Select parties to analyse", all_p, default=default, key="psel")
    if not sel: st.info("Select at least one party."); return

    # Stats table
    rows = []
    total_el = df['Year'].nunique()
    for party in sel:
        for yr, grp in df.groupby('Year'):
            ts = grp[mc_c].nunique() if mc_c in grp.columns else len(grp)
            won = (grp[wc]==party).sum()
            tv_sum = n(grp[tv]).sum() if tv in grp.columns else np.nan
            pv  = n(grp[grp[wc]==party][tv]).sum() if (tv in grp.columns and won>0) else 0
            vp  = pv/tv_sum*100 if tv_sum>0 else np.nan
            sr  = won/ts*100 if ts>0 else np.nan
            rows.append({"Party":party,"Year":yr,"Seats Won":int(won),
                         "Strike Rate %":round(sr,1) if pd.notna(sr) else np.nan,
                         "Vote %":round(vp,2) if pd.notna(vp) else np.nan})
    sr_df = pd.DataFrame(rows)
    st.markdown('<div class="section-title">Election-by-Election Stats</div>', unsafe_allow_html=True)
    st.dataframe(sr_df, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(7,4))
        for i,p in enumerate(sel):
            sub=sr_df[sr_df["Party"]==p].sort_values("Year")
            ax.plot(sub["Year"].astype(str),sub["Seats Won"],marker='o',color=PAL[i%len(PAL)],label=p)
        ax.set_title("Seats Won"); ax.legend(); ax.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig2,ax2=plt.subplots(figsize=(7,4))
        for i,p in enumerate(sel):
            sub=sr_df[sr_df["Party"]==p].sort_values("Year")
            ax2.plot(sub["Year"].astype(str),sub["Vote %"],marker='s',linestyle='--',color=PAL[i%len(PAL)],label=p)
        ax2.set_title("Vote %"); ax2.legend(); ax2.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Win patterns
    st.markdown('<div class="section-title">Constituency Win Patterns</div>', unsafe_allow_html=True)
    for party in sel:
        wons = df[df[wc]==party]
        if mc_c not in df.columns or wons.empty: continue
        cnts = wons[mc_c].value_counts()
        pdf = pd.DataFrame({"Constituency":cnts.index,"Times Won":cnts.values,
                             "Win Rate %":(cnts.values/total_el*100).round(1)})
        pdf["Pattern"] = pdf["Win Rate %"].apply(
            lambda r: "🏰 Stronghold" if r>=75 else "🏘️ Neighbourhood" if r>=50 else "🎲 Chance" if r>=25 else "⚔️ Hostile")
        with st.expander(f"**{party}** — {len(pdf)} constituencies"):
            st.dataframe(pdf.sort_values("Times Won",ascending=False), use_container_width=True, hide_index=True)

    # Margin distribution
    if mg in df.columns and tv in df.columns:
        st.markdown('<div class="section-title">Victory Margin Distribution</div>', unsafe_allow_html=True)
        fig3,ax3=plt.subplots(figsize=(10,4))
        for i,party in enumerate(sel):
            data=df[df[wc]==party]
            mgns=n(data[mg]).dropna()
            tvs=n(data[tv]).reindex(mgns.index)
            pct=(mgns/tvs*100).dropna()
            if not pct.empty:
                pct.hist(bins=20,alpha=0.6,label=party,color=PAL[i%len(PAL)],ax=ax3)
        ax3.axvline(5,color=A1,linestyle='--',alpha=0.5,label='5%')
        ax3.axvline(20,color=A3,linestyle='--',alpha=0.5,label='20%')
        ax3.set_xlabel("Margin as % of Votes"); ax3.legend()
        plt.tight_layout(); st.pyplot(fig3); plt.close()


def page_families(df):
    if "Party Family" not in df.columns: st.warning("Party Family not computed."); return
    tv = smart_col(df,"Votes Polled")
    seats = df.groupby(['Year','Party Family']).size().unstack(fill_value=0)
    st.markdown('<div class="section-title">Seats Won by Party Family</div>', unsafe_allow_html=True)
    st.dataframe(seats, use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for i,fam in enumerate(seats.columns):
            ax.plot(seats.index.astype(str),seats[fam],marker='o',color=PAL[i%len(PAL)],label=fam)
        ax.set_title("Seats by Family"); ax.legend(fontsize=7); ax.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        if tv in df.columns:
            fv=df.groupby(['Year','Party Family'])[tv].apply(lambda x: n(x).sum()).unstack(fill_value=0)
            tot=df.groupby('Year')[tv].apply(lambda x: n(x).sum())
            fvp=fv.div(tot,axis=0)*100
            fig2,ax2=plt.subplots(figsize=(7,4.5))
            for i,fam in enumerate(fvp.columns):
                ax2.plot(fvp.index.astype(str),fvp[fam],marker='s',linestyle='--',color=PAL[i%len(PAL)],label=fam)
            ax2.set_title("Vote % by Family"); ax2.legend(fontsize=7); ax2.set_xlabel("Year")
            plt.tight_layout(); st.pyplot(fig2); plt.close()


def page_blocs(df):
    tv=smart_col(df,"Votes Polled"); mc_c=smart_col(df,"Constituency Name")
    if 'Bloc' not in df.columns: st.warning("Bloc not computed."); return
    rows=[]
    for yr,grp in df.groupby('Year'):
        ts=grp[mc_c].nunique() if mc_c in grp.columns else len(grp)
        tvs=n(grp[tv]).sum() if tv in grp.columns else np.nan
        for b in ['LDF','UDF','NDA']:
            sub=grp[grp['Bloc']==b]
            s=len(sub); v=n(sub[tv]).sum() if tv in sub.columns else 0
            rows.append({"Year":yr,"Bloc":b,"Seats Won":s,
                         "Vote %":round(v/tvs*100,2) if tvs and tvs>0 else np.nan,
                         "Seat %":round(s/ts*100,2) if ts>0 else np.nan})
    bdf=pd.DataFrame(rows)
    st.dataframe(bdf, use_container_width=True, hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        fig,ax=plt.subplots(figsize=(7,4))
        for b in ['LDF','UDF','NDA']:
            sub=bdf[bdf['Bloc']==b].sort_values('Year')
            ax.plot(sub['Year'].astype(str),sub['Seats Won'],marker='o',color=BLOC_COLORS[b],label=b,linewidth=2.5)
        ax.set_title("Seats Won"); ax.legend(); ax.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        fig2,ax2=plt.subplots(figsize=(7,4))
        for b in ['LDF','UDF','NDA']:
            sub=bdf[bdf['Bloc']==b].sort_values('Year')
            ax2.plot(sub['Year'].astype(str),sub['Vote %'],marker='s',linestyle='--',color=BLOC_COLORS[b],label=b,linewidth=2.5)
        ax2.set_title("Vote %"); ax2.legend(); ax2.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Alternation chart
    st.markdown('<div class="section-title">Kerala Alternation Pattern (LDF ↔ UDF)</div>', unsafe_allow_html=True)
    alt=bdf[bdf['Bloc'].isin(['LDF','UDF'])].pivot(index='Year',columns='Bloc',values='Seats Won').fillna(0)
    if 'LDF' in alt.columns and 'UDF' in alt.columns:
        fig3,ax3=plt.subplots(figsize=(12,3.5))
        yrs=alt.index.astype(str)
        ax3.bar(yrs,alt['LDF'],color=BLOC_COLORS['LDF'],label='LDF',alpha=0.85)
        ax3.bar(yrs,-alt['UDF'],color=BLOC_COLORS['UDF'],label='UDF',alpha=0.85)
        ax3.axhline(0,color=TEXT_MAIN,linewidth=0.8)
        ax3.set_ylabel("Seats (LDF up / UDF down)"); ax3.legend()
        plt.tight_layout(); st.pyplot(fig3); plt.close()


def page_stats(df):
    wc=smart_col(df,"Win Party"); tv=smart_col(df,"Votes Polled")
    el=smart_col(df,"Electors"); mc_c=smart_col(df,"Constituency Name"); mg=smart_col(df,"Margin")

    with st.expander("📖 Index Glossary"):
        st.markdown("""
| Index | Measures |
|---|---|
| **Gallagher** | Vote→Seat distortion |
| **Loosemore-Hanby** | Total over/under-representation |
| **Pedersen** | Electoral volatility between elections |
| **ENEP / ENPP** | Effective # electoral / parliamentary parties |
| **Turnout %** | Votes / Electors |
| **Fractionalization** | Party system fragmentation |
| **HHI** | Vote concentration |
| **Close Contests** | Seats won by <5% margin |
        """)

    if wc not in df.columns or tv not in df.columns: st.warning("Need Win Party + Votes Polled."); return
    all_years=sorted(df['Year'].unique()); rows=[]
    for yr in all_years:
        grp=df[df['Year']==yr]; ts=grp[mc_c].nunique() if mc_c in grp.columns else len(grp)
        tvs=n(grp[tv]).sum()
        pw=grp[wc].value_counts(); pv=grp.groupby(wc)[tv].apply(lambda x: n(x).sum())
        sp=(pw/ts*100).values.tolist(); vp=(pv/tvs*100).values.tolist() if tvs>0 else []
        nl=min(len(sp),len(vp)); sp,vp=sp[:nl],vp[:nl]
        row={"Year":yr}
        if nl>0:
            row["Gallagher"]=round(gallagher(vp,sp),3)
            row["Loosemore-Hanby"]=round(loosemore(vp,sp),3)
            row["ENEP"]=round(enep(vp),3); row["ENPP"]=round(enpp(sp),3)
            row["HHI"]=round(hhi(vp),4); row["Frac."]=round(frac(vp),3)
        if el in grp.columns:
            te=n(grp[el]).sum()
            row["Turnout %"]=round(tvs/te*100,2) if te>0 else np.nan
        if mg in grp.columns:
            ms=n(grp[mg]); tvss=n(grp[tv])
            row["Avg Margin"]=round(ms.mean(),0)
            row["Close (<5%)"]=int(((ms/tvss)<0.05).sum())
        rows.append(row)
    idx=pd.DataFrame(rows).set_index("Year")
    st.dataframe(idx, use_container_width=True)

    # Pedersen
    if len(all_years)>=2:
        pr=[]
        for i in range(1,len(all_years)):
            y1,y2=all_years[i-1],all_years[i]
            g1=df[df['Year']==y1]; g2=df[df['Year']==y2]
            t1=n(g1[tv]).sum(); t2=n(g2[tv]).sum()
            if t1>0 and t2>0:
                pts=set(g1[wc].unique())|set(g2[wc].unique())
                d1={p:n(g1[g1[wc]==p][tv]).sum()/t1*100 for p in pts}
                d2={p:n(g2[g2[wc]==p][tv]).sum()/t2*100 for p in pts}
                pr.append({"Year":y2,"Pedersen":round(pedersen(d1,d2),3)})
        if pr:
            pf=pd.DataFrame(pr)
            fig,ax=plt.subplots(figsize=(10,3))
            ax.bar(pf['Year'].astype(str),pf['Pedersen'],color=A1,alpha=0.85)
            ax.set_title("Electoral Volatility (Pedersen Index)"); ax.set_xlabel("Year")
            plt.tight_layout(); st.pyplot(fig); plt.close()

    pcols=[c for c in ["Gallagher","ENEP","ENPP","Turnout %","Frac."] if c in idx.columns]
    if pcols:
        fig2,axes=plt.subplots(1,len(pcols),figsize=(14,3.5))
        if len(pcols)==1: axes=[axes]
        for ax,col in zip(axes,pcols):
            ax.plot(idx.index.astype(str),idx[col],marker='o',color=GOLD,linewidth=2)
            ax.fill_between(idx.index.astype(str),idx[col],alpha=0.15,color=GOLD)
            ax.set_title(col); ax.tick_params(axis='x',rotation=45)
        plt.tight_layout(); st.pyplot(fig2); plt.close()


def page_swing(df):
    cc=smart_col(df,"Constituency Name"); wc=smart_col(df,"Win Party")
    tv=smart_col(df,"Votes Polled"); mg=smart_col(df,"Margin")
    years=sorted(df['Year'].unique())
    if len(years)<2: st.warning("Need ≥2 years."); return

    c1,c2=st.columns(2)
    # Use integer years in selectbox — fix for decimal display
    year_opts = [str(y) for y in years]
    with c1: ya=st.selectbox("Baseline Year", year_opts, index=0, key="ya")
    with c2: yb=st.selectbox("Target Year",   year_opts, index=len(year_opts)-1, key="yb")
    ya,yb=int(ya),int(yb)
    if ya==yb: st.error("Select two different years."); return

    da=df[df['Year']==ya]; db=df[df['Year']==yb]
    va=n(da[tv]).sum() if tv in da.columns else 0
    vb=n(db[tv]).sum() if tv in db.columns else 0
    st.markdown('<div class="metric-row">'
        + mc(f"Votes {ya}", f"{va/1e6:.2f}M")
        + mc(f"Votes {yb}", f"{vb/1e6:.2f}M")
        + mc("Vote Δ", f"{(vb-va)/1e6:+.2f}M")
        + '</div>', unsafe_allow_html=True)

    if not all(c in df.columns for c in [cc,wc]): return
    ma=da[[cc,wc]].rename(columns={wc:f"Winner {ya}"})
    mb=db[[cc,wc]].rename(columns={wc:f"Winner {yb}"})
    comp=pd.merge(ma,mb,on=cc,how="inner")
    flipped=comp[comp[f"Winner {ya}"]!=comp[f"Winner {yb}"]].copy()

    st.markdown(f'<div class="section-title">Constituency Flips — {len(flipped)} of {len(comp)} seats changed</div>', unsafe_allow_html=True)
    if not flipped.empty:
        flipped["Bloc Before"]=flipped[f"Winner {ya}"].apply(assign_bloc)
        flipped["Bloc After"]=flipped[f"Winner {yb}"].apply(assign_bloc)
        st.dataframe(flipped, use_container_width=True, hide_index=True)
        c1,c2=st.columns(2)
        with c1:
            g=flipped[f"Winner {yb}"].value_counts().head(10).reset_index()
            g.columns=["Party","Gained"]
            fig,ax=plt.subplots(figsize=(6,4))
            sns.barplot(data=g,x="Gained",y="Party",palette="magma",ax=ax)
            ax.set_title(f"Gained in {yb}"); plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            lo=flipped[f"Winner {ya}"].value_counts().head(10).reset_index()
            lo.columns=["Party","Lost"]
            fig2,ax2=plt.subplots(figsize=(6,4))
            sns.barplot(data=lo,x="Lost",y="Party",palette="flare",ax=ax2)
            ax2.set_title(f"Lost in {yb}"); plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown('<div class="section-title">Transition Matrix</div>', unsafe_allow_html=True)
    tm=pd.crosstab(comp[f"Winner {ya}"],comp[f"Winner {yb}"])
    fig_tm,ax_tm=plt.subplots(figsize=(min(16,max(7,len(tm.columns)*0.9)),min(14,max(5,len(tm.index)*0.8))))
    sns.heatmap(tm,annot=True,fmt='d',cmap='YlOrBr',linewidths=0.3,linecolor='#1e3250',ax=ax_tm)
    ax_tm.set_title(f"Seat Transition: {ya} → {yb}")
    plt.tight_layout(); st.pyplot(fig_tm); plt.close()

    if mg in df.columns:
        st.markdown('<div class="section-title">Margin Distributions</div>', unsafe_allow_html=True)
        fig_m,ax_m=plt.subplots(figsize=(10,3.5))
        n(da[mg]).dropna().hist(bins=25,alpha=0.65,label=str(ya),ax=ax_m,color=A2)
        n(db[mg]).dropna().hist(bins=25,alpha=0.65,label=str(yb),ax=ax_m,color=A1)
        ax_m.set_xlabel("Margin (Votes)"); ax_m.legend()
        plt.tight_layout(); st.pyplot(fig_m); plt.close()


def page_constituency(df):
    cc=smart_col(df,"Constituency Name"); wc=smart_col(df,"Win Party")
    tv=smart_col(df,"Votes Polled"); el=smart_col(df,"Electors"); mg=smart_col(df,"Margin")
    cat=smart_col(df,"Category")
    if cc not in df.columns: st.warning("No Constituency Name column."); return

    sel=st.selectbox("Select Constituency", sorted(df[cc].dropna().unique()), key="csel")
    cdf=df[df[cc]==sel].sort_values('Year')
    if cdf.empty: st.info("No data."); return

    catv=cdf[cat].iloc[-1] if cat in cdf.columns else "—"
    i1,i2,i3,i4=st.columns(4)
    i1.metric("Category", str(catv))
    i2.metric("Elections", str(len(cdf)))
    if tv in cdf.columns: i3.metric("Latest Votes Polled", f"{n(cdf[tv]).iloc[-1]:,.0f}")
    if wc in cdf.columns: i4.metric("Last Winner", str(cdf[wc].iloc[-1]))

    show=[c for c in ['Year','Win Party','Win Alliance','Win Vote','Run Party','Run Alliance','Run vote','Margin','Votes Polled'] if c in cdf.columns]
    st.dataframe(cdf[show], use_container_width=True, hide_index=True)

    c1,c2=st.columns(2)
    with c1:
        if wc in cdf.columns:
            wins=cdf[wc].value_counts()
            fig,ax=plt.subplots(figsize=(6,3.5))
            colors=[BLOC_COLORS.get(assign_bloc(p),MUTED) for p in wins.index]
            ax.bar(wins.index,wins.values,color=colors)
            ax.set_title(f"{sel} — Wins by Party"); plt.xticks(rotation=30,ha='right')
            plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        if mg in cdf.columns:
            fig2,ax2=plt.subplots(figsize=(6,3.5))
            ax2.bar(cdf['Year'].astype(str),n(cdf[mg]),color=GOLD,alpha=0.8)
            ax2.set_title(f"{sel} — Victory Margin")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    if tv in cdf.columns and el in cdf.columns:
        to=(n(cdf[tv])/n(cdf[el])*100).fillna(0)
        fig3,ax3=plt.subplots(figsize=(12,2.5))
        ax3.fill_between(cdf['Year'].astype(str),to,alpha=0.25,color=A3)
        ax3.plot(cdf['Year'].astype(str),to,marker='o',color=A3,linewidth=2)
        ax3.set_title(f"{sel} — Turnout %"); ax3.set_ylim(40,100)
        plt.tight_layout(); st.pyplot(fig3); plt.close()


def page_regional(df):
    dc=smart_col(df,"District"); wc=smart_col(df,"Win Party"); tv=smart_col(df,"Votes Polled")
    NORTH=["Kasaragod","Kannur","Wayanad","Kozhikode","Malappuram"]
    SOUTH=["Thiruvananthapuram","Kollam","Pathanamthitta","Alappuzha"]
    CENTRAL=["Thrissur","Palakkad","Ernakulam","Idukki","Kottayam"]
    def region(d):
        if d in NORTH: return "North"
        if d in SOUTH: return "South"
        if d in CENTRAL: return "Central"
        return "Other"

    if dc not in df.columns:
        st.info("No 'District' column. Showing top-party breakdown.")
        if wc in df.columns:
            tp=df.groupby(['Year',wc]).size().reset_index(name='S')
            tp=tp[tp['S']>=3].pivot(index='Year',columns=wc,values='S').fillna(0)
            st.dataframe(tp.astype(int), use_container_width=True)
        return

    df2=df.copy(); df2['Region']=df2[dc].astype(str).apply(region)
    for party in ["INC","CPM","CPI"]:
        if wc not in df2.columns: continue
        sub=df2[df2[wc]==party].groupby(['Year','Region']).size().unstack(fill_value=0)
        if sub.empty: continue
        st.markdown(f'<div class="section-title">{party} — Regional Wins</div>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(10,3.5))
        rc={'North':A2,'Central':GOLD,'South':A3,'Other':MUTED}
        for r in ['North','Central','South','Other']:
            if r in sub.columns:
                ax.plot(sub.index.astype(str),sub[r],marker='o',label=r,color=rc.get(r,MUTED))
        ax.set_title(f"{party} Wins by Region"); ax.legend(); ax.set_xlabel("Year")
        plt.tight_layout(); st.pyplot(fig); plt.close()


def page_reserved(df):
    cat=smart_col(df,"Category"); wc=smart_col(df,"Win Party")
    if cat not in df.columns: st.info("No 'Category' column."); return
    if wc not in df.columns: st.info("No Win Party column."); return
    res=df[df[cat].astype(str).str.upper().isin(["SC","ST"])]
    gen=df[~df[cat].astype(str).str.upper().isin(["SC","ST"])]
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Reserved (SC/ST)</div>', unsafe_allow_html=True)
        if not res.empty:
            tr=res[wc].value_counts().head(12).reset_index(); tr.columns=["Party","Seats"]
            fig,ax=plt.subplots(figsize=(5,4))
            ax.barh(tr["Party"][::-1],tr["Seats"][::-1],color=A1)
            ax.set_xlabel("Seats"); plt.tight_layout(); st.pyplot(fig); plt.close()
            st.dataframe(tr, use_container_width=True, hide_index=True)
    with c2:
        st.markdown('<div class="section-title">General Seats</div>', unsafe_allow_html=True)
        if not gen.empty:
            tg=gen[wc].value_counts().head(12).reset_index(); tg.columns=["Party","Seats"]
            fig2,ax2=plt.subplots(figsize=(5,4))
            ax2.barh(tg["Party"][::-1],tg["Seats"][::-1],color=A2)
            ax2.set_xlabel("Seats"); plt.tight_layout(); st.pyplot(fig2); plt.close()
            st.dataframe(tg, use_container_width=True, hide_index=True)

    if not res.empty and not gen.empty:
        both=set(res[wc].unique())&set(gen[wc].unique())
        cmp=[{"Party":p,"Reserved":(res[wc]==p).sum(),"General":(gen[wc]==p).sum()} for p in sorted(both)]
        cdf=pd.DataFrame(cmp).sort_values("General",ascending=False).head(12)
        fig3,ax3=plt.subplots(figsize=(10,4))
        x=np.arange(len(cdf))
        ax3.bar(x-0.2,cdf["Reserved"],0.4,label="Reserved",color=A1,alpha=0.9)
        ax3.bar(x+0.2,cdf["General"],0.4,label="General",color=A2,alpha=0.9)
        ax3.set_xticks(x); ax3.set_xticklabels(cdf["Party"],rotation=30,ha='right')
        ax3.legend(); ax3.set_title("Reserved vs General Seat Wins")
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ─────────────────────────────────────────────
# 8b. CUSTOM METRICS PAGE
# ─────────────────────────────────────────────

def page_custom_metrics(df_edited, df_f):
    st.markdown('<div class="section-title">🛠️ Custom Metric Builder</div>', unsafe_allow_html=True)
    st.markdown(
        "Create new calculated columns using plain English. "
        "They will appear in **all analysis tabs** for the current session."
    )

    # ── Active metrics panel ──
    if st.session_state.custom_metrics:
        st.markdown('<div class="section-title">✅ Active Custom Metrics</div>', unsafe_allow_html=True)
        for mname, mcode in list(st.session_state.custom_metrics.items()):
            with st.expander(f"📊 **{mname}**", expanded=False):
                st.code(mcode, language="python")
                # Preview on current data
                try:
                    preview_df = df_f.head(10).copy()
                    exec(mcode, {"df": preview_df, "pd": pd, "np": np, "smart_lookup": smart_col, "smart_get": smart_get})
                    if mname in preview_df.columns:
                        col1, col2 = st.columns([2,1])
                        with col1:
                            st.dataframe(
                                preview_df[["Year", smart_col(preview_df,"Constituency Name"), mname]].head(8)
                                if smart_col(preview_df,"Constituency Name") in preview_df.columns
                                else preview_df[[mname]].head(8),
                                use_container_width=True, hide_index=True
                            )
                        with col2:
                            vals = pd.to_numeric(preview_df[mname], errors="coerce").dropna()
                            if not vals.empty:
                                st.metric("Mean", f"{vals.mean():.3f}")
                                st.metric("Min",  f"{vals.min():.3f}")
                                st.metric("Max",  f"{vals.max():.3f}")
                except Exception as e:
                    st.warning(f"Preview error: {e}")
                if st.button(f"🗑️ Remove {mname}", key=f"rm_{mname}"):
                    del st.session_state.custom_metrics[mname]
                    st.rerun()
        st.divider()
    else:
        st.info("No custom metrics yet. Build one below.")

    # ── Builder ──
    st.markdown('<div class="section-title">➕ Build a New Metric</div>', unsafe_allow_html=True)

    st.markdown("**Available columns:**")
    st.code(", ".join(df_f.columns.tolist()), language="text")

    c1, c2 = st.columns([1, 2])
    with c1:
        nm = st.text_input("Column Name", placeholder="e.g. Win_Margin_Pct", key="cm_name")
    with c2:
        nd = st.text_area(
            "Describe the logic in plain English",
            placeholder="e.g. Win Vote minus Run vote, divided by Votes Polled, multiplied by 100",
            key="cm_desc", height=80
        )

    if st.button("🧠 Draft with AI", key="cm_draft", disabled=(not api_key)):
        if nm and nd:
            with st.spinner("AI is translating your logic to code..."):
                code = gen_metric_code(df_f.head(), nm, nd)
                st.session_state.draft_code = code
                st.session_state.draft_name = nm
                st.rerun()
        else:
            st.warning("Please fill in both the column name and logic description.")

    if not api_key:
        st.caption("⚠️ No Gemini API key — AI drafting disabled. You can write code manually below.")
        with st.expander("✏️ Write code manually"):
            manual_code = st.text_area(
                "Python snippet (use `df` for the dataframe)",
                placeholder=f"df['{nm or 'my_metric'}'] = pd.to_numeric(df['Win Vote'], errors='coerce') / pd.to_numeric(df['Votes Polled'], errors='coerce') * 100",
                height=100, key="cm_manual"
            )
            manual_name = st.text_input("Metric name", value=nm or "", key="cm_manual_name")
            if st.button("Test & Save Manual Code", key="cm_manual_save"):
                try:
                    test_df = df_f.head(50).copy()
                    exec(manual_code, {"df": test_df, "pd": pd, "np": np, "smart_lookup": smart_col, "smart_get": smart_get})
                    st.success("✅ Code ran successfully!")
                    st.session_state.custom_metrics[manual_name] = manual_code
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Draft preview & save ──
    if "draft_code" in st.session_state:
        st.markdown('<div class="section-title">👀 Review Generated Code</div>', unsafe_allow_html=True)
        edited_code = st.text_area(
            "You can edit the code before saving:",
            value=st.session_state.draft_code,
            height=120, key="cm_edit"
        )
        st.session_state.draft_code = edited_code  # keep in sync

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🧪 Test on sample data", key="cm_test"):
                try:
                    test_df = df_f.head(50).copy()
                    exec(edited_code, {"df": test_df, "pd": pd, "np": np, "smart_lookup": smart_col, "smart_get": smart_get})
                    draft_name = st.session_state.get("draft_name","metric")
                    if draft_name in test_df.columns:
                        st.success(f"✅ Column **{draft_name}** created successfully!")
                        st.dataframe(test_df[[draft_name]].head(10), use_container_width=True, hide_index=True)
                    else:
                        st.warning("Code ran but column was not created. Check the column name in your code.")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col2:
            if st.button("💾 Save to all tabs", key="cm_save", type="primary"):
                try:
                    test_df = df_f.head(20).copy()
                    exec(edited_code, {"df": test_df, "pd": pd, "np": np, "smart_lookup": smart_col, "smart_get": smart_get})
                    draft_name = st.session_state.get("draft_name","metric")
                    st.session_state.custom_metrics[draft_name] = edited_code
                    del st.session_state.draft_code
                    if "draft_name" in st.session_state: del st.session_state.draft_name
                    st.success(f"✅ **{draft_name}** saved! It will now appear in all analysis tabs.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cannot save — fix errors first: {e}")

def page_ai(df):
    st.markdown('<div class="section-title">🤖 AI Election Analyst</div>', unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div style="background:#142236;border:1px solid #2a4060;border-radius:12px;padding:1.5rem;text-align:center;">
          <div style="font-size:2rem;">🔑</div>
          <div style="font-family:'Playfair Display',serif;color:#c9a84c;margin:0.4rem 0;">API Key Required</div>
          <div style="color:#8fa3c0;font-size:0.85rem;">Enter your Gemini API key in the sidebar to enable AI analysis.</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── Static dataset context (built once, reused every call) ──────────
    wc = smart_col(df,"Win Party"); tv = smart_col(df,"Votes Polled")
    el = smart_col(df,"Electors");  mg = smart_col(df,"Margin")
    cc = smart_col(df,"Constituency Name")
    years = sorted(df['Year'].unique())
    top5 = dict(df[wc].value_counts().head(5)) if wc in df.columns else {}
    yr_v  = {str(k): f"{v/1e6:.1f}M" for k,v in
             df.groupby('Year')[tv].apply(lambda x: n(x).sum()).items()} if tv in df.columns else {}
    DATA_CONTEXT = (
        f"Kerala Assembly Elections {fmt_year(min(years))}–{fmt_year(max(years))}. "
        f"Cols:{list(df.columns)}. "
        f"Top parties:{top5}. "
        f"Votes/yr:{yr_v}. "
        + (f"Margin mean/min/max:{n(df[mg]).mean():.0f}/{n(df[mg]).min():.0f}/{n(df[mg]).max():.0f}." if mg in df.columns else "")
    )

    # ── Session state init ───────────────────────────────────────────────
    for key, val in [("ai_messages",[]), ("ai_plots",[]), ("ai_codes",[]),
                     ("ai_conv_summary",""), ("ai_pending",None)]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Suggested questions (shown only at start) ────────────────────────
    suggestions = [
        "Which party has the highest strike rate?",
        "How has LDF vs UDF dominance shifted over decades?",
        "Plot turnout trend across all elections",
        "Which constituencies never changed winning party?",
        "Show CPM's vote share trend as a chart",
        "Which election had the most razor-thin margins?",
    ]

    if not st.session_state.ai_messages:
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0f1e30,#142236);border:1px solid #2a4060;'
            'border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1.2rem;">'
            '<div style="font-family:\'Playfair Display\',serif;color:#c9a84c;font-size:1rem;margin-bottom:0.6rem;">💡 Try asking…</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">'
            + "".join(
                f'<span style="background:#1a3050;border:1px solid #2a4060;border-radius:20px;'
                f'padding:0.3rem 0.8rem;font-size:0.78rem;color:#c8d8e8;">{q}</span>'
                for q in suggestions)
            + '</div></div>',
            unsafe_allow_html=True)

    # ── Render existing conversation ─────────────────────────────────────
    assistant_idx = 0  # tracks which assistant turn we're on for plot/code lookup
    for i, msg in enumerate(st.session_state.ai_messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end;margin:0.6rem 0;">'
                f'<div style="background:#1a3050;border:1px solid #2a4060;border-radius:12px 12px 2px 12px;'
                f'padding:0.6rem 1rem;max-width:78%;color:#e8e4da;font-size:0.88rem;">{msg["content"]}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div style="display:flex;align-items:flex-start;gap:0.6rem;margin:0.6rem 0;">'
                f'<div style="background:#c9a84c;border-radius:50%;width:28px;height:28px;'
                f'display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:0.9rem;">🗳️</div>'
                f'<div style="background:#0f1e30;border:1px solid #1e3250;border-radius:2px 12px 12px 12px;'
                f'padding:0.7rem 1rem;max-width:84%;color:#e8e4da;font-size:0.88rem;line-height:1.6;">'
                f'{msg["content"]}</div></div>',
                unsafe_allow_html=True)
            # Plot for this assistant turn
            if assistant_idx < len(st.session_state.ai_plots) and st.session_state.ai_plots[assistant_idx]:
                pb = st.session_state.ai_plots[assistant_idx]
                st.image(pb, use_column_width=True)
                st.download_button("📥 HD chart", pb, f"ai_chart_{assistant_idx}.png",
                                   "image/png", key=f"ai_dl_{assistant_idx}")
            # Code peek
            if assistant_idx < len(st.session_state.ai_codes) and st.session_state.ai_codes[assistant_idx]:
                code_html = st.session_state.ai_codes[assistant_idx].replace("<","&lt;").replace(">","&gt;")
                st.markdown(
                    f'<details style="margin-top:0.15rem;">'
                    f'<summary style="font-size:0.67rem;color:#2a4060;cursor:pointer;list-style:none;opacity:0.45;">⟨ computation ⟩</summary>'
                    f'<pre style="background:#060d16;color:#3a5a78;font-size:0.69rem;padding:0.5rem;'
                    f'border-radius:6px;overflow-x:auto;margin-top:0.25rem;">{code_html}</pre></details>',
                    unsafe_allow_html=True)
            assistant_idx += 1

    # ── If a response is pending (set before rerun), process it now ──────
    # This runs AFTER the history is rendered so the thinking animation
    # appears at the bottom, below prior messages.
    if st.session_state.ai_pending:
        prompt = st.session_state.ai_pending
        st.session_state.ai_pending = None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')

        plot_bytes  = None
        result_value = None
        code_used   = ""

        # ── CHART TYPE CATALOGUE ─────────────────────────────────────────
        CHART_SPECS = {
            "line":    "fig,ax=plt.subplots(); ax.plot(x,y,marker='o',color='#c9a84c'); ax.fill_between(x,y,alpha=0.15,color='#c9a84c')",
            "bar":     "fig,ax=plt.subplots(); bars=ax.barh(cats,vals,color=colors); ax.bar_label(bars,fmt='%.0f',padding=3,color='#e8e4da')",
            "stacked": "fig,ax=plt.subplots(); bottom=np.zeros(n); [ax.bar(x,v,bottom=b,label=l) for v,b,l in zip(vals,bottoms,labels)]",
            "pie":     "fig,ax=plt.subplots(); ax.pie(sizes,labels=lbls,autopct='%1.1f%%',wedgeprops={'edgecolor':'#0b1120','linewidth':1.5},colors=PAL)",
            "heatmap": "fig,ax=plt.subplots(); sns.heatmap(pivot,cmap='YlOrBr',annot=True,fmt='.1f',linewidths=0.3,linecolor='#1e3250',ax=ax)",
            "box":     "fig,ax=plt.subplots(); sns.boxplot(data=sub,x=cat_col,y=num_col,palette=dict(zip(cats,PAL)),ax=ax)",
            "scatter": "fig,ax=plt.subplots(); ax.scatter(x,y,c=colors,alpha=0.7,s=60,edgecolors='none')",
            "area":    "fig,ax=plt.subplots(); ax.fill_between(x,y,alpha=0.25,color='#c9a84c'); ax.plot(x,y,color='#c9a84c',linewidth=2)",
        }
        # Broad keyword list — catches "show me", "where is", "give me", follow-ups, etc.
        CHART_WORDS = {
            "plot","chart","graph","show","visual","trend","compare","heatmap",
            "distribution","breakdown","over time","by year","across","draw","display",
            "give me","where is","the plot","the chart","the graph","see it","see the",
            "map it","map the","illustrate","depict","render",
        }

        # ── Thinking animation ────────────────────────────────────────────
        with st.status("🧠 Thinking…", expanded=True) as thinking:

            conv_summary = st.session_state.ai_conv_summary
            pl = prompt.lower()

            # ── Step 1: Detect chart intent — keywords OR conversation context ──
            thinking.update(label="🔍 Choosing best visualisation…")

            # Check current prompt AND last assistant message for chart references
            last_ai = next((m["content"] for m in reversed(st.session_state.ai_messages)
                            if m["role"]=="assistant"), "")
            chart_in_context = any(w in last_ai.lower() for w in ["chart","plot","graph","figure"])

            is_chart_q = (
                any(w in pl for w in CHART_WORDS) or
                any(pl.startswith(w) for w in ["show","plot","draw","give","where","see"]) or
                (chart_in_context and any(w in pl for w in
                    ["it","the","that","this","one","same","again","also","too"]))
            )

            chosen_chart = None
            if is_chart_q:
                chart_pick_prompt = (
                    f"Q:{prompt}\n"
                    + (f"Context:{conv_summary}\n" if conv_summary else "")
                    + f"Cols:{list(df.columns)}\n"
                    f"Best chart type? Options:line,bar,stacked,pie,heatmap,box,scatter,area\n"
                    f"line=time trends;bar=category compare;stacked=part-of-whole;pie=≤8 share slices;"
                    f"heatmap=2d matrix;box=distribution;scatter=correlation;area=cumulative.\n"
                    f"One word only."
                )
                try:
                    pick_model = genai.GenerativeModel(
                        'gemini-2.0-flash',
                        generation_config={"temperature":0,"max_output_tokens":5,"candidate_count":1}
                    )
                    chosen_chart = pick_model.generate_content(chart_pick_prompt).text.strip().lower().split()[0]
                    if chosen_chart not in CHART_SPECS:
                        chosen_chart = "line"
                    thinking.update(label=f"📊 Chart type: {chosen_chart}")
                except Exception:
                    chosen_chart = "line"

            # ── Step 2: Generate computation + chart code ─────────────────
            thinking.update(label="⚙️ Computing answer from data…")

            # Build a fully self-contained code prompt.
            # When a chart IS required, the prompt is structured as chart-first
            # so the model can't skip the fig creation.
            if chosen_chart:
                code_prompt = (
                    f"Kerala election df. {DATA_CONTEXT}\nCols:{list(df.columns)}\n"
                    + (f"Prior context:{conv_summary}\n" if conv_summary else "")
                    + f"Task:{prompt}\n\n"
                    f"YOU MUST create a matplotlib figure. This is mandatory — DO NOT skip it.\n"
                    f"Chart type REQUIRED: {chosen_chart}\n"
                    f"Boilerplate to start with:\n{CHART_SPECS[chosen_chart]}\n\n"
                    f"Steps:\n"
                    f"1. Compute the data needed (use smart_get(df,'Col') for Series access).\n"
                    f"2. Create fig,ax using plt.subplots(figsize=(10,4)).\n"
                    f"3. Apply dark theme: fig.patch.set_facecolor('#0b1120'); ax.set_facecolor('#0f1e30'); "
                    f"ax.tick_params(colors='#8fa3c0'); ax.title.set_color('#c9a84c').\n"
                    f"4. Draw the {chosen_chart} chart on ax.\n"
                    f"5. Set result=<one-line string summary of key finding>.\n"
                    f"DO NOT print. DO NOT use st.*. Return ONLY Python, no markdown."
                )
            else:
                code_prompt = (
                    f"Kerala election df. {DATA_CONTEXT}\nCols:{list(df.columns)}\n"
                    + (f"Prior context:{conv_summary}\n" if conv_summary else "")
                    + f"Task:{prompt}\n"
                    f"Rules: smart_get(df,'Col') returns Series. Store answer in result (str≤200).\n"
                    f"No fig needed. No print/st. Return ONLY Python."
                )

            # ── plt proxy: captures ANY figure the model creates, regardless
            #    of variable name. Wraps plt.subplots / plt.figure so the
            #    returned Figure is always stored in _captured["fig"].
            class _PltProxy:
                """Transparent proxy around matplotlib.pyplot.
                Intercepts subplots() and figure() to capture the created Figure."""
                def __init__(self, real_plt, store):
                    self._plt = real_plt
                    self._store = store
                def subplots(self, *a, **kw):
                    fig, ax = self._plt.subplots(*a, **kw)
                    self._store["fig"] = fig
                    return fig, ax
                def figure(self, *a, **kw):
                    fig = self._plt.figure(*a, **kw)
                    self._store["fig"] = fig
                    return fig
                def __getattr__(self, name):
                    return getattr(self._plt, name)

            def _make_env():
                _cap = {"fig": None}
                _plt_proxy = _PltProxy(plt, _cap)
                env = {
                    "df":df, "plt":_plt_proxy, "sns":sns, "pd":pd, "np":np,
                    "smart_lookup":smart_col, "smart_get":smart_get,
                    "fig":None, "result":None,
                    "_cap":_cap,
                    "GOLD":"#c9a84c","A1":"#e05c4b","A2":"#4b9ce8","A3":"#6bcb77",
                    "MUTED":"#8fa3c0","TEXT_MAIN":"#e8e4da","DARK_BG":"#0b1120","CARD_BG":"#0f1e30",
                    "PAL":["#c9a84c","#e05c4b","#4b9ce8","#6bcb77","#b07aff","#ff9f7a","#7af0d8"],
                    "BLOC_COLORS":{"LDF":"#e05c4b","UDF":"#4b9ce8","NDA":"#f0a500","Other":"#888"},
                }
                return env, _cap

            def _get_fig(g, cap):
                """Return figure from exec env: explicit g['fig'] first,
                then proxy-captured fig, then any Figure in globals."""
                import matplotlib.figure as _mf
                if isinstance(g.get("fig"), _mf.Figure): return g["fig"]
                if isinstance(cap.get("fig"), _mf.Figure): return cap["fig"]
                # last resort: scan all globals for a Figure object
                for v in g.values():
                    if isinstance(v, _mf.Figure): return v
                # also check plt's open figures
                figs = [plt.figure(n) for n in plt.get_fignums()]
                return figs[-1] if figs else None

            for attempt in range(4):
                try:
                    raw = model.generate_content(code_prompt).text
                    code_used = raw.replace("```python","").replace("```","").strip()
                    g, cap = _make_env()
                    exec(code_used, g)
                    result_value = g.get("result")

                    captured_fig = _get_fig(g, cap)
                    if captured_fig is not None:
                        thinking.update(label=f"🎨 Rendering {chosen_chart or 'chart'}…")
                        # Apply dark theme in case model forgot
                        captured_fig.patch.set_facecolor('#0b1120')
                        for ax_ in captured_fig.get_axes():
                            ax_.set_facecolor('#0f1e30')
                            ax_.tick_params(colors='#8fa3c0')
                            ax_.xaxis.label.set_color('#8fa3c0')
                            ax_.yaxis.label.set_color('#8fa3c0')
                        plt.figure(captured_fig.number)
                        plt.tight_layout()
                        plot_bytes = save_fig_hd(captured_fig)
                        store_plot(captured_fig, f"ai_{assistant_idx}")
                        plt.close(captured_fig)
                        # close any other open figures from this exec
                        for n in plt.get_fignums(): plt.close(plt.figure(n))
                        break

                    elif chosen_chart:
                        code_prompt += (
                            f"\nAttempt {attempt+1} produced no figure."
                            f" Call plt.subplots() and assign result to fig,ax."
                            f" Draw the {chosen_chart} chart. This is mandatory."
                        )
                    else:
                        break  # text-only question, no fig needed

                except Exception as e:
                    code_prompt += f"\nERR {attempt+1}:{str(e)[:120]} Fix."
                    # close any leaked figures
                    for n in plt.get_fignums(): plt.close(plt.figure(n))

            # ── Step 3: Narrate ───────────────────────────────────────────
            thinking.update(label="✍️ Writing response…")
            narrate_prompt = (
                f"Kerala election expert. 2-4 prose sentences. Specific numbers. No bullets/markdown.\n"
                + (f"Conv context:{conv_summary}\n" if conv_summary else "")
                + f"Q:{prompt}\nResult:{result_value or '(see chart)'}\n"
                + (f"A {chosen_chart} chart was generated. Do NOT describe what it would show — it IS shown." if plot_bytes
                   else "No chart was generated — answer fully in text.")
            )
            narration = model.generate_content(narrate_prompt).text.strip()

            # ── Step 4: Compress conversation summary ─────────────────────
            thinking.update(label="📝 Updating conversation memory…")
            prev = st.session_state.ai_conv_summary
            compress_prompt = (
                f"Summarise in ≤80 words. Keep key facts+numbers. No filler.\n"
                f"Prev:{prev}\n"
                f"User:{prompt} | Answer:{narration[:300]}"
            )
            try:
                st.session_state.ai_conv_summary = model.generate_content(compress_prompt).text.strip()
            except Exception:
                pass

            thinking.update(label="✅ Done", state="complete", expanded=False)

        # ── Persist results ───────────────────────────────────────────────
        st.session_state.ai_messages.append({"role":"assistant","content":narration})
        st.session_state.ai_plots.append(plot_bytes)
        st.session_state.ai_codes.append(code_used)
        st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask anything about Kerala election data…")
    if prompt:
        st.session_state.ai_messages.append({"role":"user","content":prompt})
        # Set pending so thinking animation renders after history
        st.session_state.ai_pending = prompt
        st.rerun()

    # ── Footer: conversation memory peek + clear ─────────────────────────
    if st.session_state.ai_messages:
        col_sum, col_clr = st.columns([5,1])
        with col_sum:
            if st.session_state.ai_conv_summary:
                with st.expander("🧠 Conversation memory", expanded=False):
                    st.caption(st.session_state.ai_conv_summary)
        with col_clr:
            if st.button("🗑️ Clear", key="ai_clear"):
                for k in ["ai_messages","ai_plots","ai_codes","ai_pending"]:
                    st.session_state[k] = [] if k != "ai_pending" else None
                st.session_state.ai_conv_summary = ""
                st.rerun()

# 9. MAIN APP
# ─────────────────────────────────────────────
st.markdown("""
<div class="atlas-header">
  <div class="atlas-title">🗳️ Kerala Election Atlas</div>
  <div class="atlas-subtitle">Assembly Elections 1957 – 2021 · Constituency-Level Intelligence</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    if not api_key:
        api_key = st.text_input("Gemini API Key", type="password")
    # Sign out — clears the auth token from URL
    if st.secrets.get("APP_PASSWORD"):
        if st.button("🔒 Sign Out", key="_signout"):
            st.query_params.clear()
            st.rerun()
    st.divider()
    uploaded_files = st.file_uploader("Upload Excel/CSV", accept_multiple_files=True, type=['xlsx','xls','csv'])

if not uploaded_files:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;opacity:0.6;">
      <div style="font-size:3rem;">🗺️</div>
      <div style="font-family:'Playfair Display',serif;font-size:1.4rem;color:#c9a84c;margin-top:0.5rem;">Upload election data to begin</div>
      <div style="font-size:0.85rem;color:#8fa3c0;margin-top:0.4rem;">Supports .xlsx (multi-sheet per year) or .csv</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading..."):
    master_df = load_data(uploaded_files)

if master_df is None: st.error("Could not load data."); st.stop()
st.toast(f"✅ {len(master_df):,} records across {master_df['Year'].nunique()} elections", icon="🗳️")

# ── Session state init ──────────────────────
if "custom_metrics" not in st.session_state: st.session_state.custom_metrics = {}

# ── Filters (sidebar) ───────────────────────
st.sidebar.divider()
st.sidebar.markdown("### 🔍 Filters")
years_avail=sorted(master_df['Year'].unique())
sel_years=st.sidebar.select_slider("Election Years", options=years_avail, value=(years_avail[0],years_avail[-1]))
df_f=master_df[(master_df['Year']>=sel_years[0])&(master_df['Year']<=sel_years[1])].copy()

if 'Bloc' in df_f.columns:
    ba=sorted(df_f['Bloc'].dropna().unique())
    sb=st.sidebar.multiselect("Blocs",ba,default=ba)
    if sb: df_f=df_f[df_f['Bloc'].isin(sb)]

cat_c=smart_col(df_f,"Category")
if cat_c in df_f.columns:
    cats=sorted(df_f[cat_c].dropna().unique())
    sc=st.sidebar.multiselect("Constituency Type",cats,default=cats)
    if sc: df_f=df_f[df_f[cat_c].isin(sc)]

st.sidebar.caption(f"**{len(df_f):,}** rows · {df_f['Year'].nunique()} elections")

# ── HD Batch Export sidebar ──────────────────
st.sidebar.divider()
st.sidebar.markdown("### 📥 Export All Charts")
if st.session_state.get("plot_store"):
    import io, zipfile
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in st.session_state.plot_store.items():
            zf.writestr(f"{name}.png", data)
    zip_buf.seek(0)
    st.sidebar.download_button(
        f"⬇️ Download {len(st.session_state.plot_store)} charts (ZIP)",
        zip_buf.getvalue(), "kerala_election_charts.zip", "application/zip", key="batch_dl"
    )
    if st.sidebar.button("🗑️ Clear chart store", key="clear_plots"):
        st.session_state.plot_store = {}; st.rerun()
else:
    st.sidebar.caption("Charts will appear here after browsing pages.")

# ── Apply saved custom metrics to filtered data ──
# Runs AFTER filtering so new columns appear in df_edited
for _cm_name, _cm_code in st.session_state.custom_metrics.items():
    try:
        exec(_cm_code, {"df": df_f, "pd": pd, "np": np, "smart_lookup": smart_col, "smart_get": smart_get})
    except Exception as _e:
        st.warning(f"Custom metric '{_cm_name}' error: {_e}")

df_edited=df_f.copy()
with st.expander("📝 View & Edit Raw Data", expanded=False):
    df_edited=st.data_editor(df_f,num_rows="dynamic",use_container_width=True,key="ed")
    st.download_button("📥 Download CSV",df_edited.to_csv(index=False).encode(),"election_data.csv","text/csv")

# ── NAVIGATION ───────────────────────────────
NAV=[("🏠","Overview"),("📊","Dashboard"),("🎯","Party Analysis"),("👨‍👩‍👧","Party Families"),("🏛️","Blocs"),
     ("📐","Statistics"),("⚔️","Swing Analyzer"),("📍","Constituency"),
     ("🗺️","Regional"),("🏷️","Reserved Seats"),("🛠️","Custom Metrics"),("🤖","AI Analyst")]

if "tab" not in st.session_state: st.session_state.tab="Overview"

cols=st.columns(len(NAV))
for col,(icon,name) in zip(cols,NAV):
    with col:
        label=f"{icon} {name}"
        if st.button(label,key=f"nav_{name}",use_container_width=True):
            st.session_state.tab=name; st.rerun()

# Active indicator
st.markdown(f"""
<div style="display:flex;gap:0.3rem;margin:-0.5rem 0 0.8rem;padding:0 0.1rem;">
{''.join(f'<div style="height:3px;flex:1;background:{"#c9a84c" if n==st.session_state.tab else "#1e3250"};border-radius:2px;"></div>' for _,n in NAV)}
</div>
""", unsafe_allow_html=True)

st.divider()
page=st.session_state.tab

if page=="Overview":        page_overview(df_edited)
elif page=="Dashboard":      page_dashboard(df_edited)
elif page=="Party Analysis": page_parties(df_edited)
elif page=="Party Families": page_families(df_edited)
elif page=="Blocs":          page_blocs(df_edited)
elif page=="Statistics":     page_stats(df_edited)
elif page=="Swing Analyzer": page_swing(df_edited)
elif page=="Constituency":   page_constituency(df_edited)
elif page=="Regional":       page_regional(df_edited)
elif page=="Reserved Seats": page_reserved(df_edited)
elif page=="Custom Metrics": page_custom_metrics(df_edited, df_f)
elif page=="AI Analyst":    page_ai(df_edited)