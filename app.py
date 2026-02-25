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
from scipy import stats

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Kerala Election Trends & AI Analyst", layout="wide")

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.warning("API Key not found in secrets. Please enter it below:")
        api_key = st.text_input("Gemini API Key", type="password")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    st.info("Upload Excel (.xlsx) files with multiple sheets or individual CSVs.")

# --- KERALA-SPECIFIC PARTY DEFINITIONS ---
INDIVIDUAL_PARTIES = ["CPI", "CPM", "INC", "ML", "KCM", "KCJ"]

PARTY_FAMILIES = {
    "Kerala Congress Family": ["KC", "KCM", "KCJ", "KCJB", "KCB", "KCS", "KCST", "KJS", "KCAMG", "KCD"],
    "Muslim League Family":   ["ML", "AIML", "INL"],
    "Congress Family":        ["INC", "INCO", "INCA", "CS"],
    "Socialist Family":       ["PSP", "RSP", "SSP", "ISP", "KTP", "CS", "KSP", "LKD", "LJD",
                               "BLD", "DSP", "ICS", "JDU", "NCP", "RSPB"],
    "CPM Breakaway":          ["CMP", "JSS", "RMP"],
}

BLOCS = {
    "LDF": ["CPM", "CPI", "NCP", "JDU", "RSP", "KTP", "CMP", "JSS", "RMP"],
    "UDF": ["INC", "ML", "AIML", "INL", "KC", "KCM", "KCJ", "KCJB", "KCB", "KCS",
            "KCST", "KJS", "KCAMG", "KCD", "INCO", "INCA", "CS"],
    "NDA": ["BJP", "BDJS", "KCP", "BDP"],
}

# --- 2. HELPER FUNCTIONS ---
def smart_column_lookup(df, guessed_name):
    if guessed_name in df.columns:
        return guessed_name
    for col in df.columns:
        if col.lower() == guessed_name.lower():
            return col
    matches = difflib.get_close_matches(guessed_name, df.columns, n=1, cutoff=0.5)
    return matches[0] if matches else guessed_name

def assign_family(party, families=PARTY_FAMILIES):
    for family, members in families.items():
        if party in members:
            return family
    return "Other"

def assign_bloc(party, blocs=BLOCS):
    for bloc, members in blocs.items():
        if party in members:
            return bloc
    return "Other"

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

@st.cache_data
def load_and_combine_data(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            file_ext = file.name.split('.')[-1].lower()
            def process_df(df_temp, source_name):
                if 'Cons No.' in df_temp.columns:
                    df_temp['Cons No.'] = df_temp['Cons No.'].astype(str)
                if 'Year' not in df_temp.columns:
                    try:
                        year_match = re.search(r'\d{4}', str(source_name))
                        df_temp['Year'] = int(year_match.group(0)) if year_match else source_name
                    except:
                        df_temp['Year'] = source_name
                return df_temp

            if file_ext in ['xlsx', 'xls']:
                xls_dict = pd.read_excel(file, sheet_name=None)
                for sheet_name, df in xls_dict.items():
                    df = process_df(df, sheet_name)
                    all_dfs.append(df)
            elif file_ext == 'csv':
                df = pd.read_csv(file)
                df = process_df(df, file.name)
                all_dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Auto-assign family and bloc columns
        party_col = smart_column_lookup(combined, "Party")
        win_party_col = smart_column_lookup(combined, "Win Party")
        ref_col = win_party_col if win_party_col in combined.columns else (party_col if party_col in combined.columns else None)
        if ref_col:
            combined["Party Family"] = combined[ref_col].astype(str).apply(assign_family)
            combined["Bloc"] = combined[ref_col].astype(str).apply(assign_bloc)
        return combined
    return None

def generate_custom_metric_code(df, metric_name, description):
    genai.configure(api_key=api_key)
    columns_list = list(df.columns)
    prompt = f"""
    You are a Python Pandas expert.
    Task: Write a Python snippet to create a new column named '{metric_name}' in the dataframe `df`.
    Existing Columns: {columns_list}
    User Description: "{description}"
    RULES:
    1. Use `smart_lookup(df, 'column_name')` for ALL column references.
    2. Ensure columns are numeric before math. Use `pd.to_numeric(..., errors='coerce')`.
    3. Return ONLY the code snippet. No markdown, no comments.
    EXAMPLE: df['{metric_name}'] = pd.to_numeric(df[smart_lookup(df, 'Win Vote')], errors='coerce') / pd.to_numeric(df[smart_lookup(df, 'Electors')], errors='coerce')
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(prompt)
    return response.text.strip()

def query_gemini_smart(query, dataframe):
    if not api_key:
        return None, None, "⚠️ Please enter your API Key in the sidebar."
    genai.configure(api_key=api_key)
    try:
        data_sample = dataframe.sample(n=5).to_markdown()
    except:
        data_sample = dataframe.head(5).to_markdown()
    columns_list = list(dataframe.columns)
    system_instruction = """
    You are an expert Kerala Election Analyst Python Agent.
    1. You are given a pandas DataFrame `df`.
    2. ALWAYS use `smart_lookup(df, 'col_name')` to access columns.
    3. If the user asks for a Plot/Graph: Create a matplotlib figure named `fig`.
    4. If the user asks for Text/Insight: Calculate the answer and store it in a string variable named `answer`.
    5. You can do BOTH.
    6. Return ONLY valid Python code. No markdown, no comments outside the code.
    """
    prompt = f"""
    {system_instruction}
    Columns: {columns_list}
    Sample Data:
    {data_sample}
    USER REQUEST: {query}
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            code = response.text.replace("```python", "").replace("```", "").strip()
            exec_globals = {
                "df": dataframe, "plt": plt, "sns": sns, "pd": pd, "np": np,
                "smart_lookup": smart_column_lookup, "fig": None, "answer": None
            }
            exec(code, exec_globals)
            return exec_globals.get('fig'), exec_globals.get('answer'), code
        except Exception as e:
            prompt += f"\n\nPrevious code failed with error: {e}. Fix it."
    return None, None, "Failed to generate analysis."

# ============================================================
# STATISTICAL INDEX FUNCTIONS
# ============================================================

def calc_gallagher_index(votes_pct, seats_pct):
    """Gallagher Index (Least Squares)"""
    diffs = np.array(votes_pct) - np.array(seats_pct)
    return np.sqrt(0.5 * np.sum(diffs**2))

def calc_loosemore_hanby(votes_pct, seats_pct):
    """Loosemore–Hanby Index"""
    return 0.5 * np.sum(np.abs(np.array(votes_pct) - np.array(seats_pct)))

def calc_seat_bonus(votes_pct, seats_pct):
    """Seat Bonus Index (largest party bonus)"""
    idx = np.argmax(votes_pct)
    return seats_pct[idx] - votes_pct[idx]

def calc_pedersen_index(vote_share_t1, vote_share_t2):
    """Pedersen Index (Total Electoral Volatility)"""
    return 0.5 * np.sum(np.abs(np.array(vote_share_t2) - np.array(vote_share_t1)))

def calc_enep(vote_shares):
    """Effective Number of Electoral Parties"""
    p = np.array(vote_shares) / 100.0
    p = p[p > 0]
    return 1.0 / np.sum(p**2)

def calc_enpp(seat_shares):
    """Effective Number of Parliamentary Parties"""
    p = np.array(seat_shares) / 100.0
    p = p[p > 0]
    return 1.0 / np.sum(p**2)

def calc_turnout_rate(votes_polled, electors):
    return (votes_polled / electors) * 100 if electors > 0 else np.nan

def calc_strike_rate(seats_won, seats_contested):
    return (seats_won / seats_contested) * 100 if seats_contested > 0 else np.nan

def calc_vote_efficiency(votes_won, total_votes):
    return (votes_won / total_votes) * 100 if total_votes > 0 else np.nan

def calc_wasted_votes(wasted, total):
    return (wasted / total) * 100 if total > 0 else np.nan

def calc_fractionalization(vote_shares):
    p = np.array(vote_shares) / 100.0
    return 1.0 - np.sum(p**2)

def calc_dalton_polarization(vote_shares, positions):
    """Dalton Polarization Index (requires party positions on a scale)"""
    p = np.array(vote_shares) / 100.0
    mean_pos = np.sum(p * np.array(positions))
    return np.sqrt(np.sum(p * (np.array(positions) - mean_pos)**2))

def calc_hhi(vote_shares):
    """Herfindahl–Hirschman Index"""
    return np.sum((np.array(vote_shares) / 100.0)**2)

def calc_entropy_index(vote_shares):
    p = np.array(vote_shares) / 100.0
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def build_transition_matrix(df_a, df_b, cons_col, party_col):
    """Build a seat transition matrix between two elections"""
    m = pd.merge(
        df_a[[cons_col, party_col]].rename(columns={party_col: 'A'}),
        df_b[[cons_col, party_col]].rename(columns={party_col: 'B'}),
        on=cons_col, how='inner'
    )
    return pd.crosstab(m['A'], m['B'])

def calc_alternation_absorption(wins_by_bloc):
    """Kerala-specific: how much alternation between LDF/UDF"""
    blocs = list(wins_by_bloc.keys())
    if len(blocs) < 2:
        return np.nan
    vals = np.array([wins_by_bloc[b] for b in blocs])
    return 1 - (np.max(vals) - np.min(vals)) / (np.sum(vals) if np.sum(vals) > 0 else 1)

def calc_alliance_dependency(party_votes_solo, party_votes_alliance):
    """Alliance Dependency Index"""
    if party_votes_solo + party_votes_alliance == 0:
        return np.nan
    return party_votes_alliance / (party_votes_solo + party_votes_alliance)

def calc_margin_category(margin, total_votes):
    """Classify margin into Brute / Comfortable / Narrow / Very Thin"""
    if total_votes == 0:
        return "Unknown"
    pct = (margin / total_votes) * 100
    if pct > 20:   return "Brute Majority"
    elif pct > 10: return "Comfortable"
    elif pct > 5:  return "Narrow"
    else:          return "Very Thin"

def get_win_pattern(party, df, cons_col, party_col, win_col):
    """Classify constituencies as stronghold / neighborhood / chance / hostile"""
    counts = df[df[win_col] == party][cons_col].value_counts()
    total_elections = df['Year'].nunique()
    patterns = {}
    for cons, wins in counts.items():
        rate = wins / total_elections
        if rate >= 0.75:   patterns[cons] = "Stronghold"
        elif rate >= 0.5:  patterns[cons] = "Neighbourhood"
        elif rate >= 0.25: patterns[cons] = "Chance"
        else:              patterns[cons] = "Hostile"
    return patterns

def compute_party_stats(df, party, year_col='Year', party_col=None,
                         win_col=None, cons_col=None, votes_col=None,
                         total_votes_col=None, electors_col=None,
                         margin_col=None, contested_col=None):
    """Compute all individual party statistics (Section C of PDF)"""
    if party_col is None: party_col = smart_column_lookup(df, "Party")
    if win_col is None:   win_col   = smart_column_lookup(df, "Win Party")
    if cons_col is None:  cons_col  = smart_column_lookup(df, "Constituency Name")
    if votes_col is None: votes_col = smart_column_lookup(df, "Win Vote")
    if total_votes_col is None: total_votes_col = smart_column_lookup(df, "Votes Polled")
    if electors_col is None: electors_col = smart_column_lookup(df, "Electors")
    if margin_col is None: margin_col = smart_column_lookup(df, "Margin")

    stats_out = {}
    for year, grp in df.groupby(year_col):
        total_seats = len(grp[cons_col].unique()) if cons_col in grp.columns else np.nan
        won = grp[grp[win_col] == party] if win_col in grp.columns else pd.DataFrame()
        seats_won = len(won)

        total_v  = safe_numeric(grp[total_votes_col]).sum() if total_votes_col in grp.columns else np.nan
        party_v  = safe_numeric(won[votes_col]).sum()       if (votes_col in grp.columns and not won.empty) else 0

        vote_pct  = (party_v / total_v * 100) if total_v > 0 else np.nan
        seat_pct  = (seats_won / total_seats * 100) if (total_seats and total_seats > 0) else np.nan

        margins = safe_numeric(won[margin_col]) if (margin_col in won.columns and not won.empty) else pd.Series(dtype=float)
        tv_won  = safe_numeric(won[total_votes_col]) if (total_votes_col in won.columns and not won.empty) else pd.Series(dtype=float)

        margin_cats = {}
        if not margins.empty and not tv_won.empty:
            for m, t in zip(margins, tv_won):
                cat = calc_margin_category(m, t)
                margin_cats[cat] = margin_cats.get(cat, 0) + 1

        stats_out[year] = {
            "seats_won": seats_won,
            "vote_pct": round(vote_pct, 2) if not np.isnan(vote_pct) else np.nan,
            "seat_pct": round(seat_pct, 2) if (seat_pct is not None and not np.isnan(seat_pct)) else np.nan,
            "avg_margin": round(margins.mean(), 0) if not margins.empty else np.nan,
            "margin_categories": margin_cats,
        }
    return stats_out

def compute_bloc_stats(df, year_col='Year', win_col=None, cons_col=None,
                        total_votes_col=None, votes_col=None):
    if win_col is None:          win_col          = smart_column_lookup(df, "Win Party")
    if cons_col is None:         cons_col         = smart_column_lookup(df, "Constituency Name")
    if total_votes_col is None:  total_votes_col  = smart_column_lookup(df, "Votes Polled")
    if votes_col is None:        votes_col        = smart_column_lookup(df, "Win Vote")

    results = {}
    for year, grp in df.groupby(year_col):
        total_seats = grp[cons_col].nunique() if cons_col in grp.columns else np.nan
        total_v     = safe_numeric(grp[total_votes_col]).sum() if total_votes_col in grp.columns else np.nan
        bloc_row    = {}
        for bloc, members in BLOCS.items():
            won      = grp[grp[win_col].isin(members)] if win_col in grp.columns else pd.DataFrame()
            s_won    = len(won)
            v_won    = safe_numeric(won[votes_col]).sum() if (votes_col in won.columns and not won.empty) else 0
            bloc_row[bloc] = {
                "seats_won":  s_won,
                "vote_pct":   round(v_won / total_v * 100, 2) if (total_v and total_v > 0) else np.nan,
                "seat_pct":   round(s_won / total_seats * 100, 2) if (total_seats and total_seats > 0) else np.nan,
            }
        results[year] = bloc_row
    return results

# ============================================================
# RENDER HELPERS
# ============================================================

def render_party_analysis(df_edited, party):
    st.markdown(f"#### 🔍 {party} — Detailed Analysis")
    stats = compute_party_stats(df_edited, party)
    if not stats:
        st.warning(f"No data found for {party}.")
        return

    rows = []
    for yr, s in sorted(stats.items()):
        row = {"Year": yr, "Seats Won": s["seats_won"],
               "Vote %": s["vote_pct"], "Seat %": s["seat_pct"],
               "Avg Margin": s["avg_margin"]}
        row.update({f"Margin – {k}": v for k, v in s["margin_categories"].items()})
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("Year")
    st.dataframe(tbl, use_container_width=True)

    # Trend chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tbl["Seats Won"].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title(f"{party} — Seats Won per Election")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Seats")

    vote_data = tbl["Vote %"].dropna()
    if not vote_data.empty:
        vote_data.plot(kind='line', marker='o', ax=axes[1], color='darkorange')
        axes[1].set_title(f"{party} — Vote % Trend")
        axes[1].set_xlabel("Year"); axes[1].set_ylabel("Vote %")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Win patterns
    win_col  = smart_column_lookup(df_edited, "Win Party")
    cons_col = smart_column_lookup(df_edited, "Constituency Name")
    votes_col= smart_column_lookup(df_edited, "Win Vote")
    if all(c in df_edited.columns for c in [win_col, cons_col]):
        patterns = get_win_pattern(party, df_edited, cons_col, None, win_col)
        if patterns:
            pat_df = pd.DataFrame(list(patterns.items()), columns=["Constituency", "Win Pattern"])
            with st.expander(f"Win Pattern Breakdown ({len(patterns)} constituencies)"):
                st.dataframe(pat_df.sort_values("Win Pattern"), use_container_width=True)

def render_bloc_analysis(df_edited):
    st.markdown("#### 🏛️ Bloc-Level Analysis (LDF / UDF / NDA)")
    bloc_stats = compute_bloc_stats(df_edited)
    if not bloc_stats:
        st.warning("No bloc data.")
        return

    rows = []
    for yr, blocs in sorted(bloc_stats.items()):
        for bloc, s in blocs.items():
            rows.append({"Year": yr, "Bloc": bloc,
                         "Seats Won": s["seats_won"],
                         "Vote %": s["vote_pct"],
                         "Seat %": s["seat_pct"]})
    bdf = pd.DataFrame(rows)
    st.dataframe(bdf, use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for bloc in BLOCS:
        sub = bdf[bdf["Bloc"] == bloc].sort_values("Year")
        axes[0].plot(sub["Year"], sub["Seats Won"], marker='o', label=bloc)
        axes[1].plot(sub["Year"], sub["Vote %"],   marker='o', label=bloc)
    axes[0].set_title("Seats Won by Bloc"); axes[0].legend()
    axes[1].set_title("Vote % by Bloc");    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def render_statistical_indices(df_edited):
    st.markdown("#### 📐 Statistical Indices")

    win_col   = smart_column_lookup(df_edited, "Win Party")
    cons_col  = smart_column_lookup(df_edited, "Constituency Name")
    votes_col = smart_column_lookup(df_edited, "Win Vote")
    tv_col    = smart_column_lookup(df_edited, "Votes Polled")
    el_col    = smart_column_lookup(df_edited, "Electors")
    margin_col= smart_column_lookup(df_edited, "Margin")
    year_col  = "Year"

    needed = [win_col, cons_col, tv_col]
    if not all(c in df_edited.columns for c in needed):
        st.warning("Required columns (Win Party, Constituency Name, Votes Polled) not found.")
        return

    all_years = sorted(df_edited[year_col].unique())
    summary_rows = []

    for year in all_years:
        grp = df_edited[df_edited[year_col] == year].copy()
        total_seats = grp[cons_col].nunique()
        total_votes = safe_numeric(grp[tv_col]).sum()

        # Per-party vote / seat shares
        party_wins  = grp[win_col].value_counts()
        party_votes = grp.groupby(win_col)[votes_col].apply(lambda x: safe_numeric(x).sum()) if votes_col in grp.columns else pd.Series()

        seat_pcts  = (party_wins  / total_seats * 100).values.tolist()
        vote_pcts  = (party_votes / total_votes * 100).values.tolist() if total_votes > 0 else []

        # Align lengths
        min_len = min(len(seat_pcts), len(vote_pcts))
        seat_pcts = seat_pcts[:min_len]
        vote_pcts = vote_pcts[:min_len]

        row = {"Year": year}

        # A. Disproportionality
        if min_len > 0:
            row["Gallagher Index"]      = round(calc_gallagher_index(vote_pcts, seat_pcts), 3)
            row["Loosemore-Hanby"]      = round(calc_loosemore_hanby(vote_pcts, seat_pcts), 3)
            row["Seat Bonus"]           = round(calc_seat_bonus(vote_pcts, seat_pcts), 3)

        # C. Party System
        if len(vote_pcts) > 0:
            row["ENEP"]  = round(calc_enep(vote_pcts), 3)
            row["ENPP"]  = round(calc_enpp(seat_pcts), 3)

        # D. Turnout
        if el_col in grp.columns:
            total_el = safe_numeric(grp[el_col]).sum()
            row["Turnout Rate %"] = round(calc_turnout_rate(total_votes, total_el), 2) if total_el > 0 else np.nan

        # F. Fragmentation
        if len(vote_pcts) > 0:
            row["Fractionalization"] = round(calc_fractionalization(vote_pcts), 3)
            row["Entropy Index"]     = round(calc_entropy_index(vote_pcts), 3)

        # I. Dominance
        if len(vote_pcts) > 0:
            row["HHI"] = round(calc_hhi(vote_pcts), 4)

        # Margin stats
        if margin_col in grp.columns and tv_col in grp.columns:
            margins  = safe_numeric(grp[margin_col])
            tv_s     = safe_numeric(grp[tv_col])
            valid    = margins.dropna()
            if not valid.empty:
                row["Avg Margin"]         = round(valid.mean(), 0)
                row["Close Contest (<5%)"]= int(((margins / tv_s) < 0.05).sum())
                row["Safety Index"]       = round((margins / tv_s * 100).mean(), 2) if not tv_s.empty else np.nan

        summary_rows.append(row)

    # Pedersen index (needs consecutive years)
    if len(all_years) >= 2:
        pedersen_vals = []
        for i in range(1, len(all_years)):
            y1, y2 = all_years[i-1], all_years[i]
            g1 = df_edited[df_edited[year_col] == y1]
            g2 = df_edited[df_edited[year_col] == y2]
            tv1 = safe_numeric(g1[tv_col]).sum()
            tv2 = safe_numeric(g2[tv_col]).sum()
            if votes_col in df_edited.columns and tv1 > 0 and tv2 > 0:
                all_parties = set(g1[win_col].unique()) | set(g2[win_col].unique())
                vs1 = {p: safe_numeric(g1[g1[win_col]==p][votes_col]).sum() / tv1 * 100 for p in all_parties}
                vs2 = {p: safe_numeric(g2[g2[win_col]==p][votes_col]).sum() / tv2 * 100 for p in all_parties}
                peds = calc_pedersen_index(
                    [vs1.get(p, 0) for p in all_parties],
                    [vs2.get(p, 0) for p in all_parties]
                )
                pedersen_vals.append({"Year": y2, "Pedersen Index": round(peds, 3)})
        if pedersen_vals:
            ped_df = pd.DataFrame(pedersen_vals).set_index("Year")
            st.markdown("**Electoral Volatility (Pedersen Index)**")
            st.dataframe(ped_df, use_container_width=True)

    if summary_rows:
        idx_df = pd.DataFrame(summary_rows).set_index("Year")
        st.dataframe(idx_df, use_container_width=True)

        # Visualize key indices
        plot_cols = [c for c in ["Gallagher Index", "ENEP", "ENPP", "Turnout Rate %", "Fractionalization"] if c in idx_df.columns]
        if plot_cols:
            fig, ax = plt.subplots(figsize=(12, 5))
            for col in plot_cols:
                ax.plot(idx_df.index, idx_df[col], marker='o', label=col)
            ax.set_title("Key Electoral Indices over Time")
            ax.legend(); ax.set_xlabel("Year")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

def render_swing_analyzer(df_edited):
    """Enhanced swing analyzer with transition matrix"""
    st.markdown("#### ⚔️ Election Comparison & Swing Analysis")
    years_available = sorted(df_edited['Year'].unique())
    if len(years_available) < 2:
        st.warning("Need at least 2 years of data.")
        return

    c1, c2 = st.columns(2)
    with c1:
        year_a = int(st.selectbox("Baseline Year", years_available, index=0, key="sw_ya"))
    with c2:
        year_b = int(st.selectbox("Target Year",   years_available, index=len(years_available)-1, key="sw_yb"))

    if year_a == year_b:
        st.error("Select two different years.")
        return

    df_a = df_edited[df_edited['Year'] == year_a].copy()
    df_b = df_edited[df_edited['Year'] == year_b].copy()

    tv_col    = smart_column_lookup(df_edited, "Votes Polled")
    cons_col  = smart_column_lookup(df_edited, "Constituency Name")
    win_col   = smart_column_lookup(df_edited, "Win Party")
    margin_col= smart_column_lookup(df_edited, "Margin")

    votes_a = safe_numeric(df_a[tv_col]).sum() if tv_col in df_a.columns else 0
    votes_b = safe_numeric(df_b[tv_col]).sum() if tv_col in df_b.columns else 0

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Total Votes {year_a}", f"{votes_a:,.0f}")
    m2.metric(f"Total Votes {year_b}", f"{votes_b:,.0f}")
    m3.metric("Δ Votes", f"{votes_b - votes_a:,.0f}", delta_color="normal")
    st.divider()

    # Constituency flippers
    if all(c in df_edited.columns for c in [cons_col, win_col]):
        merge_a = df_a[[cons_col, win_col]].rename(columns={win_col: f"Winner_{year_a}"})
        merge_b = df_b[[cons_col, win_col]].rename(columns={win_col: f"Winner_{year_b}"})
        comp    = pd.merge(merge_a, merge_b, on=cons_col, how="inner")
        flippers= comp[comp[f"Winner_{year_a}"] != comp[f"Winner_{year_b}"]]
        st.markdown(f"**{len(flippers)} constituencies changed hands.**")
        st.dataframe(flippers, use_container_width=True)

        gainers = flippers[f"Winner_{year_b}"].value_counts().reset_index()
        gainers.columns = ["Party", "Seats Gained"]
        fig_flip, ax_flip = plt.subplots(figsize=(8, 4))
        sns.barplot(data=gainers, x="Seats Gained", y="Party", palette="magma", ax=ax_flip)
        plt.title(f"Parties gaining seats in {year_b}")
        st.pyplot(fig_flip)
        plt.close()

        # Transition Matrix
        st.markdown("**🔄 Transition Matrix (who beat whom)**")
        tm = build_transition_matrix(df_a, df_b, cons_col, win_col)
        fig_tm, ax_tm = plt.subplots(figsize=(min(16, max(6, len(tm.columns))),
                                              min(14, max(5, len(tm.index)))))
        sns.heatmap(tm, annot=True, fmt='d', cmap='YlOrRd', ax=ax_tm)
        ax_tm.set_title(f"Seat Transition: {year_a} → {year_b}")
        ax_tm.set_xlabel(f"Winner {year_b}")
        ax_tm.set_ylabel(f"Winner {year_a}")
        plt.tight_layout()
        st.pyplot(fig_tm)
        plt.close()

        # Bloc-level swing
        st.markdown("**Bloc-level Seat Changes**")
        comp["Bloc_A"] = comp[f"Winner_{year_a}"].apply(assign_bloc)
        comp["Bloc_B"] = comp[f"Winner_{year_b}"].apply(assign_bloc)
        bloc_trans = pd.crosstab(comp["Bloc_A"], comp["Bloc_B"])
        st.dataframe(bloc_trans, use_container_width=True)

    # Margin distribution comparison
    if margin_col in df_a.columns and margin_col in df_b.columns:
        fig_m, ax_m = plt.subplots(figsize=(9, 4))
        safe_numeric(df_a[margin_col]).dropna().hist(bins=20, alpha=0.6, label=str(year_a), ax=ax_m, color='steelblue')
        safe_numeric(df_b[margin_col]).dropna().hist(bins=20, alpha=0.6, label=str(year_b), ax=ax_m, color='darkorange')
        ax_m.set_title("Margin Distribution Comparison")
        ax_m.legend(); ax_m.set_xlabel("Margin")
        st.pyplot(fig_m)
        plt.close()

def render_regional_analysis(df_edited):
    st.markdown("#### 🗺️ Regional Analysis (North / South / Central)")
    dist_col = smart_column_lookup(df_edited, "District")
    if dist_col not in df_edited.columns:
        st.info("No 'District' column found. Upload data with district information for regional analysis.")
        return

    NORTH_DISTRICTS  = ["Kasargod", "Kannur", "Wayanad", "Kozhikode", "Malappuram"]
    SOUTH_DISTRICTS  = ["Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha"]
    CENTRAL_DISTRICTS= ["Thrissur", "Palakkad", "Ernakulam", "Idukki", "Kottayam"]

    def get_region(d):
        if d in NORTH_DISTRICTS:   return "North"
        if d in SOUTH_DISTRICTS:   return "South"
        if d in CENTRAL_DISTRICTS: return "Central"
        return "Other"

    df_reg = df_edited.copy()
    df_reg["Region"] = df_reg[dist_col].astype(str).apply(get_region)

    win_col  = smart_column_lookup(df_edited, "Win Party")
    tv_col   = smart_column_lookup(df_edited, "Votes Polled")

    for party in ["INC", "CPM", "CPI"]:
        won = df_reg[df_reg[win_col] == party] if win_col in df_reg.columns else pd.DataFrame()
        if won.empty:
            continue
        region_wins = won.groupby(["Year", "Region"]).size().unstack(fill_value=0)
        st.markdown(f"**{party} — Wins by Region per Election**")
        st.dataframe(region_wins, use_container_width=True)

def render_reserved_seats(df_edited):
    st.markdown("#### 🏷️ Reserved Constituency Analysis (SC/ST)")
    res_col = smart_column_lookup(df_edited, "Category")
    if res_col not in df_edited.columns:
        res_col = smart_column_lookup(df_edited, "Reserved")
    if res_col not in df_edited.columns:
        st.info("No 'Category' / 'Reserved' column found for reserved seat analysis.")
        return

    reserved = df_edited[df_edited[res_col].str.upper().isin(["SC", "ST"])] if res_col in df_edited.columns else pd.DataFrame()
    general  = df_edited[~df_edited[res_col].str.upper().isin(["SC", "ST"])] if res_col in df_edited.columns else pd.DataFrame()

    win_col  = smart_column_lookup(df_edited, "Win Party")
    if win_col not in df_edited.columns:
        st.info("Win Party column not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reserved Seats — Top Winners**")
        if not reserved.empty:
            st.dataframe(reserved[win_col].value_counts().head(15).reset_index(), use_container_width=True)
    with c2:
        st.markdown("**General Seats — Top Winners**")
        if not general.empty:
            st.dataframe(general[win_col].value_counts().head(15).reset_index(), use_container_width=True)

# ============================================================
# MAIN APP
# ============================================================

st.title("📊 Kerala Election Insights & Trends Hub")

uploaded_files = st.file_uploader(
    "Upload Election Data (Excel/CSV)",
    accept_multiple_files=True,
    type=['xlsx', 'xls', 'csv']
)

if uploaded_files:
    with st.spinner("Processing data..."):
        master_df = load_and_combine_data(uploaded_files)

    if master_df is not None:
        st.toast(f"Data Loaded! {len(master_df)} records.", icon="✅")

        # --- CUSTOM METRICS ---
        if "custom_metrics" not in st.session_state:
            st.session_state.custom_metrics = {}

        with st.sidebar.expander("🛠️ Build Custom Metrics", expanded=False):
            st.caption("Create new columns using plain English.")
            new_metric_name = st.text_input("Name (e.g., Win_Margin_Percent)")
            new_metric_desc = st.text_area("Logic (e.g., Win Vote minus Run Vote divided by Votes Polled)")
            if st.button("Draft Metric"):
                if new_metric_name and new_metric_desc:
                    with st.spinner("Translating logic..."):
                        code = generate_custom_metric_code(master_df.head(), new_metric_name, new_metric_desc)
                        st.session_state.draft_code = code
                        st.session_state.draft_name = new_metric_name
                        st.rerun()
            if "draft_code" in st.session_state:
                st.write("---")
                st.write("**Preview Code:**")
                st.code(st.session_state.draft_code, language="python")
                try:
                    test_df = master_df.head(50).copy()
                    exec_globals = {"df": test_df, "pd": pd, "smart_lookup": smart_column_lookup}
                    exec(st.session_state.draft_code, exec_globals)
                    st.success("Test Calculation Successful!")
                    st.dataframe(test_df[[st.session_state.draft_name]].head(3))
                    if st.button("✅ Save Metric"):
                        st.session_state.custom_metrics[st.session_state.draft_name] = st.session_state.draft_code
                        del st.session_state.draft_code
                        st.rerun()
                except Exception as e:
                    st.error(f"Error in logic: {e}")

        if not master_df.empty and st.session_state.custom_metrics:
            for name, code in st.session_state.custom_metrics.items():
                try:
                    exec_globals = {"df": master_df, "pd": pd, "smart_lookup": smart_column_lookup}
                    exec(code, exec_globals)
                except Exception as e:
                    st.warning(f"Could not apply metric '{name}': {e}")
            st.sidebar.success(f"Active Metrics: {list(st.session_state.custom_metrics.keys())}")

        # --- SIDEBAR FILTERS ---
        st.sidebar.header("🔍 Global Filters")
        try:
            master_df['Year'] = pd.to_numeric(master_df['Year'])
            min_year, max_year = int(master_df['Year'].min()), int(master_df['Year'].max())
            selected_years = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
            df_filtered = master_df[(master_df['Year'] >= selected_years[0]) & (master_df['Year'] <= selected_years[1])]
        except:
            st.sidebar.warning("Year column issue. Showing all data.")
            df_filtered = master_df

        dist_col = smart_column_lookup(df_filtered, "District")
        if dist_col in df_filtered.columns:
            districts = sorted(df_filtered[dist_col].dropna().astype(str).unique())
            selected_districts = st.sidebar.multiselect("Filter Districts", options=districts)
            if selected_districts:
                df_filtered = df_filtered[df_filtered[dist_col].isin(selected_districts)]

        cons_col = smart_column_lookup(df_filtered, "Constituency Name")
        if cons_col in df_filtered.columns:
            constituencies = sorted(df_filtered[cons_col].dropna().astype(str).unique())
            selected_cons = st.sidebar.multiselect("Filter Constituencies", options=constituencies)
            if selected_cons:
                df_filtered = df_filtered[df_filtered[cons_col].isin(selected_cons)]

        party_col = smart_column_lookup(df_filtered, "Party")
        if party_col in df_filtered.columns:
            parties = sorted(df_filtered[party_col].dropna().astype(str).unique())
            selected_parties = st.sidebar.multiselect("Filter Parties", options=parties)
            if selected_parties:
                df_filtered = df_filtered[df_filtered[party_col].isin(selected_parties)]

        # Bloc filter
        if "Bloc" in df_filtered.columns:
            all_blocs = sorted(df_filtered["Bloc"].dropna().unique())
            selected_blocs = st.sidebar.multiselect("Filter Blocs (LDF/UDF/NDA)", options=all_blocs)
            if selected_blocs:
                df_filtered = df_filtered[df_filtered["Bloc"].isin(selected_blocs)]

        # Party family filter
        if "Party Family" in df_filtered.columns:
            all_families = sorted(df_filtered["Party Family"].dropna().unique())
            selected_families = st.sidebar.multiselect("Filter Party Families", options=all_families)
            if selected_families:
                df_filtered = df_filtered[df_filtered["Party Family"].isin(selected_families)]

        st.sidebar.caption(f"Showing {len(df_filtered)} rows")

        # --- DATA EDITOR ---
        st.subheader("📝 Data Editor & Analysis")
        with st.expander("View & Edit Raw Data (Add Rows/Change Values)", expanded=False):
            df_edited = st.data_editor(
                df_filtered, num_rows="dynamic",
                use_container_width=True, key="editor_main"
            )
            csv_data = df_edited.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Modified Data", csv_data, "modified_election_data.csv", "text/csv")

        # --- TABS ---
        tab_trends, tab_parties, tab_families, tab_blocs, tab_stats, tab_swing, tab_regional, tab_reserved, tab_ai, tab_data = st.tabs([
            "📈 Dashboard",
            "🎯 Party Analysis",
            "👨‍👩‍👧 Party Families",
            "🏛️ Bloc Analysis",
            "📐 Statistical Indices",
            "⚔️ Swing Analyzer",
            "🗺️ Regional Trends",
            "🏷️ Reserved Seats",
            "🤖 AI Analyst",
            "📋 Data Stats"
        ])

        # TAB 1: DASHBOARD
        with tab_trends:
            st.subheader("Visual Analysis")
            default_metrics = ["Votes", "Margin", "Electors"]
            custom_keys = list(st.session_state.custom_metrics.keys())
            available_metrics = list(set(default_metrics + custom_keys))

            c1, c2, c3, c4 = st.columns(4)
            with c1: chart_type = st.selectbox("Chart Type", ["Line (Time Series)", "Bar (Comparison)", "Pie (Share)", "Box (Distribution)"])
            with c2: metric    = st.selectbox("Metric", available_metrics, index=0)
            with c3: split_by  = st.selectbox("Category / Split", ["Party", "Bloc", "Party Family", "Alliance", "Constituency Name", "District", "None"], index=0)
            with c4: agg_type  = st.selectbox("Aggregation", ["Sum", "Average", "Maximum", "Count"], index=0)

            agg_map = {"Sum": "sum", "Average": "mean", "Maximum": "max", "Count": "count"}
            metric_col = smart_column_lookup(df_edited, metric)
            cat_col    = smart_column_lookup(df_edited, split_by) if split_by != "None" else None
            year_col   = "Year"

            if metric_col:
                try:
                    df_edited[metric_col] = pd.to_numeric(df_edited[metric_col], errors='coerce')
                    fig, ax = plt.subplots(figsize=(10, 6))

                    if "Line" in chart_type:
                        if year_col:
                            group_cols = [year_col] + ([cat_col] if cat_col else [])
                            data = df_edited.groupby(group_cols)[metric_col].agg(agg_map[agg_type]).reset_index()
                            if cat_col:
                                sns.lineplot(data=data, x=year_col, y=metric_col, hue=cat_col, marker="o", ax=ax)
                            else:
                                sns.lineplot(data=data, x=year_col, y=metric_col, marker="o", ax=ax)
                            plt.title(f"{agg_type} {metric} over Years")

                    elif "Bar" in chart_type:
                        if not cat_col:
                            st.warning("⚠️ Select a Category for Bar Charts.")
                        else:
                            data = df_edited.groupby(cat_col)[metric_col].agg(agg_map[agg_type]).reset_index()
                            data = data.sort_values(metric_col, ascending=False).head(15)
                            sns.barplot(data=data, x=metric_col, y=cat_col, palette="viridis", ax=ax)
                            plt.title(f"Top 15 {split_by} by {agg_type} {metric}")
                            ax.bar_label(ax.containers[0], fmt='%.0f', padding=3)

                    elif "Pie" in chart_type:
                        if not cat_col:
                            st.warning("⚠️ Select a Category for Pie Charts.")
                        else:
                            data = df_edited.groupby(cat_col)[metric_col].sum().reset_index()
                            data = data.sort_values(metric_col, ascending=False).head(10)
                            ax.pie(data[metric_col], labels=data[cat_col], autopct='%1.1f%%', startangle=140)
                            plt.title(f"{metric} Share (Top 10)")

                    elif "Box" in chart_type:
                        if not cat_col:
                            sns.boxplot(y=df_edited[metric_col], ax=ax)
                        else:
                            top_cats = df_edited[cat_col].value_counts().head(10).index
                            sub_data = df_edited[df_edited[cat_col].isin(top_cats)]
                            sns.boxplot(data=sub_data, x=cat_col, y=metric_col, ax=ax)
                            plt.xticks(rotation=45)
                        plt.title(f"Distribution of {metric}")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Plot Error: {e}")

        # TAB 2: PARTY ANALYSIS (NEW — from PDF Section C)
        with tab_parties:
            st.subheader("🎯 Individual Party Analysis")
            st.info("Detailed analysis for the 6 key parties listed in the PDF (CPI, CPM, INC, ML, KCM, KCJ), plus any custom party.")

            win_col_check = smart_column_lookup(df_edited, "Win Party")
            if win_col_check not in df_edited.columns:
                st.warning("'Win Party' column not found. Please verify column names.")
            else:
                available_parties_in_data = sorted(df_edited[win_col_check].dropna().unique().tolist())
                # Pre-select the 6 PDF parties that exist in data
                default_sel = [p for p in INDIVIDUAL_PARTIES if p in available_parties_in_data]
                selected_analysis_parties = st.multiselect(
                    "Select parties to analyze", available_parties_in_data,
                    default=default_sel
                )
                for party in selected_analysis_parties:
                    with st.expander(f"📊 {party}", expanded=(party in INDIVIDUAL_PARTIES[:2])):
                        render_party_analysis(df_edited, party)

                # Strike rate table for all selected parties
                st.divider()
                st.markdown("#### 🎯 Strike Rate Comparison")
                sr_rows = []
                for party in selected_analysis_parties:
                    for year, grp in df_edited.groupby("Year"):
                        won  = (grp[win_col_check] == party).sum()
                        cont = len(grp)  # approximate: seats per year = contested
                        sr   = calc_strike_rate(won, cont)
                        sr_rows.append({"Party": party, "Year": year, "Seats Won": won,
                                        "Contested (approx)": cont, "Strike Rate %": round(sr, 1) if sr else np.nan})
                if sr_rows:
                    sr_df = pd.DataFrame(sr_rows)
                    st.dataframe(sr_df, use_container_width=True)

                    fig_sr, ax_sr = plt.subplots(figsize=(11, 5))
                    for p in selected_analysis_parties:
                        sub = sr_df[sr_df["Party"] == p].sort_values("Year")
                        ax_sr.plot(sub["Year"], sub["Strike Rate %"], marker='o', label=p)
                    ax_sr.set_title("Strike Rate (%) over Elections")
                    ax_sr.legend(); ax_sr.set_xlabel("Year")
                    plt.tight_layout()
                    st.pyplot(fig_sr)
                    plt.close()

        # TAB 3: PARTY FAMILIES (NEW — from PDF Section B)
        with tab_families:
            st.subheader("👨‍👩‍👧 Party Family Analysis")
            st.markdown("Grouped analysis for Kerala Congress family, Muslim League family, Congress family, and Socialist family.")

            win_col_check = smart_column_lookup(df_edited, "Win Party")
            if "Party Family" not in df_edited.columns:
                st.warning("Party Family column not assigned. Check that 'Win Party' or 'Party' column exists.")
            else:
                family_stats = df_edited.groupby(["Year", "Party Family"]).size().unstack(fill_value=0)
                st.markdown("**Seats Won per Family per Election**")
                st.dataframe(family_stats, use_container_width=True)

                fig_fam, ax_fam = plt.subplots(figsize=(12, 5))
                for fam in family_stats.columns:
                    ax_fam.plot(family_stats.index, family_stats[fam], marker='o', label=fam)
                ax_fam.set_title("Party Family Seats Won over Elections")
                ax_fam.legend(loc='upper left', fontsize=8)
                ax_fam.set_xlabel("Year")
                plt.tight_layout()
                st.pyplot(fig_fam)
                plt.close()

                # Vote share by family
                tv_col = smart_column_lookup(df_edited, "Votes Polled")
                if tv_col in df_edited.columns and win_col_check in df_edited.columns:
                    family_votes = df_edited.groupby(["Year", "Party Family"])[tv_col].apply(
                        lambda x: safe_numeric(x).sum()
                    ).unstack(fill_value=0)
                    totals = df_edited.groupby("Year")[tv_col].apply(lambda x: safe_numeric(x).sum())
                    family_vote_pct = family_votes.div(totals, axis=0) * 100

                    st.markdown("**Vote % per Family per Election**")
                    st.dataframe(family_vote_pct.round(2), use_container_width=True)

        # TAB 4: BLOC ANALYSIS (NEW — PDF Section D)
        with tab_blocs:
            st.subheader("🏛️ LDF / UDF / NDA Bloc Analysis")
            render_bloc_analysis(df_edited)

        # TAB 5: STATISTICAL INDICES (NEW — PDF Section E)
        with tab_stats:
            st.subheader("📐 Statistical Indices (PDF Section E)")
            with st.expander("ℹ️ Index Glossary", expanded=False):
                st.markdown("""
| Index | Category | What it measures |
|---|---|---|
| Gallagher Index | Disproportionality | Vote→Seat distortion |
| Loosemore-Hanby | Disproportionality | Total over/under-representation |
| Seat Bonus | Disproportionality | Largest party seat bonus |
| Pedersen Index | Volatility | Net vote change between elections |
| ENEP | Party System | Effective # of electoral parties |
| ENPP | Party System | Effective # of parliamentary parties |
| Turnout Rate % | Participation | Votes / Electors |
| Fractionalization | Fragmentation | Party system fragmentation |
| Entropy Index | Fragmentation | Diversity of vote distribution |
| HHI | Dominance | Vote concentration |
| Avg Margin | Competitiveness | Average winning margin |
| Close Contest (<5%) | Competitiveness | Number of razor-thin wins |
| Safety Index | Competitiveness | Avg margin as % of votes |
                """)
            render_statistical_indices(df_edited)

        # TAB 6: SWING ANALYZER (Enhanced — was tab_compare)
        with tab_swing:
            render_swing_analyzer(df_edited)

        # TAB 7: REGIONAL TRENDS (NEW — PDF: Regional strength of INC, CPM, CPI)
        with tab_regional:
            st.subheader("🗺️ Regional Analysis")
            render_regional_analysis(df_edited)

        # TAB 8: RESERVED SEATS (NEW — PDF: SC/ST analysis)
        with tab_reserved:
            st.subheader("🏷️ Reserved Seat Analysis")
            render_reserved_seats(df_edited)

        # TAB 9: AI ANALYST
        with tab_ai:
            st.markdown("### 🤖 Ask questions about your data")
            st.info("Tip: Ask about any of the 44 statistical indices, party trends, bloc comparisons, or swing analysis.")

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask your data..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)
                with st.chat_message("assistant"):
                    with st.status("🧠 Analyzing...", expanded=True) as status:
                        fig, text_response, code = query_gemini_smart(prompt, df_edited)
                        status.update(label="Complete", state="complete", expanded=False)
                        if text_response:
                            st.markdown(f"**Insight:** \n {text_response}")
                            st.session_state.messages.append({"role": "assistant", "content": text_response})
                        elif not fig:
                            st.warning("Analysis complete, check code.")
                        if fig:
                            st.pyplot(fig)
                            plt.close()
                        with st.expander("🔎 View Python Logic"):
                            st.code(code, language="python")

        # TAB 10: DATA STATS
        with tab_data:
            st.info("The detailed raw data is available in the 'View & Edit' section above.")
            st.write("**Dataset Statistics:**")
            st.write(df_edited.describe())
            st.write("**Party Family Distribution:**")
            if "Party Family" in df_edited.columns:
                st.dataframe(df_edited["Party Family"].value_counts().reset_index(), use_container_width=True)
            st.write("**Bloc Distribution:**")
            if "Bloc" in df_edited.columns:
                st.dataframe(df_edited["Bloc"].value_counts().reset_index(), use_container_width=True)
else:
    st.info("👆 Upload Data to start.")