import matplotlib
matplotlib.use('Agg')  # Prevents backend errors

import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import re
import numpy as np

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Election Trends & AI Analyst", layout="wide")

# API Key Handling
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.warning("API Key not found in secrets. Please enter it below:")
        api_key = st.text_input("Gemini API Key", type="password")

# --- 2. DOMAIN KNOWLEDGE (FROM PDF) ---
# Mappings based on the uploaded PDF content
PARTY_FAMILIES = {
    "Kerala Congress Family": ["KC", "KCM", "KCJ", "KCJB", "KCB", "KCS", "KCST", "KIS", "KCAMG", "KCD"],
    "Muslim League Family": ["ML", "AIML", "INL", "IUML"],
    "Socialist Family": ["PSP", "RSP", "SSP", "ISP", "KTP", "CS", "KSP", "LKD", "LJD", "BLD", "DSP", "ICS", "JDU", "NCP", "RSPB"],
    "Congress Family": ["INC", "INCO", "INCA", "CS"],
    "Left Core": ["CPM", "CPI"],
    "NDA Core": ["BJP", "BDJS"]
}

DISTRICT_REGIONS = {
    "North": ["Kasaragod", "Kannur", "Wayanad", "Kozhikode", "Malappuram"],
    "Central": ["Palakkad", "Thrissur", "Ernakulam", "Idukki"],
    "South": ["Kottayam", "Alappuzha", "Pathanamthitta", "Kollam", "Thiruvananthapuram"]
}

PRESET_METRICS = {
    "Win_Margin_Abs": {
        "desc": "Absolute vote difference between Winner and Runner-up",
        "code": "df['Win_Margin_Abs'] = pd.to_numeric(df[smart_lookup(df, 'Win Vote')], errors='coerce') - pd.to_numeric(df[smart_lookup(df, 'Run vote')], errors='coerce')"
    },
    "Win_Margin_Percent": {
        "desc": "Margin as a % of Total Votes Polled (Closeness Index)",
        "code": "df['Win_Margin_Percent'] = (pd.to_numeric(df[smart_lookup(df, 'Win Vote')], errors='coerce') - pd.to_numeric(df[smart_lookup(df, 'Run vote')], errors='coerce')) / pd.to_numeric(df[smart_lookup(df, 'Votes polled')], errors='coerce') * 100"
    },
    "Victory_Risk_Index": {
        "desc": "Inverse of Margin % (Higher value = Higher Risk/Closer Contest)",
        "code": "df['Victory_Risk_Index'] = 1 / ((pd.to_numeric(df[smart_lookup(df, 'Win Vote')], errors='coerce') - pd.to_numeric(df[smart_lookup(df, 'Run vote')], errors='coerce')) / pd.to_numeric(df[smart_lookup(df, 'Votes polled')], errors='coerce') + 0.001)"
    },
    "Turnout_Ratio": {
        "desc": "Votes Polled / Electors",
        "code": "df['Turnout_Ratio'] = pd.to_numeric(df[smart_lookup(df, 'Votes polled')], errors='coerce') / pd.to_numeric(df[smart_lookup(df, 'Elecors')], errors='coerce')"
    }
}

# --- 3. HELPER FUNCTIONS ---
def smart_column_lookup(df, guessed_name):
    """Fuzzy matching to find the correct column name."""
    if guessed_name in df.columns: return guessed_name
    for col in df.columns:
        if col.lower() == guessed_name.lower(): return col
    matches = difflib.get_close_matches(guessed_name, df.columns, n=1, cutoff=0.5)
    return matches[0] if matches else guessed_name

def enrich_data(df):
    """Adds Family and Region columns based on PDF definitions."""
    # 1. Map Party Families
    def get_family(party):
        party = str(party).strip().upper()
        for family, members in PARTY_FAMILIES.items():
            if party in members:
                return family
        return "Other/Independent"

    # Try to find the 'Party' or 'Win Party' column
    party_col = smart_column_lookup(df, "Win Party")
    if party_col:
        df['Party_Family'] = df[party_col].apply(get_family)

    # 2. Map Regions
    # Try to find District column
    dist_col = smart_column_lookup(df, "District")
    # If not found, try to map from Constituency Name if separate mapping exists (skipped for now, relies on District col)
    if dist_col in df.columns:
        def get_region(dist):
            for region, districts in DISTRICT_REGIONS.items():
                if str(dist) in districts:
                    return region
            return "Unknown"
        df['Region'] = df[dist_col].apply(get_region)
        
    return df

@st.cache_data
def load_and_combine_data(uploaded_files):
    all_dfs = []
    for file in uploaded_files:
        try:
            file_ext = file.name.split('.')[-1].lower()
            
            def process_df(df_temp, source_name):
                # Fix Cons No to string
                if 'Cons No.' in df_temp.columns:
                    df_temp['Cons No.'] = df_temp['Cons No.'].astype(str)
                # Fix Year
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
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # --- NEW: ENRICH DATA WITH FAMILIES & REGIONS ---
        return enrich_data(combined_df)
    return None

def generate_custom_metric_code(df, metric_name, description):
    """Asks AI to generate pandas code."""
    genai.configure(api_key=api_key)
    columns_list = list(df.columns)
    prompt = f"""
    You are a Pandas expert. Write Python code to create column '{metric_name}' in `df`.
    Context Columns: {columns_list}
    Description: "{description}"
    Rules: Use `smart_lookup(df, 'col')`. Ensure numeric conversion. Return ONLY code.
    """
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    return model.generate_content(prompt).text.strip()

def query_gemini_smart(query, dataframe):
    """AI Agent with enhanced PDF context."""
    if not api_key: return None, None, "⚠️ Please enter API Key."
    genai.configure(api_key=api_key)
    
    # Contextualize with PDF definitions
    pdf_context = f"""
    ### DOMAIN KNOWLEDGE (FROM USER PDF)
    - **Party Families**: {PARTY_FAMILIES}
    - **Regions**: {DISTRICT_REGIONS}
    - **Indices**: The user is interested in Strike Rate, HHI, Margin, Swing, and Risk Indices.
    - If asked about "Strike Rate", note that dataset only has Winner/Runner-up, so calculate based on available wins/losses in the data.
    """
    
    system_instruction = f"""
    You are an expert Election Analyst Python Agent.
    1. Given DataFrame `df`. Use `smart_lookup(df, 'col')` for columns.
    2. **GOAL**: Write Python code to answer user question.
    3. Outputs: `fig` (matplotlib) for plots, `answer` (str) for text.
    4. Return ONLY valid Python code.
    {pdf_context}
    """
    
    try:
        data_sample = dataframe.sample(n=3).to_markdown()
    except:
        data_sample = dataframe.head(3).to_markdown()
        
    prompt = f"{system_instruction}\n### DATA\nCols: {list(dataframe.columns)}\nSample:\n{data_sample}\n### REQUEST\n{query}"
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            code = response.text.replace("```python", "").replace("```", "").strip()
            exec_globals = {"df": dataframe, "plt": plt, "sns": sns, "pd": pd, "np": np, 
                           "smart_lookup": smart_column_lookup, "fig": None, "answer": None}
            exec(code, exec_globals)
            return exec_globals.get('fig'), exec_globals.get('answer'), code
        except Exception as e:
            prompt += f"\nFix error: {e}"
            print(f"Retrying... {e}")
            
    return None, None, "Failed to generate analysis."

# --- 4. MAIN APP UI ---
st.title("📊 Election Insights & Trends Hub")

# File Uploader
with st.sidebar:
    st.header("📂 Data Import")
    uploaded_files = st.file_uploader("Upload Data (Excel/CSV)", accept_multiple_files=True, type=['xlsx', 'xls', 'csv'])

if uploaded_files:
    with st.spinner("Processing & Enriching Data..."):
        master_df = load_and_combine_data(uploaded_files)
    
    if master_df is not None:
        st.toast(f"Loaded {len(master_df)} records.", icon="✅")
        
        # --- CUSTOM METRICS SECTION ---
        if "custom_metrics" not in st.session_state: st.session_state.custom_metrics = {}

        with st.sidebar.expander("🛠️ Custom & Advanced Metrics", expanded=False):
            # A. PRESETS FROM PDF
            st.markdown("**Load PDF Presets**")
            preset_choice = st.selectbox("Select Index/Metric", ["Select..."] + list(PRESET_METRICS.keys()))
            if preset_choice != "Select..." and st.button("Load Preset"):
                meta = PRESET_METRICS[preset_choice]
                st.session_state.draft_name = preset_choice
                st.session_state.draft_code = meta["code"]
                st.info(f"Loaded logic for: {meta['desc']}")

            st.divider()
            
            # B. AI GENERATOR
            st.markdown("**Or Draft with AI**")
            new_metric_name = st.text_input("Name (e.g., Efficiency)")
            new_metric_desc = st.text_area("Logic (e.g., Votes / Electors)")
            if st.button("Draft Metric"):
                if new_metric_name and new_metric_desc:
                    with st.spinner("Translating..."):
                        code = generate_custom_metric_code(master_df.head(), new_metric_name, new_metric_desc)
                        st.session_state.draft_code = code
                        st.session_state.draft_name = new_metric_name
                        st.rerun()

            # C. PREVIEW & SAVE
            if "draft_code" in st.session_state:
                st.code(st.session_state.draft_code, language="python")
                if st.button("✅ Save Metric"):
                    st.session_state.custom_metrics[st.session_state.draft_name] = st.session_state.draft_code
                    del st.session_state.draft_code
                    st.rerun()

        # APPLY METRICS
        if st.session_state.custom_metrics:
            for name, code in st.session_state.custom_metrics.items():
                try:
                    exec_globals = {"df": master_df, "pd": pd, "smart_lookup": smart_column_lookup}
                    exec(code, exec_globals)
                except Exception as e:
                    st.warning(f"Metric '{name}' failed: {e}")

        # --- FILTERS ---
        st.sidebar.header("🔍 Filters")
        try:
            master_df['Year'] = pd.to_numeric(master_df['Year'])
            min_y, max_y = int(master_df['Year'].min()), int(master_df['Year'].max())
            sel_years = st.sidebar.slider("Years", min_y, max_y, (min_y, max_y))
            df_filtered = master_df[(master_df['Year'] >= sel_years[0]) & (master_df['Year'] <= sel_years[1])]
        except:
            df_filtered = master_df

        # Region Filter (New from PDF)
        if 'Region' in df_filtered.columns:
            regions = sorted(df_filtered['Region'].astype(str).unique())
            sel_region = st.sidebar.multiselect("Region (North/Central/South)", regions)
            if sel_region: df_filtered = df_filtered[df_filtered['Region'].isin(sel_region)]

        # District Filter
        dist_col = smart_column_lookup(df_filtered, "District")
        if dist_col in df_filtered.columns:
            dists = sorted(df_filtered[dist_col].dropna().astype(str).unique())
            sel_dist = st.sidebar.multiselect("District", dists)
            if sel_dist: df_filtered = df_filtered[df_filtered[dist_col].isin(sel_dist)]

        # Constituency Filter
        cons_col = smart_column_lookup(df_filtered, "Constituency Name")
        if cons_col in df_filtered.columns:
            cons = sorted(df_filtered[cons_col].dropna().astype(str).unique())
            sel_cons = st.sidebar.multiselect("Constituency", cons)
            if sel_cons: df_filtered = df_filtered[df_filtered[cons_col].isin(sel_cons)]

        # Party Family Filter (New from PDF)
        if 'Party_Family' in df_filtered.columns:
            fams = sorted(df_filtered['Party_Family'].astype(str).unique())
            sel_fam = st.sidebar.multiselect("Party Family", fams)
            if sel_fam: df_filtered = df_filtered[df_filtered['Party_Family'].isin(sel_fam)]

        # Party Filter
        party_col = smart_column_lookup(df_filtered, "Party")
        if party_col in df_filtered.columns:
            parties = sorted(df_filtered[party_col].dropna().astype(str).unique())
            sel_party = st.sidebar.multiselect("Party", parties)
            if sel_party: df_filtered = df_filtered[df_filtered[party_col].isin(sel_party)]

        st.sidebar.caption(f"Rows: {len(df_filtered)}")

        # --- DATA EDITOR ---
        with st.expander("📝 View & Edit Data", expanded=False):
            df_edited = st.data_editor(df_filtered, num_rows="dynamic", use_container_width=True, key="editor")
            st.download_button("Download CSV", df_edited.to_csv(index=False).encode('utf-8'), "data.csv")

        # --- TABS ---
        tab_trends, tab_compare, tab_ai, tab_stats = st.tabs(["📈 Dashboard", "⚔️ Compare", "🤖 AI Analyst", "📋 Stats"])

        # TAB 1: DASHBOARD
        with tab_trends:
            st.subheader("Visual Analytics")
            
            # Setup Metrics
            def_metrics = ["Votes", "Margin", "Electors"]
            cust_metrics = list(st.session_state.custom_metrics.keys())
            all_metrics = list(set(def_metrics + cust_metrics))
            
            # Controls
            c1, c2, c3, c4 = st.columns(4)
            with c1: c_type = st.selectbox("Chart", ["Line", "Bar", "Pie", "Box"])
            with c2: met = st.selectbox("Metric", all_metrics)
            # Add new split options from PDF logic
            with c3: split = st.selectbox("Split By", ["Party", "Party_Family", "Region", "Alliance", "District", "None"])
            with c4: agg = st.selectbox("Agg", ["Sum", "Average", "Maximum", "Count"])
            
            agg_map = {"Sum": "sum", "Average": "mean", "Maximum": "max", "Count": "count"}
            met_col = smart_column_lookup(df_edited, met)
            cat_col = smart_column_lookup(df_edited, split) if split != "None" else None
            year_col = smart_column_lookup(df_edited, "Year")

            if met_col:
                try:
                    df_edited[met_col] = pd.to_numeric(df_edited[met_col], errors='coerce')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    if "Line" in c_type and year_col:
                        grp = [year_col] + ([cat_col] if cat_col else [])
                        data = df_edited.groupby(grp)[met_col].agg(agg_map[agg]).reset_index()
                        sns.lineplot(data=data, x=year_col, y=met_col, hue=cat_col if cat_col else None, marker="o", ax=ax)
                        
                    elif "Bar" in c_type and cat_col:
                        data = df_edited.groupby(cat_col)[met_col].agg(agg_map[agg]).reset_index().sort_values(met_col, ascending=False).head(15)
                        sns.barplot(data=data, x=met_col, y=cat_col, palette="viridis", ax=ax)
                        
                    elif "Pie" in c_type and cat_col:
                        data = df_edited.groupby(cat_col)[met_col].sum().reset_index().sort_values(met_col, ascending=False).head(10)
                        ax.pie(data[met_col], labels=data[cat_col], autopct='%1.1f%%')
                        
                    elif "Box" in c_type:
                        if cat_col:
                            top = df_edited[cat_col].value_counts().head(10).index
                            sns.boxplot(data=df_edited[df_edited[cat_col].isin(top)], x=cat_col, y=met_col, ax=ax)
                        else:
                            sns.boxplot(y=df_edited[met_col], ax=ax)
                            
                    st.pyplot(fig)
                except Exception as e: st.error(f"Plot Error: {e}")

        # TAB 2: COMPARE
        with tab_compare:
            st.subheader("Swing & Flip Analysis")
            years = sorted(df_edited['Year'].unique())
            if len(years) < 2:
                st.warning("Need 2+ years.")
            else:
                c1, c2 = st.columns(2)
                y1 = int(st.selectbox("Year A", years, index=0))
                y2 = int(st.selectbox("Year B", years, index=len(years)-1))
                
                if y1 != y2:
                    df1 = df_edited[df_edited['Year'] == y1]
                    df2 = df_edited[df_edited['Year'] == y2]
                    
                    # Totals
                    v1 = pd.to_numeric(df1[smart_column_lookup(df1, 'Votes Polled')], errors='coerce').sum()
                    v2 = pd.to_numeric(df2[smart_column_lookup(df2, 'Votes Polled')], errors='coerce').sum()
                    st.metric("Total Votes Swing", f"{v2-v1:,.0f}", help="Difference in total votes polled")
                    
                    # Seat Flips
                    st.markdown("#### 🔄 Seat Flips")
                    cons_col = smart_column_lookup(df_edited, "Constituency Name")
                    win_col = smart_column_lookup(df_edited, "Win Party")
                    
                    if cons_col and win_col:
                        m1 = df1[[cons_col, win_col]].rename(columns={win_col: f"Win_{y1}"})
                        m2 = df2[[cons_col, win_col]].rename(columns={win_col: f"Win_{y2}"})
                        merged = pd.merge(m1, m2, on=cons_col)
                        flips = merged[merged[f"Win_{y1}"] != merged[f"Win_{y2}"]]
                        
                        st.write(f"**{len(flips)} Seats Changed Hands**")
                        st.dataframe(flips, use_container_width=True)
                        
                        # Flip Visual
                        gains = flips[f"Win_{y2}"].value_counts().reset_index()
                        gains.columns = ["Party", "Gains"]
                        fig, ax = plt.subplots(figsize=(8,4))
                        sns.barplot(data=gains, x="Gains", y="Party", palette="magma", ax=ax)
                        st.pyplot(fig)

        # TAB 3: AI ANALYST
        with tab_ai:
            st.markdown("### 🤖 Ask about Trends, Families, or Indices")
            st.caption("Try: *'Analyze the performance of the Socialist Family'*, *'Plot the Victory Risk Index trend'*, *'Compare North vs South vote share'*")
            
            if "messages" not in st.session_state: st.session_state.messages = []
            for m in st.session_state.messages:
                with st.chat_message(m["role"]): st.markdown(m["content"])
                
            if prompt := st.chat_input("Ask Question..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                
                with st.chat_message("assistant"):
                    with st.status("Analyzing...", expanded=True):
                        fig, txt, code = query_gemini_smart(prompt, df_edited)
                        if txt: 
                            st.markdown(txt)
                            st.session_state.messages.append({"role": "assistant", "content": txt})
                        if fig: st.pyplot(fig)
                        with st.expander("Code"): st.code(code)

        # TAB 4: STATS
        with tab_stats:
            st.write(df_edited.describe())