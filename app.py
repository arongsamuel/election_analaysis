import matplotlib
matplotlib.use('Agg')  # Prevents backend errors

import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import re

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Election Trends & AI Analyst", layout="wide")

# API Key Input
# Automatically fetch API Key from secrets
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # Fallback if secret isn't found (useful for first-time setup)
    with st.sidebar:
        st.warning("API Key not found in secrets. Please enter it below:")
        api_key = st.text_input("Gemini API Key", type="password")
with st.sidebar:
    st.header("⚙️ Configuration")
    st.divider()
    st.info("Upload Excel (.xlsx) files with multiple sheets or individual CSVs.")

# --- 2. HELPER FUNCTIONS ---
def smart_column_lookup(df, guessed_name):
    """Fuzzy matching to find the correct column name."""
    # 1. Exact match
    if guessed_name in df.columns:
        return guessed_name
    
    # 2. Case-insensitive
    for col in df.columns:
        if col.lower() == guessed_name.lower():
            return col
            
    # 3. Fuzzy match
    matches = difflib.get_close_matches(guessed_name, df.columns, n=1, cutoff=0.5)
    return matches[0] if matches else guessed_name

# ... keep your imports ...

# --- NEW: Add caching to speed up the app ---
@st.cache_data
def load_and_combine_data(uploaded_files):
    all_dfs = []
    
    for file in uploaded_files:
        try:
            file_ext = file.name.split('.')[-1].lower()
            
            # Helper to process each dataframe before adding to list
            def process_df(df_temp, source_name):
                # 1. Force "Cons No." to be string to fix ArrowInvalid error
                if 'Cons No.' in df_temp.columns:
                    df_temp['Cons No.'] = df_temp['Cons No.'].astype(str)
                
                # 2. Ensure Year exists
                if 'Year' not in df_temp.columns:
                    try:
                        year_match = re.search(r'\d{4}', str(source_name))
                        df_temp['Year'] = int(year_match.group(0)) if year_match else source_name
                    except:
                        df_temp['Year'] = source_name
                return df_temp

            # --- CASE A: EXCEL ---
            if file_ext in ['xlsx', 'xls']:
                xls_dict = pd.read_excel(file, sheet_name=None)
                for sheet_name, df in xls_dict.items():
                    df = process_df(df, sheet_name)
                    all_dfs.append(df)
                    
            # --- CASE B: CSV ---
            elif file_ext == 'csv':
                df = pd.read_csv(file)
                df = process_df(df, file.name)
                all_dfs.append(df)
                
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")
            
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return None

def query_gemini_smart(query, dataframe):
    """
    AI Agent that can generate Text Insights AND/OR Visualizations.
    """
    if not api_key:
        return None, None, "⚠️ Please enter your API Key in the sidebar."
    
    genai.configure(api_key=api_key)
    
    # Create a small sample for context (random sampling helps avoid bias)
    try:
        data_sample = dataframe.sample(n=5).to_markdown()
    except:
        data_sample = dataframe.head(5).to_markdown()
        
    columns_list = list(dataframe.columns)
    
    system_instruction = """
    You are an expert Election Analyst Python Agent.
    1. You are given a pandas DataFrame `df`.
    2. ALWAYS use `smart_lookup(df, 'col_name')` to access columns.
    3. **YOUR GOAL**: Write Python code to answer the user's question.
    
    **OUTPUT REQUIREMENTS (IMPORTANT)**:
    - If the user asks for a **Plot/Graph**: Create a matplotlib figure named `fig`.
    - If the user asks for **Text/Insight**: Calculate the answer and store it in a string variable named `answer`.
    - You can do BOTH (create `fig` and define `answer`).
    - Return ONLY valid Python code. No markdown, no comments outside the code.
    """

    prompt = f"""
    {system_instruction}
    
    ### DATA CONTEXT
    Columns: {columns_list}
    Sample Data:
    {data_sample}

    ### USER REQUEST
    {query}
    """
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            code = response.text.replace("```python", "").replace("```", "").strip()
            
            # Sandbox environment
            exec_globals = {
                "df": dataframe, 
                "plt": plt, 
                "sns": sns, 
                "pd": pd, 
                "smart_lookup": smart_column_lookup,
                "fig": None,
                "answer": None # <--- New variable to capture text
            }
            exec(code, exec_globals)
            
            # Retrieve results
            generated_fig = exec_globals.get('fig')
            generated_text = exec_globals.get('answer')
            
            if generated_fig is None and generated_text is None:
                # If code ran but produced nothing, try to capture printed output
                pass 
                
            return generated_fig, generated_text, code
            
        except Exception as e:
            prompt += f"\n\nPrevious code failed with error: {e}. Fix it."
            print(f"Retrying... {e}")
            
    return None, None, f"Failed to generate analysis."

# --- 3. MAIN APP UI ---
st.title("📊 Election Insights & Trends Hub")

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
        
        # --- B. SIDEBAR FILTERS ---
        st.sidebar.header("🔍 Global Filters")
        
        # Year Filter
        try:
            master_df['Year'] = pd.to_numeric(master_df['Year'])
            min_year, max_year = int(master_df['Year'].min()), int(master_df['Year'].max())
            selected_years = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))
            df_filtered = master_df[(master_df['Year'] >= selected_years[0]) & (master_df['Year'] <= selected_years[1])]
        except:
            st.sidebar.warning("Year column issue. Showing all data.")
            df_filtered = master_df

        # Party Filter
        party_col = smart_column_lookup(df_filtered, "Party")
        if party_col:
            parties = sorted(df_filtered[party_col].astype(str).unique())
            selected_parties = st.sidebar.multiselect("Filter Parties", options=parties)
            if selected_parties:
                df_filtered = df_filtered[df_filtered[party_col].isin(selected_parties)]

        # --- C. TABS ---
        tab_trends, tab_ai, tab_data = st.tabs(["📈 Dashboard", "🤖 AI Analyst", "📋 Data"])

        # TAB 1: VISUAL TRENDS
        with tab_trends:
            st.subheader("Quick Trends")
            col_x, col_y = st.columns(2)
            with col_x:
                metric = st.selectbox("Metric", ["Votes", "Margin", "Electors"])
            with col_y:
                split_by = st.selectbox("Split By", ["Party", "Alliance", "None"], index=0)
            
            metric_col = smart_column_lookup(df_filtered, metric)
            year_col = smart_column_lookup(df_filtered, "Year")
            
            if metric_col and year_col:
                try:
                    group_cols = [year_col]
                    if split_by != "None":
                        split_col = smart_column_lookup(df_filtered, split_by)
                        group_cols.append(split_col)
                    
                    chart_data = df_filtered.groupby(group_cols)[metric_col].sum().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    if split_by != "None":
                        sns.lineplot(data=chart_data, x=year_col, y=metric_col, hue=split_col, marker="o", ax=ax)
                    else:
                        sns.lineplot(data=chart_data, x=year_col, y=metric_col, marker="o", ax=ax)
                    
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot: {e}")

        # TAB 2: AI ANALYST (UPDATED)
        with tab_ai:
            st.markdown("### 🤖 Ask questions in plain English")
            st.caption("Examples: *'Who won the most seats in 1980?'*, *'Plot the margin trend for INC'*, *'What is the average vote share?'*")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # If there was a plot associated with this message, we can't easily persist it in this simple list
                    # usually, we just persist text. For complex history, we'd need a robust state object.

            if prompt := st.chat_input("Ask your data..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.chat_message("assistant"):
                    with st.status("🧠 Thinking...", expanded=True) as status:
                        fig, text_response, code = query_gemini_smart(prompt, df_filtered)
                        
                        status.update(label="Complete", state="complete", expanded=False)
                        
                        # 1. Show Text Insight
                        if text_response:
                            st.markdown(f"**Insight:** \n {text_response}")
                            st.session_state.messages.append({"role": "assistant", "content": text_response})
                        elif not fig:
                            st.warning("The agent ran the code but didn't return a specific answer. Check the code below.")

                        # 2. Show Plot
                        if fig:
                            st.pyplot(fig)
                            
                        # 3. Show Code (Optional)
                        with st.expander("🔎 View Python Logic"):
                            st.code(code, language="python")

        # TAB 3: DATA
        with tab_data:
            st.dataframe(df_filtered)
else:
    st.info("👆 Upload Data to start.")
