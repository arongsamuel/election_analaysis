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
def generate_custom_metric_code(df, metric_name, description):
    """
    Asks AI to generate pandas code for a new column based on description.
    """
    genai.configure(api_key=api_key)
    
    columns_list = list(df.columns)
    
    prompt = f"""
    You are a Python Pandas expert.
    Task: Write a Python snippet to create a new column named '{metric_name}' in the dataframe `df`.
    
    ### CONTEXT
    - Existing Columns: {columns_list}
    - User Description: "{description}"
    
    ### RULES
    1. Use `smart_lookup(df, 'column_name')` for ALL column references. 
       (Example: `df[smart_lookup(df, 'Votes Polled')]`)
    2. Ensure columns are numeric before math. Use `pd.to_numeric(..., errors='coerce')`.
    3. Return ONLY the code snippet. No markdown, no comments.
    
    ### EXAMPLE OUTPUT
    df['{metric_name}'] = pd.to_numeric(df[smart_lookup(df, 'Win Vote')], errors='coerce') / pd.to_numeric(df[smart_lookup(df, 'Electors')], errors='coerce')
    """
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    response = model.generate_content(prompt)
    return response.text.strip()

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
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
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
        
        # ==========================================
        # 1. NEW: CUSTOM METRICS SECTION STARTS HERE
        # ==========================================
        
        # Initialize Session State for Custom Metrics if not exists
        if "custom_metrics" not in st.session_state:
            st.session_state.custom_metrics = {}

        # --- SIDEBAR: METRIC BUILDER ---
        with st.sidebar.expander("🛠️ Build Custom Metrics", expanded=False):
            st.caption("Create new columns using plain English.")
            
            # Inputs
            new_metric_name = st.text_input("Name (e.g., Win_Margin_Percent)")
            new_metric_desc = st.text_area("Logic (e.g., Win Vote minus Run Vote divided by Votes Polled)")
            
            if st.button("Draft Metric"):
                if new_metric_name and new_metric_desc:
                    with st.spinner("Translating logic..."):
                        # Generate Code using your helper function
                        code = generate_custom_metric_code(master_df.head(), new_metric_name, new_metric_desc)
                        st.session_state.draft_code = code
                        st.session_state.draft_name = new_metric_name
                        st.rerun()

            # Preview & Save Section
            if "draft_code" in st.session_state:
                st.write("---")
                st.write("**Preview Code:**")
                st.code(st.session_state.draft_code, language="python")
                
                # Test Run on a small sample
                try:
                    test_df = master_df.head(50).copy()
                    exec_globals = {"df": test_df, "pd": pd, "smart_lookup": smart_column_lookup}
                    exec(st.session_state.draft_code, exec_globals)
                    
                    # Show result
                    st.success("Test Calculation Successful!")
                    st.dataframe(test_df[[st.session_state.draft_name]].head(3))
                    
                    if st.button("✅ Save Metric"):
                        st.session_state.custom_metrics[st.session_state.draft_name] = st.session_state.draft_code
                        del st.session_state.draft_code # Clear draft
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error in logic: {e}")

        # --- APPLY SAVED METRICS TO MASTER DATA ---
        if not master_df.empty and st.session_state.custom_metrics:
            for name, code in st.session_state.custom_metrics.items():
                try:
                    # Execute the saved code on the full dataset
                    exec_globals = {"df": master_df, "pd": pd, "smart_lookup": smart_column_lookup}
                    exec(code, exec_globals)
                    # Note: master_df is modified in-place by the exec code
                except Exception as e:
                    st.warning(f"Could not apply metric '{name}': {e}")
            
            st.sidebar.success(f"Active Metrics: {list(st.session_state.custom_metrics.keys())}")

        # ==========================================
        # END OF CUSTOM METRICS SECTION
        # ==========================================

        # --- B. SIDEBAR FILTERS (Existing Code) ---
        st.sidebar.header("🔍 Global Filters")
        # ... (Rest of your code continues here)
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
            st.subheader("Visual Analysis")
            
            # 1. Setup Data Options
            default_metrics = ["Votes", "Margin", "Electors"]
            custom_keys = list(st.session_state.custom_metrics.keys()) if "custom_metrics" in st.session_state else []
            available_metrics = list(set(default_metrics + custom_keys))
            
            # 2. Control Panel
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                chart_type = st.selectbox("Chart Type", ["Line (Time Series)", "Bar (Comparison)", "Pie (Share)", "Box (Distribution)"])
            with c2:
                metric = st.selectbox("Metric", available_metrics, index=0)
            with c3:
                # Renamed "Split By" to "Category" for clarity, but logic is similar
                split_by = st.selectbox("Category / Split", ["Party", "Alliance", "Constituency Name", "District", "None"], index=0)
            with c4:
                agg_type = st.selectbox("Aggregation", ["Sum", "Average", "Maximum", "Count"], index=0)

            # Map selection to pandas function strings
            agg_map = {"Sum": "sum", "Average": "mean", "Maximum": "max", "Count": "count"}
            
            # 3. Resolve Columns
            metric_col = smart_column_lookup(df_filtered, metric)
            cat_col = smart_column_lookup(df_filtered, split_by) if split_by != "None" else None
            year_col = smart_column_lookup(df_filtered, "Year")

            if metric_col:
                try:
                    # Force numeric
                    df_filtered[metric_col] = pd.to_numeric(df_filtered[metric_col], errors='coerce')
                    
                    # --- PLOT LOGIC SWITCHER ---
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # A. LINE CHART (Time Series)
                    if "Line" in chart_type:
                        if year_col:
                            group_cols = [year_col]
                            if cat_col: group_cols.append(cat_col)
                            
                            data = df_filtered.groupby(group_cols)[metric_col].agg(agg_map[agg_type]).reset_index()
                            
                            if cat_col:
                                sns.lineplot(data=data, x=year_col, y=metric_col, hue=cat_col, marker="o", ax=ax)
                            else:
                                sns.lineplot(data=data, x=year_col, y=metric_col, marker="o", ax=ax)
                            plt.title(f"{agg_type} {metric} over Years")

                    # B. BAR CHART (Categorical Comparison)
                    elif "Bar" in chart_type:
                        if not cat_col:
                            st.warning("⚠️ Please select a Category (e.g., Party) to create a Bar Chart.")
                        else:
                            # Group by Category only (aggregating across all selected years)
                            data = df_filtered.groupby(cat_col)[metric_col].agg(agg_map[agg_type]).reset_index()
                            # Sort for better visuals
                            data = data.sort_values(metric_col, ascending=False).head(15) # Top 15 to avoid overcrowding
                            
                            sns.barplot(data=data, x=metric_col, y=cat_col, palette="viridis", ax=ax)
                            plt.title(f"Top 15 {split_by} by {agg_type} {metric}")
                            ax.bar_label(ax.containers[0], fmt='%.0f', padding=3) # Add labels

                    # C. PIE CHART (Part to Whole)
                    elif "Pie" in chart_type:
                        if not cat_col:
                            st.warning("⚠️ Please select a Category to create a Pie Chart.")
                        else:
                            data = df_filtered.groupby(cat_col)[metric_col].agg("sum").reset_index()
                            data = data.sort_values(metric_col, ascending=False).head(10) # Top 10
                            
                            # Use matplotlib pie directly
                            ax.pie(data[metric_col], labels=data[cat_col], autopct='%1.1f%%', startangle=140)
                            plt.title(f"{metric} Share (Top 10)")

                    # D. BOX PLOT (Distribution/Outliers)
                    elif "Box" in chart_type:
                        if not cat_col:
                            # Single distribution if no category
                            sns.boxplot(y=df_filtered[metric_col], ax=ax)
                        else:
                            # Distribution per category
                            top_cats = df_filtered[cat_col].value_counts().head(10).index
                            sub_data = df_filtered[df_filtered[cat_col].isin(top_cats)]
                            sns.boxplot(data=sub_data, x=cat_col, y=metric_col, ax=ax)
                            plt.xticks(rotation=45)
                        plt.title(f"Distribution of {metric} (Outlier Analysis)")

                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Could not generate plot: {e}")
            else:
                st.info("Select a valid metric to begin.")
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
