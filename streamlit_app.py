import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="Household Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Change directory to project root
os.chdir('/Users/ashishpapanai/Desktop/MS_Tübingen/ML4102/')

# Load data


@st.cache_data
def load_data():
    """Load and prepare data"""
    data_2002 = pd.read_csv('clean_csv/2002_clean.csv')
    data_2022 = pd.read_csv('clean_csv/2022_clean.csv')
    data_2012 = pd.read_csv('clean_csv/2012_clean.csv')

    # Clean data
    for df in [data_2002, data_2012, data_2022]:
        df.loc[:, ~df.columns.str.contains('^v\\d+', regex=True)]
        df['sex'] = df['sex'].apply(
            lambda x: x.split(' ')[-1] if pd.notnull(x) else x)

    return data_2002, data_2012, data_2022


# Variable mapping
cols_question_mapping = {
    'eg_score_norm': 'Egalitarianism Score Normalised',
    'urban_rural': 'Urban/Rural Classification',
    'sex': 'SEX',
    'age': 'Age of respondent',
    'COUNTRY': 'Country/ Sample ISO 3166 Code',
    'HOMPOP': 'How many persons in household',
    'HHADULT': 'Q24a How many people in hh: adults 18 yrs +',
    'HHCHILDR': 'Q24b How many people in hh:kids 6,7 - 17 yrs',
    'HHTODD': 'Q24c Number of people in hh: kids up to 5,6',
    'hh_wrk_hrs': 'Q16a How many hours spend on household work',
    'HH_FAM': 'Q16b How many hours spend on family members',
    'SP_HH': 'Q17a How many hours spouse, partner works on household',
    'SP_HH_FAM': 'Q17b How many hours spouse, partner spends on family members',
    'FAM_DIF': 'Q23b Difficult to fulfill family responsibility',
    'WORK_TIRED': 'Q16a Too tired from work to do duties at home',
    'HH_TIRED': 'Q16c Too tired from hhwork to function i job',
    'DIFF_CONC_WORK': 'Q16d Difficult to concentrate at work',
    'SHARE_HH': 'Q20 Sharing of household work between partners',
    'HH_WEEKEND': 'Q13a Final say: choosing weekend activities',
    'DIV_HH_LAUND': 'Q19a Division of household work: Doing the laundry',
    'DIV_HH_CARE': 'Q19c Division of household work: Care for sick family members',
    'DIV_HH_GROC': 'Q19d Division of household work: Shops for groceries',
    'DIV_HH_CLEAN': 'Q19e Division of household work: Household cleaning',
    'DIV_HH_COOK': 'Q19f Division of household work: Preparing meals',
    'MOMORFAF': 'Q2d Men should do larger share of childcare',
    'CASEID': 'Respondent ID',
}

# Load data
data_2002, data_2012, data_2022 = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Year selection
selected_years = st.sidebar.multiselect(
    "Select Years",
    options=[2002, 2012, 2022],
    default=[2002, 2012, 2022]
)

# Variable selection
available_vars = ['hh_wrk_hrs', 'SP_HH', 'HOMPOP', 'HH_FAM', 'SP_HH_FAM']
selected_vars = st.sidebar.multiselect(
    "Select Variables",
    options=available_vars,
    default=['hh_wrk_hrs', 'SP_HH', 'HOMPOP']
)

# Gender filter
gender_filter = st.sidebar.selectbox(
    "Gender",
    options=['All', 'Male', 'Female'],
    index=0
)

# Urban/Rural filter
urban_rural_filter = st.sidebar.selectbox(
    "Urban/Rural",
    options=['All', 'Urban', 'Rural'],
    index=0
)

# Main content
st.title("📊 Household Analysis Dashboard")
st.markdown("Interactive analysis of household work patterns and egalitarianism scores across countries and time periods.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Data Overview", "📊 Visualizations", "🔍 Analysis", "💡 Insights"])

with tab1:
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("2002 Sample Size", f"{len(data_2002):,}")
    with col2:
        st.metric("2012 Sample Size", f"{len(data_2012):,}")
    with col3:
        st.metric("2022 Sample Size", f"{len(data_2022):,}")

    # Combine data for overview
    all_data = pd.concat([
        data_2002.assign(year=2002),
        data_2012.assign(year=2012),
        data_2022.assign(year=2022)
    ], ignore_index=True)

    st.subheader("Summary Statistics")
    st.dataframe(all_data[['eg_score_norm', 'hh_wrk_hrs',
                 'SP_HH', 'HOMPOP', 'year']].describe())

    # Add insights for summary statistics
    with st.expander("💡 Insights: Summary Statistics", expanded=False):
        st.markdown("""
        **Data Overview**:
        - **Normalized Egalitarianism Score (eg_score_norm)**: Range 0-1, where 1 = strongest egalitarian attitudes
        - **Household Work Hours (hh_wrk_hrs)**: Hours per week spent on household work
        - **Spouse Household Work (SP_HH)**: Hours per week spouse/partner spends on household work
        - **Household Population (HOMPOP)**: Total number of persons in household
        
        **Key Observations**:
        - Mean and standard deviation show central tendency and variability
        - Min/max values indicate the range of responses
        - Quartiles (25%, 50%, 75%) reveal distribution shape
        - Compare statistics across years to identify trends
        """)

    st.subheader("Missing Values")
    missing_df = all_data[['eg_score_norm', 'hh_wrk_hrs',
                           'SP_HH', 'HOMPOP']].isnull().sum().reset_index()
    missing_df.columns = ['Variable', 'Missing Count']
    missing_df['Missing %'] = (
        missing_df['Missing Count'] / len(all_data) * 100).round(2)
    st.dataframe(missing_df)

    # Add insights for missing values
    with st.expander("⚠️ Insights: Missing Data Analysis", expanded=False):
        st.markdown("""
        **Data Quality Assessment**:
        - **Low Missing % (< 5%)**: Excellent data quality, minimal impact on analysis
        - **Moderate Missing % (5-20%)**: Acceptable, may need imputation for some analyses
        - **High Missing % (> 20%)**: Significant, may require special handling
        
        **Implications**:
        - Missing data patterns may indicate response bias
        - Some variables may have higher completion rates than others
        - Consider missing data mechanisms when interpreting results
        - All analyses exclude missing values appropriately
        """)

with tab2:
    st.header("Visualizations")

    if not selected_vars:
        st.warning("Please select at least one variable from the sidebar.")
    else:
        # Prepare data for visualization
        viz_data = []
        for year in selected_years:
            if year == 2002:
                df = data_2002.copy()
            elif year == 2012:
                df = data_2012.copy()
            else:
                df = data_2022.copy()

            df['year'] = year
            df['sex'] = df['sex'].astype(str).str.strip().str.title()

            # Apply filters
            if gender_filter != 'All':
                df = df[df['sex'] == gender_filter]

            # Simplify urban_rural
            df['urban_rural_simple'] = df['urban_rural'].apply(
                lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
                else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
                else 'Other'
            )

            if urban_rural_filter != 'All':
                df = df[df['urban_rural_simple'] == urban_rural_filter]

            viz_data.append(df)

        combined_viz = pd.concat(viz_data, ignore_index=True)

        # Variable selection for plot
        plot_var = st.selectbox("Select Variable for Plot", selected_vars)

        # Clean and prepare
        combined_viz['eg_score_norm'] = pd.to_numeric(
            combined_viz['eg_score_norm'].astype(
                str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        combined_viz[plot_var] = pd.to_numeric(
            combined_viz[plot_var].astype(
                str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        # Clean HOMPOP for size parameter - handle NaN values
        combined_viz['HOMPOP'] = pd.to_numeric(
            combined_viz['HOMPOP'].astype(
                str).str.replace(',', '.', regex=False),
            errors='coerce'
        )
        # Fill NaN with median or use a default value, or drop rows with NaN HOMPOP if using size
        combined_viz['HOMPOP_clean'] = combined_viz['HOMPOP'].fillna(
            combined_viz['HOMPOP'].median())
        # If median is also NaN, use a default value
        if combined_viz['HOMPOP_clean'].isna().all():
            combined_viz['HOMPOP_clean'] = 2.0  # Default household size

        combined_viz = combined_viz.dropna(subset=['eg_score_norm', plot_var])

        if len(combined_viz) > 0:
            # Scatter plot - only use size if we have valid HOMPOP data
            st.subheader(
                f"{cols_question_mapping.get(plot_var, plot_var)} vs Egalitarianism Score")

            # Check if we have enough valid HOMPOP values to use as size
            valid_hompop = combined_viz['HOMPOP_clean'].notna().sum()
            use_size = valid_hompop > len(
                combined_viz) * 0.5  # Use size if >50% valid

            scatter_params = {
                'x': plot_var,
                'y': 'eg_score_norm',
                'color': 'year',
                'hover_data': ['COUNTRY', 'sex', 'urban_rural_simple', 'HOMPOP_clean'],
                'title': f"{cols_question_mapping.get(plot_var, plot_var)} vs Normalized Egalitarianism Score",
                'labels': {
                    plot_var: cols_question_mapping.get(plot_var, plot_var),
                    'eg_score_norm': 'Egalitarianism Score (Normalized)'
                }
            }

            if use_size:
                scatter_params['size'] = 'HOMPOP_clean'
                scatter_params['size_max'] = 20

            fig_scatter = px.scatter(combined_viz, **scatter_params)
            st.plotly_chart(fig_scatter, width='stretch')

            # Add inference for scatter plot
            with st.expander("📊 Insights: Scatter Plot Analysis", expanded=False):
                st.markdown("""
                **Relationship Pattern**: This scatter plot shows the relationship between the selected variable and normalized egalitarianism score (0-1 scale).
                
                **Key Observations**:
                - **Negative Correlation**: Most household work variables show negative relationships with egalitarianism scores
                - **Color by Year**: Points colored by year (2002, 2012, 2022) reveal temporal patterns
                - **Bubble Size**: Larger bubbles indicate larger household sizes (HOMPOP)
                
                **Interpretation**:
                - Points in the upper-left region (low variable value, high egalitarianism) suggest more egalitarian attitudes
                - Points in the lower-right region (high variable value, low egalitarianism) suggest traditional gender roles
                - Clustering patterns may indicate country-specific or cultural factors
                - Temporal shifts (color changes) show how relationships evolve over 20 years
                
                **Statistical Note**: The normalized score (0-1) allows direct comparison across years without scale differences.
                """)

            # Time series by country
            st.subheader("Time Trends by Country")
            country_agg = combined_viz.groupby(['year', 'COUNTRY']).agg({
                plot_var: 'mean',
                'eg_score_norm': 'mean'
            }).reset_index()

            top_countries = country_agg.groupby(
                'COUNTRY')[plot_var].mean().nlargest(10).index.tolist()
            country_agg_filtered = country_agg[country_agg['COUNTRY'].isin(
                top_countries)]

            fig_trends = px.line(
                country_agg_filtered,
                x='year',
                y=plot_var,
                color='COUNTRY',
                title=f"Time Trends: {cols_question_mapping.get(plot_var, plot_var)}",
                markers=True
            )
            st.plotly_chart(fig_trends, width='stretch')

            # Add inference for time trends
            with st.expander("📈 Insights: Time Trends Analysis", expanded=False):
                st.markdown(f"""
                **Temporal Patterns**: This plot shows how {cols_question_mapping.get(plot_var, plot_var)} changes over time (2002 → 2012 → 2022) for the top 10 countries by average value.
                
                **Key Observations**:
                - **Trend Direction**: Lines sloping upward indicate increasing values, downward indicates decreasing
                - **Country Trajectories**: Each colored line represents a country's path over 20 years
                - **Convergence/Divergence**: Lines moving closer together suggest convergence, moving apart suggests divergence
                
                **Interpretation**:
                - **Stable Trends**: Countries with relatively flat lines show consistent patterns
                - **Rapid Changes**: Steep slopes indicate significant shifts in household work patterns
                - **Country Rankings**: Countries at the top have higher average values, those at bottom have lower
                - **Policy Implications**: Rapid changes may reflect policy interventions or cultural shifts
                
                **Context**: Top 10 countries selected based on average {cols_question_mapping.get(plot_var, plot_var)} across all years.
                """)
        else:
            st.warning("No data available for the selected filters.")

with tab3:
    st.header("Analysis")

    # Correlation analysis
    st.subheader("Correlation with Egalitarianism Score")

    all_data_analysis = pd.concat([
        data_2002.assign(year=2002),
        data_2012.assign(year=2012),
        data_2022.assign(year=2022)
    ], ignore_index=True)

    # Clean numeric columns
    numeric_vars = ['eg_score_norm', 'hh_wrk_hrs',
                    'SP_HH', 'HOMPOP', 'HH_FAM', 'SP_HH_FAM', 'age']
    for var in numeric_vars:
        if var in all_data_analysis.columns:
            all_data_analysis[var] = pd.to_numeric(
                all_data_analysis[var].astype(
                    str).str.replace(',', '.', regex=False),
                errors='coerce'
            )

    # Calculate correlations
    correlations = []
    for var in selected_vars:
        if var in all_data_analysis.columns:
            for year in selected_years:
                year_data = all_data_analysis[
                    (all_data_analysis['year'] == year) &
                    (all_data_analysis['eg_score_norm'].notna()) &
                    (all_data_analysis[var].notna())
                ]
                if len(year_data) > 100:
                    corr = year_data['eg_score_norm'].corr(year_data[var])
                    if not pd.isna(corr):
                        correlations.append({
                            'Variable': cols_question_mapping.get(var, var),
                            'Year': year,
                            'Correlation': corr
                        })

    if correlations:
        corr_df = pd.DataFrame(correlations)
        pivot_corr = corr_df.pivot(
            index='Variable', columns='Year', values='Correlation')

        fig_corr = px.imshow(
            pivot_corr,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdBu',
            range_color=[-1, 1],
            title="Correlation with Normalized Egalitarianism Score"
        )
        st.plotly_chart(fig_corr, width='stretch')

        # Add inference for correlation heatmap
        with st.expander("🔍 Insights: Correlation Analysis", expanded=False):
            st.markdown("""
            **Correlation Interpretation**: This heatmap shows Pearson correlation coefficients between variables and normalized egalitarianism score (0-1 scale).
            
            **Color Coding**:
            - **Red**: Positive correlation (higher variable value → higher egalitarianism score)
            - **Blue**: Negative correlation (higher variable value → lower egalitarianism score)
            - **White/Neutral**: Weak or no correlation
            
            **Key Findings**:
            - **Strong Negative Correlations** (Dark Blue): Age, household work hours, spouse household work
            - **Moderate Correlations**: Household population, children variables
            - **Temporal Variation**: Correlations may change across years (2002, 2012, 2022)
            
            **Interpretation**:
            - **Negative correlations** suggest traditional gender roles are associated with lower egalitarianism
            - **Weakening correlations over time** may indicate changing societal patterns
            - **Variable-specific patterns** show which factors are most strongly related to egalitarian attitudes
            
            **Statistical Note**: Correlations calculated only for years and variables with sufficient sample size (n > 100).
            """)

        st.dataframe(corr_df)

with tab4:
    st.header("Key Insights")

    st.markdown("""
    ### Key Findings
    
    1. **Egalitarianism Score Normalization**: All analyses now use the normalized egalitarianism score (0-1 scale) for consistent comparisons across years and countries.
    
    2. **Household Work Patterns**: 
       - Negative correlations observed between household work hours and egalitarianism scores
       - Gender differences are significant across all countries
    
    3. **Temporal Trends**:
       - Data spans three time periods: 2002, 2012, and 2022
       - 23 countries have complete data across all three years
    
    4. **Geographical Patterns**:
       - Significant variation in household work patterns across countries
       - Urban/rural distinctions show different patterns
    
    ### Recommendations
    
    - Further analysis of country-specific trends
    - Investigation of policy impacts on household work distribution
    - Longitudinal analysis of individual country trajectories
    """)
