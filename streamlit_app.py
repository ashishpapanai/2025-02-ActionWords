import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

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

# Identify 15 most important countries for consistent analysis


def extract_country_code(val):
    if pd.isna(val):
        return None
    val_str = str(val).upper()
    match = re.search(r'\b([A-Z]{2})(?:-[A-Z])?\b', val_str)
    if match:
        return match.group(1)
    match = re.search(r'\.\s*([A-Z]{2})-', val_str)
    if match:
        return match.group(1)
    return val_str[:2] if len(val_str) >= 2 else None


@st.cache_data
def get_selected_countries():
    """Get the 15 selected countries"""
    all_data_for_countries = pd.concat([
        data_2002.assign(year=2002),
        data_2012.assign(year=2012),
        data_2022.assign(year=2022)
    ], ignore_index=True)

    all_data_for_countries['country_code'] = all_data_for_countries['COUNTRY'].apply(
        extract_country_code)

    # Filter to countries with data across all years
    country_counts = all_data_for_countries.groupby('country_code')[
        'year'].nunique()
    valid_countries = country_counts[country_counts == 3].index.tolist()
    all_data_for_countries = all_data_for_countries[all_data_for_countries['country_code'].isin(
        valid_countries)]

    # Calculate country statistics
    country_stats = all_data_for_countries.groupby('country_code').agg({
        'CASEID': 'count',
        'eg_score_norm': lambda x: x.notna().sum() if 'eg_score_norm' in all_data_for_countries.columns else 0,
        'year': 'nunique'
    }).rename(columns={'CASEID': 'total_samples', 'eg_score_norm': 'eg_score_completeness'})

    # Composite score
    country_stats['composite_score'] = (
        country_stats['total_samples'] * 0.6 +
        country_stats['eg_score_completeness'] * 0.4
    )

    # Select top 15 countries
    return country_stats.nlargest(15, 'composite_score').index.tolist()


SELECTED_COUNTRIES = get_selected_countries()

# Helper function to prepare composite data


@st.cache_data
def prepare_composite_data():
    """Prepare composite data for analysis"""
    all_years_data = []

    for year, data in zip([2002, 2012, 2022], [data_2002, data_2012, data_2022]):
        df = data.copy()
        df['year'] = year
        df['sex'] = df['sex'].astype(str).str.strip().str.title()
        df['sex'] = df['sex'].replace(
            {'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'})

        # Clean eg_score_norm
        df['eg_score_norm'] = pd.to_numeric(
            df['eg_score_norm'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        # Simplify urban_rural
        df['urban_rural_simple'] = df['urban_rural'].apply(
            lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
            else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
            else 'Other'
        )

        # Extract country code
        df['country_code'] = df['COUNTRY'].apply(extract_country_code)
        all_years_data.append(df)

    composite_data = pd.concat(all_years_data, ignore_index=True)

    # Filter to countries with data across all years
    country_counts = composite_data.groupby('country_code')['year'].nunique()
    valid_countries = country_counts[country_counts == 3].index.tolist()
    composite_data = composite_data[composite_data['country_code'].isin(
        valid_countries)]

    return composite_data


composite_data = prepare_composite_data()

# Helper function for aggregation (matching notebook)


def aggregate_for_plotting(df, var, is_numerical=True):
    """Aggregate data by year, country, sex, and urban_rural"""
    group_cols = ['year', 'country_code', 'sex', 'urban_rural']

    # Filter valid data
    plot_df = df[df['eg_score_norm'].notna() &
                 df[var].notna() &
                 df['sex'].isin(['Male', 'Female'])].copy()

    if len(plot_df) == 0:
        return None

    # Aggregate
    if is_numerical:
        # Convert to numeric first
        plot_df[f'{var}_numeric'] = pd.to_numeric(
            plot_df[var].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        if plot_df[f'{var}_numeric'].notna().sum() == 0:
            return None

        # Aggregate: mean of numeric, count of rows, mean of eg_score_norm
        agg_df = plot_df.groupby(group_cols).agg({
            f'{var}_numeric': 'mean',
            'eg_score_norm': 'mean'
        }).reset_index()

        # Add count separately
        counts = plot_df.groupby(group_cols).size(
        ).reset_index(name=f'{var}_count')
        agg_df = agg_df.merge(counts, on=group_cols, how='left')

        # Rename columns
        agg_df.columns = group_cols + \
            [f'{var}_mean', 'eg_score_norm_mean', f'{var}_count']
    else:
        # For categorical
        plot_df[f'{var}_numeric'] = pd.to_numeric(
            plot_df[var], errors='coerce')

        agg_dict = {'eg_score_norm': 'mean'}

        if plot_df[f'{var}_numeric'].notna().sum() > 0:
            agg_dict[f'{var}_numeric'] = 'median'

        def safe_mode(x):
            try:
                modes = x.mode()
                return modes[0] if len(modes) > 0 else None
            except:
                return None

        agg_dict[var] = safe_mode

        agg_df = plot_df.groupby(group_cols).agg(agg_dict).reset_index()

        new_cols = group_cols.copy()
        if f'{var}_numeric' in agg_df.columns:
            new_cols.append(f'{var}_median')
        new_cols.append(f'{var}_mode')
        new_cols.append('eg_score_norm_mean')

        agg_df.columns = new_cols
        agg_df[f'{var}_value'] = agg_df[f'{var}_mode']

    # Simplify urban_rural
    agg_df['urban_rural_simple'] = agg_df['urban_rural'].apply(
        lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
        else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
        else 'Other'
    )

    return agg_df


# ISO-2 to ISO-3 mapping for geographical plots
ISO2_TO_ISO3 = {
    'AU': 'AUS', 'AT': 'AUT', 'BE': 'BEL', 'BR': 'BRA', 'BG': 'BGR',
    'CA': 'CAN', 'CL': 'CHL', 'CN': 'CHN', 'HR': 'HRV', 'CZ': 'CZE',
    'DK': 'DNK', 'FI': 'FIN', 'FR': 'FRA', 'DE': 'DEU', 'GR': 'GRC',
    'HU': 'HUN', 'IS': 'ISL', 'IN': 'IND', 'IE': 'IRL', 'IL': 'ISR',
    'IT': 'ITA', 'JP': 'JPN', 'KR': 'KOR', 'LV': 'LVA', 'LT': 'LTU',
    'MX': 'MEX', 'NL': 'NLD', 'NZ': 'NZL', 'NO': 'NOR', 'PH': 'PHL',
    'PL': 'POL', 'PT': 'PRT', 'RO': 'ROU', 'RU': 'RUS', 'SK': 'SVK',
    'SI': 'SVN', 'ZA': 'ZAF', 'ES': 'ESP', 'SE': 'SWE', 'CH': 'CHE',
    'TW': 'TWN', 'TH': 'THA', 'TR': 'TUR', 'UA': 'UKR', 'GB': 'GBR',
    'US': 'USA', 'VE': 'VEN', 'VN': 'VNM', 'AR': 'ARG', 'BY': 'BLR',
    'CY': 'CYP', 'EE': 'EST', 'GT': 'GTM', 'HK': 'HKG', 'ID': 'IDN',
    'MY': 'MYS', 'PE': 'PER', 'SG': 'SGP', 'UY': 'URY'
}


def get_iso3_code(iso2_code):
    """Convert ISO-2 to ISO-3 country code"""
    if pd.isna(iso2_code) or iso2_code is None:
        return None
    iso2_code = str(iso2_code).upper().strip()

    try:
        import pycountry
        country = pycountry.countries.get(alpha_2=iso2_code)
        if country:
            return country.alpha_3
    except:
        pass

    return ISO2_TO_ISO3.get(iso2_code, iso2_code)


# Sidebar filters
st.sidebar.header("🎛️ Filters & Controls")

# Plot type selector
plot_type = st.sidebar.selectbox(
    "📊 Select Plot Type",
    options=[
        "Scatter Plot",
        "Time Series",
        "Geographical Map",
        "Bar Chart",
        "Distribution Plot",
        "Correlation Heatmap",
        "Animated Progressive Plot"
    ],
    index=0
)

# Year selection
selected_years = st.sidebar.multiselect(
    "📅 Select Years",
    options=[2002, 2012, 2022],
    default=[2002, 2012, 2022]
)

# Variable selection
available_vars = ['hh_wrk_hrs', 'SP_HH', 'HOMPOP', 'HH_FAM', 'SP_HH_FAM']
selected_vars = st.sidebar.multiselect(
    "📈 Select Variables",
    options=available_vars,
    default=['hh_wrk_hrs', 'SP_HH', 'HOMPOP']
)

# Gender filter
gender_filter = st.sidebar.selectbox(
    "👥 Gender",
    options=['All', 'Male', 'Female'],
    index=0
)

# Urban/Rural filter
urban_rural_filter = st.sidebar.selectbox(
    "🏙️ Urban/Rural",
    options=['All', 'Urban', 'Rural'],
    index=0
)

# Country filter - make it more prominent and consistent
st.sidebar.markdown("### 🌍 Country Selection")
use_all_countries = st.sidebar.checkbox(
    "Use All Selected Countries (Top 15)",
    value=True,
    help="When checked, uses the top 15 countries. Uncheck to select specific countries."
)

if not use_all_countries:
    selected_countries_filter = st.sidebar.multiselect(
        "Select Specific Countries",
        options=SELECTED_COUNTRIES,
        default=SELECTED_COUNTRIES[:5],  # Default to first 5
        help="Select countries to include in visualizations"
    )
    # Ensure at least one country is selected
    if not selected_countries_filter:
        selected_countries_filter = SELECTED_COUNTRIES[:5]
        st.sidebar.warning(
            "⚠️ No countries selected. Using first 5 countries.")
else:
    selected_countries_filter = SELECTED_COUNTRIES

# Animation toggle (for temporal plots)
show_animation = st.sidebar.checkbox("🎬 Show Animation (Temporal)", value=True)

# Main content
st.title("📊 Household Analysis Dashboard")
st.markdown(
    "**Interactive Tableau-style analysis of household work patterns and egalitarianism scores**")

# Prepare data based on filters


def prepare_filtered_data():
    """Prepare data based on current filters"""
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

        # Extract country code
        df['country_code'] = df['COUNTRY'].apply(extract_country_code)

        # Filter countries - use selected_countries_filter which is now always a list
        df = df[df['country_code'].isin(selected_countries_filter)]

        viz_data.append(df)

    return pd.concat(viz_data, ignore_index=True) if viz_data else pd.DataFrame()


# Main visualization area
if not selected_vars:
    st.warning("⚠️ Please select at least one variable from the sidebar.")
else:
    combined_viz = prepare_filtered_data()

    if len(combined_viz) == 0:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        # Clean numeric columns
        for var in selected_vars + ['eg_score_norm', 'HOMPOP']:
            if var in combined_viz.columns:
                combined_viz[var] = pd.to_numeric(
                    combined_viz[var].astype(str).str.replace(
                        ',', '.', regex=False),
                    errors='coerce'
                )

        combined_viz = combined_viz.dropna(subset=['eg_score_norm'])

        # Plot type routing
        if plot_type == "Scatter Plot":
            st.header("📊 Scatter Plot: Variable vs Egalitarianism Score")

            plot_var = st.selectbox(
                "Select Variable", selected_vars, key="scatter_var")

            if plot_var in combined_viz.columns:
                combined_viz = combined_viz.dropna(subset=[plot_var])

                # Clean HOMPOP for size
                combined_viz['HOMPOP_clean'] = combined_viz['HOMPOP'].fillna(
                    combined_viz['HOMPOP'].median())
                if combined_viz['HOMPOP_clean'].isna().all():
                    combined_viz['HOMPOP_clean'] = 2.0

                valid_hompop = combined_viz['HOMPOP_clean'].notna().sum()
                use_size = valid_hompop > len(combined_viz) * 0.5

                scatter_params = {
                    'x': plot_var,
                    'y': 'eg_score_norm',
                    'color': 'year',
                    'hover_data': ['COUNTRY', 'sex', 'urban_rural_simple', 'country_code'],
                    'title': f"{cols_question_mapping.get(plot_var, plot_var)} vs Normalized Egalitarianism Score",
                    'labels': {
                        plot_var: cols_question_mapping.get(plot_var, plot_var),
                        'eg_score_norm': 'Egalitarianism Score (Normalized)'
                    }
                }

                if use_size:
                    scatter_params['size'] = 'HOMPOP_clean'
                    scatter_params['size_max'] = 20

                fig = px.scatter(combined_viz, **scatter_params)
                st.plotly_chart(fig, width='stretch', use_container_width=True)

                with st.expander("📊 Insights", expanded=False):
                    st.markdown(f"""
                    **Relationship**: {cols_question_mapping.get(plot_var, plot_var)} vs Normalized Egalitarianism Score
                    - **Color**: Year (2002, 2012, 2022)
                    - **Size**: Household size (if available)
                    - **Interpretation**: Negative relationships suggest traditional gender roles
                    """)

        elif plot_type == "Time Series":
            st.header("📈 Time Series: Temporal Trends by Country")

            plot_var = st.selectbox(
                "Select Variable", selected_vars, key="timeseries_var")

            if plot_var in combined_viz.columns:
                combined_viz = combined_viz.dropna(subset=[plot_var])

                # Aggregate by country and year
                country_agg = combined_viz.groupby(['year', 'country_code']).agg({
                    plot_var: 'mean',
                    'eg_score_norm': 'mean'
                }).reset_index()

                country_agg = country_agg[country_agg['country_code'].isin(
                    selected_countries_filter)]

                fig = px.line(
                    country_agg,
                    x='year',
                    y=plot_var,
                    color='country_code',
                    title=f"Time Trends: {cols_question_mapping.get(plot_var, plot_var)} (2002 → 2012 → 2022)",
                    markers=True,
                    labels={'country_code': 'Country', 'year': 'Year'}
                )
                st.plotly_chart(fig, width='stretch', use_container_width=True)

                with st.expander("📈 Insights", expanded=False):
                    st.markdown(f"""
                    **Temporal Patterns**: Shows how {cols_question_mapping.get(plot_var, plot_var)} changes over 20 years
                    - Each line represents a country's trajectory
                    - Upward slopes = increasing values
                    - Downward slopes = decreasing values
                    """)

        elif plot_type == "Geographical Map":
            st.header("🗺️ Geographical Map: Country-Level Patterns")

            plot_var = st.selectbox(
                "Select Variable", selected_vars, key="geo_var")
            year_for_map = st.selectbox("Select Year for Map", [
                                        None] + selected_years, key="geo_year")

            if plot_var in combined_viz.columns:
                # Aggregate data for geographical plot
                agg_df = aggregate_for_plotting(
                    composite_data, plot_var, is_numerical=True)

                if agg_df is not None and len(agg_df) > 0:
                    # Filter by year if specified
                    if year_for_map:
                        agg_df = agg_df[agg_df['year'] == year_for_map]

                    # Aggregate by country (weighted average) - need to recalculate
                    count_col = f'{plot_var}_count'
                    if count_col not in agg_df.columns:
                        # Calculate count if missing
                        counts = composite_data.groupby(
                            ['year', 'country_code', 'sex', 'urban_rural']).size().reset_index(name=count_col)
                        agg_df = agg_df.merge(
                            counts, on=['year', 'country_code', 'sex', 'urban_rural'], how='left')
                        agg_df[count_col] = agg_df[count_col].fillna(1)

                    agg_df['weighted_sum'] = agg_df[f'{plot_var}_mean'] * \
                        agg_df[count_col]
                    agg_df['weighted_sum_eg'] = agg_df['eg_score_norm_mean'] * \
                        agg_df[count_col]

                    # Filter to selected countries
                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

                    # Create choropleth
                    if show_animation and year_for_map is None and 'year' in agg_df.columns:
                        # Animated version - aggregate by country and year
                        geo_agg = agg_df.groupby(['country_code', 'year']).agg({
                            'weighted_sum': 'sum',
                            'weighted_sum_eg': 'sum',
                            count_col: 'sum'
                        }).reset_index()
                        geo_agg[f'{plot_var}_mean'] = geo_agg['weighted_sum'] / \
                            geo_agg[count_col]
                        geo_agg['eg_score_norm_mean'] = geo_agg['weighted_sum_eg'] / \
                            geo_agg[count_col]
                        geo_agg['country_iso3'] = geo_agg['country_code'].apply(
                            get_iso3_code)
                        geo_agg = geo_agg[geo_agg['country_iso3'].notna()]

                        fig = px.choropleth(
                            geo_agg,
                            locations='country_iso3',
                            locationmode='ISO-3',
                            color=f'{plot_var}_mean',
                            animation_frame='year',
                            hover_name='country_code',
                            hover_data={f'{plot_var}_mean': ':.2f',
                                        'eg_score_norm_mean': ':.2f', count_col: ':d'},
                            color_continuous_scale='Viridis',
                            title=f"Geographical Distribution: {cols_question_mapping.get(plot_var, plot_var)} (Animated 2002 → 2012 → 2022)"
                        )
                    else:
                        # Static version - aggregate by country
                        country_agg = agg_df.groupby('country_code').agg({
                            'weighted_sum': 'sum',
                            'weighted_sum_eg': 'sum',
                            count_col: 'sum'
                        }).reset_index()

                        country_agg[f'{plot_var}_mean'] = country_agg['weighted_sum'] / \
                            country_agg[count_col]
                        country_agg['eg_score_norm_mean'] = country_agg['weighted_sum_eg'] / \
                            country_agg[count_col]
                        country_agg['country_iso3'] = country_agg['country_code'].apply(
                            get_iso3_code)
                        country_agg = country_agg[country_agg['country_iso3'].notna(
                        )]

                        fig = px.choropleth(
                            country_agg,
                            locations='country_iso3',
                            locationmode='ISO-3',
                            color=f'{plot_var}_mean',
                            hover_name='country_code',
                            hover_data={f'{plot_var}_mean': ':.2f',
                                        'eg_score_norm_mean': ':.2f', count_col: ':d'},
                            color_continuous_scale='Viridis',
                            title=f"Geographical Distribution: {cols_question_mapping.get(plot_var, plot_var)} ({year_for_map if year_for_map else 'All Years'})"
                        )

                    fig.update_layout(height=600, geo=dict(
                        showframe=False, showcoastlines=True))
                    st.plotly_chart(fig, width='stretch',
                                    use_container_width=True)

                    with st.expander("🗺️ Insights", expanded=False):
                        st.markdown(f"""
                        **Geographical Patterns**: {cols_question_mapping.get(plot_var, plot_var)} by Country
                        - **Color intensity**: Average value (darker = higher)
                        - **Hover**: Shows country, value, EG score, and sample size
                        - **Regional patterns**: Compare countries within regions
                        """)

        elif plot_type == "Bar Chart":
            st.header("📊 Bar Chart: By Gender and Region")

            plot_var = st.selectbox(
                "Select Variable", selected_vars, key="bar_var")
            year_for_bar = st.selectbox(
                "Select Year", [None] + selected_years, key="bar_year")

            if plot_var in combined_viz.columns:
                # Aggregate data
                agg_df = aggregate_for_plotting(
                    composite_data, plot_var, is_numerical=True)

                if agg_df is not None and len(agg_df) > 0:
                    # Filter by year if specified
                    if year_for_bar:
                        agg_df = agg_df[agg_df['year'] == year_for_bar]

                    # Filter to selected countries
                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

                    # Aggregate by country, sex, urban_rural (weighted)
                    count_col = f'{plot_var}_count'
                    if count_col not in agg_df.columns:
                        counts = composite_data.groupby(
                            ['year', 'country_code', 'sex', 'urban_rural']).size().reset_index(name=count_col)
                        agg_df = agg_df.merge(
                            counts, on=['year', 'country_code', 'sex', 'urban_rural'], how='left')
                        agg_df[count_col] = agg_df[count_col].fillna(1)

                    agg_df['weighted_sum'] = agg_df[f'{plot_var}_mean'] * \
                        agg_df[count_col]

                    if show_animation and year_for_bar is None and 'year' in agg_df.columns:
                        # Animated version - aggregate by year too
                        bar_agg_animated = agg_df.groupby(['country_code', 'sex', 'urban_rural_simple', 'year']).agg({
                            'weighted_sum': 'sum',
                            count_col: 'sum'
                        }).reset_index()
                        bar_agg_animated[f'{plot_var}_mean'] = bar_agg_animated['weighted_sum'] / \
                            bar_agg_animated[count_col]

                        fig = px.bar(
                            bar_agg_animated,
                            x='country_code',
                            y=f'{plot_var}_mean',
                            color='sex',
                            pattern_shape='urban_rural_simple',
                            animation_frame='year',
                            barmode='group',
                            title=f"Bar Plot: {cols_question_mapping.get(plot_var, plot_var)} by Country, Gender, and Region (Animated)"
                        )
                    else:
                        # Static version
                        bar_agg = agg_df.groupby(['country_code', 'sex', 'urban_rural_simple']).agg({
                            'weighted_sum': 'sum',
                            count_col: 'sum'
                        }).reset_index()

                        bar_agg[f'{plot_var}_mean'] = bar_agg['weighted_sum'] / \
                            bar_agg[count_col]

                        fig = px.bar(
                            bar_agg,
                            x='country_code',
                            y=f'{plot_var}_mean',
                            color='sex',
                            pattern_shape='urban_rural_simple',
                            barmode='group',
                            title=f"Bar Plot: {cols_question_mapping.get(plot_var, plot_var)} by Country, Gender, and Region ({year_for_bar if year_for_bar else 'All Years'})"
                        )

                    st.plotly_chart(fig, width='stretch',
                                    use_container_width=True)

                    with st.expander("📊 Insights", expanded=False):
                        st.markdown(f"""
                        **Bar Chart**: {cols_question_mapping.get(plot_var, plot_var)}
                        - **Color**: Gender (Male/Female)
                        - **Pattern**: Urban/Rural
                        - **Grouping**: By Country
                        """)

        elif plot_type == "Distribution Plot":
            st.header("📊 Distribution: Egalitarianism Score")

            # Prepare data for distribution
            dist_data = []
            for year in selected_years:
                if year == 2002:
                    df = data_2002.copy()
                elif year == 2012:
                    df = data_2012.copy()
                else:
                    df = data_2022.copy()

                df['year'] = year
                df['sex'] = df['sex'].astype(str).str.strip().str.title()
                df['eg_score_norm'] = pd.to_numeric(
                    df['eg_score_norm'].astype(
                        str).str.replace(',', '.', regex=False),
                    errors='coerce'
                )

                if gender_filter != 'All':
                    df = df[df['sex'] == gender_filter]

                dist_data.append(df)

            dist_combined = pd.concat(dist_data, ignore_index=True)
            dist_combined = dist_combined.dropna(
                subset=['eg_score_norm', 'sex'])

            if len(dist_combined) > 0:
                # Create distribution plot
                fig = px.histogram(
                    dist_combined,
                    x='eg_score_norm',
                    color='year',
                    facet_row='sex',
                    nbins=30,
                    title="Distribution of Normalized Egalitarianism Score by Year and Gender",
                    labels={
                        'eg_score_norm': 'Egalitarianism Score (Normalized)', 'count': 'Frequency'}
                )
                st.plotly_chart(fig, width='stretch', use_container_width=True)

                with st.expander("📊 Insights", expanded=False):
                    st.markdown("""
                    **Distribution Analysis**: Normalized Egalitarianism Score (0-1 scale)
                    - **Histogram**: Shows frequency distribution
                    - **Faceted by**: Gender (rows) and Year (color)
                    - **Interpretation**: Compare distributions across years and genders
                    """)

        elif plot_type == "Correlation Heatmap":
            st.header("🔍 Correlation Heatmap")

            # Calculate correlations
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
                            corr = year_data['eg_score_norm'].corr(
                                year_data[var])
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

                fig = px.imshow(
                    pivot_corr,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu',
                    range_color=[-1, 1],
                    title="Correlation with Normalized Egalitarianism Score"
                )
                st.plotly_chart(fig, width='stretch', use_container_width=True)

                st.dataframe(corr_df)

                with st.expander("🔍 Insights", expanded=False):
                    st.markdown("""
                    **Correlation Heatmap**: Pearson correlation coefficients
                    - **Red**: Positive correlation
                    - **Blue**: Negative correlation
                    - **White**: Weak/no correlation
                    """)
            else:
                st.warning(
                    "No correlations calculated. Check data availability.")

        elif plot_type == "Animated Progressive Plot":
            st.header("🎬 Animated Progressive Plot: Temporal Trajectories")

            if len(selected_vars) > 0:
                # Aggregate all selected variables
                all_agg_data = []

                for var in selected_vars[:3]:  # Limit to 3 for performance
                    agg_df = aggregate_for_plotting(
                        composite_data, var, is_numerical=True)
                    if agg_df is None or len(agg_df) == 0:
                        continue

                    # Filter to selected countries
                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

                    # Weighted average by country, sex (aggregate urban/rural)
                    count_col = f'{var}_count'
                    agg_df['weighted_sum_var'] = agg_df[f'{var}_mean'] * \
                        agg_df[count_col]
                    agg_df['weighted_sum_eg'] = agg_df['eg_score_norm_mean'] * \
                        agg_df[count_col]

                    agg_df_simplified = agg_df.groupby(['year', 'country_code', 'sex']).agg({
                        'weighted_sum_var': 'sum',
                        'weighted_sum_eg': 'sum',
                        count_col: 'sum'
                    }).reset_index()

                    agg_df_simplified[f'{var}_mean'] = agg_df_simplified['weighted_sum_var'] / \
                        agg_df_simplified[count_col]
                    agg_df_simplified['eg_score_norm_mean'] = agg_df_simplified['weighted_sum_eg'] / \
                        agg_df_simplified[count_col]
                    agg_df_simplified = agg_df_simplified.drop(
                        columns=['weighted_sum_var', 'weighted_sum_eg'])

                    agg_df_simplified['plot_value'] = agg_df_simplified[f'{var}_mean']
                    agg_df_simplified['Variable'] = cols_question_mapping.get(
                        var, var)
                    var_short = var if len(var) <= 30 else var[:27] + '...'
                    agg_df_simplified['Variable_Short'] = var_short

                    all_agg_data.append(agg_df_simplified)

                if len(all_agg_data) > 0:
                    combined_df = pd.concat(all_agg_data, ignore_index=True)

                    # Ensure all count columns are present and consistent
                    # Each variable has its own count column, so we need to create a unified one
                    count_cols = [
                        col for col in combined_df.columns if col.endswith('_count')]
                    if count_cols:
                        # Use the first available count column, or sum if multiple exist
                        if len(count_cols) == 1:
                            combined_df['count'] = combined_df[count_cols[0]].fillna(
                                1)
                        else:
                            # If multiple count columns, use the first non-null value
                            combined_df['count'] = combined_df[count_cols[0]].fillna(
                                1)
                            for col in count_cols[1:]:
                                combined_df['count'] = combined_df['count'].fillna(
                                    combined_df[col].fillna(1))
                    else:
                        combined_df['count'] = 1

                    # Create animated scatter plot
                    scatter_params = {
                        'x': 'plot_value',
                        'y': 'eg_score_norm_mean',
                        'color': 'country_code',
                        'animation_frame': 'year',
                        'facet_col': 'Variable_Short',
                        'facet_row': 'sex',
                        'title': 'Composite Analysis: Key Variables vs EG Score Over Time (2002 → 2012 → 2022)',
                        'labels': {
                            'plot_value': 'Average Variable Value',
                            'eg_score_norm_mean': 'Egalitarianism Score Normalized (Average)',
                            'year': 'Year',
                            'country_code': 'Country'
                        },
                        'hover_data': ['year', 'country_code', 'Variable', 'count'],
                        'height': 900,
                        'animation_group': 'country_code'
                    }

                    # Add size using unified count column
                    scatter_params['size'] = 'count'
                    scatter_params['size_max'] = 20

                    fig = px.scatter(combined_df, **scatter_params)

                    st.plotly_chart(fig, width='stretch',
                                    use_container_width=True)

                    with st.expander("🎬 Insights", expanded=False):
                        st.markdown("""
                        **Animated Progressive Plot**: Shows temporal trajectories
                        - **Animation**: Years (2002 → 2012 → 2022)
                        - **Facets**: Variable (columns) × Gender (rows)
                        - **Color**: Country
                        - **Size**: Sample size
                        - **Interpretation**: Watch how countries move through the variable-EG score space over time
                        """)
                else:
                    st.warning("No data available for animated plot.")
            else:
                st.warning("Please select at least one variable.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Active Countries:**")
if use_all_countries:
    st.sidebar.write(
        f"Using all {len(SELECTED_COUNTRIES)} selected countries")
    st.sidebar.caption(f"Countries: {', '.join(SELECTED_COUNTRIES)}")
else:
    st.sidebar.write(
        f"Using {len(selected_countries_filter)} selected countries")
    st.sidebar.caption(f"Countries: {', '.join(selected_countries_filter)}")
