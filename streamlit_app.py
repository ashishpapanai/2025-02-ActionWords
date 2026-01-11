import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

st.set_page_config(
    page_title="Household Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.chdir('/Users/ashishpapanai/Desktop/MS_Tübingen/ML4102/')


@st.cache_data
def load_data():
    """Load and prepare data"""
    data_2002 = pd.read_csv('clean_csv/2002_clean.csv')
    data_2022 = pd.read_csv('clean_csv/2022_clean.csv')
    data_2012 = pd.read_csv('clean_csv/2012_clean.csv')

    for df in [data_2002, data_2012, data_2022]:
        df.loc[:, ~df.columns.str.contains('^v\\d+', regex=True)]
        df['sex'] = df['sex'].apply(
            lambda x: x.split(' ')[-1] if pd.notnull(x) else x)

    return data_2002, data_2012, data_2022


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

data_2002, data_2012, data_2022 = load_data()


def extract_country_code(val):
    if pd.isna(val):
        return None
    val_str = str(val).upper().strip()
    match = re.search(r'\b([A-Z]{2})(?:-[A-Z])?\b', val_str)
    if match:
        return match.group(1)
    match = re.search(r'\.\s*([A-Z]{2})-', val_str)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z]{2})\s', val_str)
    if match:
        return match.group(1)
    if len(val_str) >= 2 and val_str[:2].isalpha():
        return val_str[:2]
    return None


CONTINENT_MAPPING = {
    'ZA': 'Africa',
    'CN': 'Asia', 'IN': 'Asia', 'JP': 'Asia', 'KR': 'Asia',
    'MY': 'Asia', 'PH': 'Asia', 'SG': 'Asia', 'TH': 'Asia',
    'TW': 'Asia', 'VN': 'Asia', 'HK': 'Asia', 'IL': 'Asia',
    'AT': 'Europe', 'BE': 'Europe', 'BG': 'Europe', 'CZ': 'Europe',
    'DK': 'Europe', 'EE': 'Europe', 'FI': 'Europe', 'FR': 'Europe',
    'DE': 'Europe', 'GR': 'Europe', 'HU': 'Europe', 'IS': 'Europe',
    'IE': 'Europe', 'IT': 'Europe', 'LV': 'Europe', 'LT': 'Europe',
    'LU': 'Europe', 'NL': 'Europe', 'NO': 'Europe', 'PL': 'Europe',
    'PT': 'Europe', 'RO': 'Europe', 'RU': 'Europe', 'SK': 'Europe',
    'SI': 'Europe', 'ES': 'Europe', 'SE': 'Europe', 'CH': 'Europe',
    'TR': 'Europe', 'UA': 'Europe', 'GB': 'Europe', 'CY': 'Europe',
    'BR': 'Americas', 'CA': 'Americas', 'CL': 'Americas', 'MX': 'Americas',
    'US': 'Americas', 'UY': 'Americas',
    'AU': 'Oceania', 'NZ': 'Oceania'
}


def select_countries_by_continent(country_stats, n_countries=15):
    """
    Select countries proportionally from all continents.
    Ensures at least one country from each continent if data is available.
    """
    valid_countries = country_stats[country_stats['continent'] != 'Unknown'].copy(
    )

    if len(valid_countries) == 0:
        return country_stats.nlargest(n_countries, 'composite_score').index.tolist()

    continent_groups = {}
    for continent in valid_countries['continent'].unique():
        continent_data = valid_countries[valid_countries['continent'] == continent].sort_values(
            'composite_score', ascending=False
        )
        continent_groups[continent] = continent_data

    total_valid_countries = len(valid_countries)
    selected = []

    for continent, data in continent_groups.items():
        if len(data) > 0:
            top_country = data.index[0]
            selected.append(top_country)

    remaining_slots = n_countries - len(selected)

    if remaining_slots > 0:
        continent_ratios = {}
        for continent, data in continent_groups.items():
            continent_ratios[continent] = len(data) / total_valid_countries

        continent_allocations = {}
        allocated = 0
        for continent, ratio in sorted(continent_ratios.items(), key=lambda x: x[1], reverse=True):
            slots = max(1, round(ratio * remaining_slots))
            continent_allocations[continent] = slots
            allocated += slots

        if allocated > remaining_slots:
            diff = allocated - remaining_slots
            for continent in sorted(continent_allocations.keys(), key=lambda x: continent_allocations[x], reverse=True):
                if continent_allocations[continent] > 1 and diff > 0:
                    continent_allocations[continent] -= 1
                    diff -= 1
                    if diff == 0:
                        break

        for continent, slots in continent_allocations.items():
            data = continent_groups[continent]
            available = data[~data.index.isin(selected)]
            if len(available) > 0:
                n_to_select = min(slots, len(available))
                selected.extend(available.head(n_to_select).index.tolist())
    selected_with_scores = [
        (code, country_stats.loc[code, 'composite_score']) for code in selected]
    selected_with_scores.sort(key=lambda x: x[1], reverse=True)

    return [code for code, _ in selected_with_scores]


@st.cache_data
def get_selected_countries():
    """Get the 15 selected countries with proportional continent representation"""
    all_data_for_countries = pd.concat([
        data_2002.assign(year=2002),
        data_2012.assign(year=2012),
        data_2022.assign(year=2022)
    ], ignore_index=True)

    all_data_for_countries['country_code'] = all_data_for_countries['COUNTRY'].apply(
        extract_country_code)

    country_counts = all_data_for_countries.groupby('country_code')[
        'year'].nunique()
    valid_countries = country_counts[country_counts == 3].index.tolist()
    all_data_for_countries = all_data_for_countries[all_data_for_countries['country_code'].isin(
        valid_countries)]

    country_stats = all_data_for_countries.groupby('country_code').agg({
        'CASEID': 'count',
        'eg_score_norm': lambda x: x.notna().sum() if 'eg_score_norm' in all_data_for_countries.columns else 0,
        'year': 'nunique'
    }).rename(columns={'CASEID': 'total_samples', 'eg_score_norm': 'eg_score_completeness'})

    country_stats['composite_score'] = (
        country_stats['total_samples'] * 0.6 +
        country_stats['eg_score_completeness'] * 0.4
    )

    country_stats['continent'] = country_stats.index.map(
        CONTINENT_MAPPING).fillna('Unknown')

    return select_countries_by_continent(country_stats, n_countries=15)


SELECTED_COUNTRIES = get_selected_countries()


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

        df['eg_score_norm'] = pd.to_numeric(
            df['eg_score_norm'].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        df['urban_rural_simple'] = df['urban_rural'].apply(
            lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
            else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
            else 'Other'
        )

        df['country_code'] = df['COUNTRY'].apply(extract_country_code)
        all_years_data.append(df)

    composite_data = pd.concat(all_years_data, ignore_index=True)

    country_counts = composite_data.groupby('country_code')['year'].nunique()
    valid_countries = country_counts[country_counts == 3].index.tolist()
    composite_data = composite_data[composite_data['country_code'].isin(
        valid_countries)]

    return composite_data


composite_data = prepare_composite_data()
ALL_AVAILABLE_COUNTRIES = sorted(
    composite_data['country_code'].unique().tolist())


def aggregate_for_plotting(df, var, is_numerical=True):
    """Aggregate data by year, country, sex, and urban_rural"""
    group_cols = ['year', 'country_code', 'sex', 'urban_rural']

    plot_df = df[df['eg_score_norm'].notna() &
                 df[var].notna() &
                 df['sex'].isin(['Male', 'Female'])].copy()

    if len(plot_df) == 0:
        return None

    if is_numerical:
        plot_df[f'{var}_numeric'] = pd.to_numeric(
            plot_df[var].astype(str).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        if plot_df[f'{var}_numeric'].notna().sum() == 0:
            return None

        agg_df = plot_df.groupby(group_cols).agg({
            f'{var}_numeric': 'mean',
            'eg_score_norm': 'mean'
        }).reset_index()

        counts = plot_df.groupby(group_cols).size(
        ).reset_index(name=f'{var}_count')
        agg_df = agg_df.merge(counts, on=group_cols, how='left')

        agg_df.columns = group_cols + \
            [f'{var}_mean', 'eg_score_norm_mean', f'{var}_count']
    else:
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

    agg_df['urban_rural_simple'] = agg_df['urban_rural'].apply(
        lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
        else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
        else 'Other'
    )

    return agg_df


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


st.sidebar.header("🎛️ Filters & Controls")

plot_type = st.sidebar.selectbox(
    "📊 Select Plot Type",
    options=[
        "Scatter Plot",
        "Time Series",
        "Geographical Map",
        "Bar Chart",
        "Distribution Plot",
        "Correlation Heatmap",
        "Animated Progressive Plot",
        "Composite Dashboard"
    ],
    index=0
)

selected_years = st.sidebar.multiselect(
    "📅 Select Years",
    options=[2002, 2012, 2022],
    default=[2002, 2012, 2022]
)

excluded_vars = ['CASEID', 'COUNTRY', 'sex', 'urban_rural']
available_vars = [var for var in cols_question_mapping.keys()
                  if var not in excluded_vars]
selected_vars = st.sidebar.multiselect(
    "📈 Select Variables",
    options=available_vars,
    default=['hh_wrk_hrs', 'SP_HH', 'HOMPOP']
)

gender_filter = st.sidebar.multiselect(
    "👥 Gender",
    options=['Male', 'Female'],
    default=['Male', 'Female'],
    help="Select one or more genders to include in the analysis"
)

urban_rural_filter = st.sidebar.multiselect(
    "🏙️ Urban/Rural",
    options=['Urban', 'Rural'],
    default=['Urban', 'Rural'],
    help="Select one or more location types to include in the analysis"
)

st.sidebar.markdown("### 🌍 Country Selection")
st.sidebar.caption(f"Available: {len(ALL_AVAILABLE_COUNTRIES)} countries")
selected_countries_filter = st.sidebar.multiselect(
    "Select Countries",
    options=ALL_AVAILABLE_COUNTRIES,
    default=ALL_AVAILABLE_COUNTRIES,
    help="Select countries to include in visualizations. You can select multiple countries or all of them."
)

if not selected_countries_filter:
    selected_countries_filter = ALL_AVAILABLE_COUNTRIES
    st.sidebar.info("ℹ️ No countries selected. Using all available countries.")

show_animation = st.sidebar.checkbox("🎬 Show Animation (Temporal)", value=True)

st.title("📊 Household Analysis Dashboard")
st.markdown(
    "**Interactive Tableau-style analysis of household work patterns and egalitarianism scores**")


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

        if len(gender_filter) > 0:
            df = df[df['sex'].isin(gender_filter)]
        else:
            df = df[df['sex'].isin(['Male', 'Female'])]

        df['urban_rural_simple'] = df['urban_rural'].apply(
            lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
            else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
            else 'Other'
        )

        if len(urban_rural_filter) > 0:
            df = df[df['urban_rural_simple'].isin(urban_rural_filter)]
        else:
            df = df[df['urban_rural_simple'].isin(['Urban', 'Rural'])]

        df['country_code'] = df['COUNTRY'].apply(extract_country_code)

        df = df[df['country_code'].isin(selected_countries_filter)]

        viz_data.append(df)

    return pd.concat(viz_data, ignore_index=True) if viz_data else pd.DataFrame()


if not selected_vars:
    st.warning("⚠️ Please select at least one variable from the sidebar.")
else:
    combined_viz = prepare_filtered_data()

    if len(combined_viz) == 0:
        st.warning("⚠️ No data available for the selected filters.")
    else:
        for var in selected_vars + ['eg_score_norm', 'HOMPOP']:
            if var in combined_viz.columns:
                combined_viz[var] = pd.to_numeric(
                    combined_viz[var].astype(str).str.replace(
                        ',', '.', regex=False),
                    errors='coerce'
                )

        combined_viz = combined_viz.dropna(subset=['eg_score_norm'])

        if plot_type == "Scatter Plot":
            st.header("📊 Scatter Plot: Variable vs Egalitarianism Score")

            plot_var = st.selectbox(
                "Select Variable", selected_vars, key="scatter_var")

            if plot_var in combined_viz.columns:
                combined_viz = combined_viz.dropna(subset=[plot_var])

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
                fig.update_layout(height=600, width=900)
                st.plotly_chart(fig, use_container_width=False)

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
                fig.update_layout(height=500, width=1000)
                st.plotly_chart(fig, use_container_width=False)

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
                agg_df = aggregate_for_plotting(
                    composite_data, plot_var, is_numerical=True)

                if agg_df is not None and len(agg_df) > 0:
                    if year_for_map:
                        agg_df = agg_df[agg_df['year'] == year_for_map]

                    count_col = f'{plot_var}_count'
                    if count_col not in agg_df.columns:
                        counts = composite_data.groupby(
                            ['year', 'country_code', 'sex', 'urban_rural']).size().reset_index(name=count_col)
                        agg_df = agg_df.merge(
                            counts, on=['year', 'country_code', 'sex', 'urban_rural'], how='left')
                        agg_df[count_col] = agg_df[count_col].fillna(1)

                    agg_df['weighted_sum'] = agg_df[f'{plot_var}_mean'] * \
                        agg_df[count_col]
                    agg_df['weighted_sum_eg'] = agg_df['eg_score_norm_mean'] * \
                        agg_df[count_col]

                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

                    if show_animation and year_for_map is None and 'year' in agg_df.columns:
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

                    fig.update_layout(height=600, width=1100, geo=dict(
                        showframe=False, showcoastlines=True))
                    st.plotly_chart(fig, use_container_width=False)

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
                agg_df = aggregate_for_plotting(
                    composite_data, plot_var, is_numerical=True)

                if agg_df is not None and len(agg_df) > 0:
                    if year_for_bar:
                        agg_df = agg_df[agg_df['year'] == year_for_bar]

                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

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

                    fig.update_layout(height=500, width=1000)
                    st.plotly_chart(fig, use_container_width=False)

                    with st.expander("📊 Insights", expanded=False):
                        st.markdown(f"""
                        **Bar Chart**: {cols_question_mapping.get(plot_var, plot_var)}
                        - **Color**: Gender (Male/Female)
                        - **Pattern**: Urban/Rural
                        - **Grouping**: By Country
                        """)

        elif plot_type == "Distribution Plot":
            st.header("📊 Distribution: Egalitarianism Score")

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

                if len(gender_filter) > 0:
                    df = df[df['sex'].isin(gender_filter)]
                else:
                    df = df[df['sex'].isin(['Male', 'Female'])]

                dist_data.append(df)

            dist_combined = pd.concat(dist_data, ignore_index=True)
            dist_combined = dist_combined.dropna(
                subset=['eg_score_norm', 'sex'])

            if len(dist_combined) > 0:
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
                fig.update_layout(height=600, width=900)
                st.plotly_chart(fig, use_container_width=False)

                with st.expander("📊 Insights", expanded=False):
                    st.markdown("""
                    **Distribution Analysis**: Normalized Egalitarianism Score (0-1 scale)
                    - **Histogram**: Shows frequency distribution
                    - **Faceted by**: Gender (rows) and Year (color)
                    - **Interpretation**: Compare distributions across years and genders
                    """)

        elif plot_type == "Correlation Heatmap":
            st.header("🔍 Correlation Heatmap")

            all_data_analysis = pd.concat([
                data_2002.assign(year=2002),
                data_2012.assign(year=2012),
                data_2022.assign(year=2022)
            ], ignore_index=True)

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
                fig.update_layout(height=400, width=800)
                st.plotly_chart(fig, use_container_width=False)

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
                all_agg_data = []

                for var in selected_vars[:3]:
                    agg_df = aggregate_for_plotting(
                        composite_data, var, is_numerical=True)
                    if agg_df is None or len(agg_df) == 0:
                        continue

                    agg_df = agg_df[agg_df['country_code'].isin(
                        selected_countries_filter)]

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

                    count_cols = [
                        col for col in combined_df.columns if col.endswith('_count')]
                    if count_cols:
                        if len(count_cols) == 1:
                            combined_df['count'] = combined_df[count_cols[0]].fillna(
                                1)
                        else:
                            combined_df['count'] = combined_df[count_cols[0]].fillna(
                                1)
                            for col in count_cols[1:]:
                                combined_df['count'] = combined_df['count'].fillna(
                                    combined_df[col].fillna(1))
                    else:
                        combined_df['count'] = 1

                    combined_df['year_symbol'] = combined_df['year'].map(
                        {2002: 'circle', 2012: 'x', 2022: 'cross'})

                    scatter_params = {
                        'x': 'plot_value',
                        'y': 'eg_score_norm_mean',
                        'color': 'country_code',
                        'symbol': 'year_symbol',
                        'animation_frame': 'year',
                        'facet_col': 'Variable_Short',
                        'facet_row': 'sex',
                        'title': 'Composite Analysis: Key Variables vs EG Score Over Time (2002 → 2012 → 2022)',
                        'labels': {
                            'plot_value': 'Average Variable Value',
                            'eg_score_norm_mean': 'Egalitarianism Score Normalized (Average)',
                            'year': 'Year',
                            'country_code': 'Country',
                            'year_symbol': 'Year'
                        },
                        'hover_data': ['year', 'country_code', 'Variable', 'count'],
                        'height': 900,
                        'animation_group': 'country_code'
                    }

                    fig = px.scatter(combined_df, **scatter_params)

                    fig.update_traces(marker=dict(size=8))

                    fig.update_layout(height=900, width=1200)

                    st.plotly_chart(fig, use_container_width=False)

                    with st.expander("🎬 Insights", expanded=False):
                        st.markdown("""
                        **Animated Progressive Plot**: Shows temporal trajectories
                        - **Animation**: Years (2002 → 2012 → 2022)
                        - **Facets**: Variable (columns) × Gender (rows)
                        - **Color**: Country
                        - **Symbol**: Year (circle=2002, x=2012, cross=2022)
                        - **Size**: Constant for readability
                        - **Interpretation**: Watch how countries move through the variable-EG score space over time
                        """)
                else:
                    st.warning("No data available for animated plot.")
            else:
                st.warning("Please select at least one variable.")

        elif plot_type == "Composite Dashboard":
            st.header("📊 Composite Dashboard: Multi-Variable Analysis")

            dashboard_vars = st.multiselect(
                "Select Variables (up to 3)",
                options=available_vars,
                default=selected_vars[:3] if len(
                    selected_vars) >= 3 else selected_vars,
                key="dashboard_vars",
                help="Select up to 3 variables to display in the composite dashboard"
            )

            all_available_countries = sorted(
                composite_data['country_code'].unique().tolist())
            default_dashboard_countries = selected_countries_filter if all(c in all_available_countries for c in selected_countries_filter) else (
                all_available_countries[:15] if len(all_available_countries) >= 15 else all_available_countries)

            dashboard_countries = st.multiselect(
                "Select Countries for Dashboard",
                options=all_available_countries,
                default=default_dashboard_countries,
                key="dashboard_countries",
                help="Select countries to include in the composite dashboard. You can choose from all available countries."
            )

            if len(dashboard_countries) == 0:
                st.warning(
                    "⚠️ No countries selected. Using default countries.")
                dashboard_countries = default_dashboard_countries

            if len(dashboard_vars) == 0:
                st.warning(
                    "⚠️ Please select at least one variable for the composite dashboard.")
            elif len(dashboard_vars) > 3:
                st.warning(
                    "⚠️ Please select a maximum of 3 variables. Showing first 3.")
                dashboard_vars = dashboard_vars[:3]
            else:
                summary_data = []

                for var in dashboard_vars:
                    question = cols_question_mapping.get(var, var)
                    agg_df = aggregate_for_plotting(
                        composite_data, var, is_numerical=True)

                    if agg_df is None or len(agg_df) == 0:
                        continue

                    agg_df = agg_df[agg_df['country_code'].isin(
                        dashboard_countries)]

                    required_cols = [
                        'year', 'sex', 'urban_rural_simple', f'{var}_mean', 'eg_score_norm_mean']
                    missing_cols = [
                        col for col in required_cols if col not in agg_df.columns]
                    if missing_cols:
                        continue

                    grouped = agg_df.groupby(
                        ['year', 'sex', 'urban_rural_simple'])

                    var_mean_agg = grouped[f'{var}_mean'].mean().reset_index()
                    eg_score_agg = grouped['eg_score_norm_mean'].mean(
                    ).reset_index()

                    summary_agg = var_mean_agg.merge(
                        eg_score_agg[[
                            'year', 'sex', 'urban_rural_simple', 'eg_score_norm_mean']],
                        on=['year', 'sex', 'urban_rural_simple'],
                        how='inner'
                    )

                    summary_agg['Variable'] = question
                    summary_agg['Variable_Code'] = var
                    summary_agg['Value'] = summary_agg[f'{var}_mean']

                    summary_data.append(summary_agg[[
                                        'year', 'sex', 'urban_rural_simple', 'Variable', 'Variable_Code', 'Value', 'eg_score_norm_mean']])

                if len(summary_data) > 0:
                    summary_df = pd.concat(summary_data, ignore_index=True)

                    # Get unique variables (up to 3)
                    unique_vars = summary_df['Variable'].unique()[
                        :len(dashboard_vars)]
                    n_vars = len(unique_vars)
                    n_rows = n_vars
                    n_cols = 2

                    def format_title(text, suffix=""):
                        """Format title with line break if more than 6 words"""
                        full_text = f'{text}{suffix}'
                        words = full_text.split()
                        if len(words) > 6:
                            mid = len(words) // 2
                            return '<br>'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                        return full_text

                    subplot_titles = []
                    for var in unique_vars:
                        subplot_titles.extend([
                            format_title(var, " - Time Trend"),
                            format_title(var, " - vs EG Score")
                        ])

                    fig = make_subplots(
                        rows=n_rows, cols=n_cols,
                        subplot_titles=subplot_titles[:n_rows*n_cols],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.10,
                        column_titles=('Time Trends (2002-2022)',
                                       'Relationship with EG Score'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]
                               for _ in range(n_rows)]
                    )

                    color_map = {
                        'Male_Urban': '#2E86AB',
                        'Female_Urban': '#A23B72',
                        'Male_Rural': '#F18F01',
                        'Female_Rural': '#C73E1D'
                    }

                    legend_groups_used = set()

                    for var_idx, var_name in enumerate(unique_vars):
                        var_data = summary_df[summary_df['Variable'] == var_name].copy(
                        )
                        var_data = var_data.sort_values('year')

                        row = var_idx + 1

                        col = 1
                        for sex in ['Male', 'Female']:
                            for urb in ['Urban', 'Rural']:
                                subset = var_data[(var_data['sex'] == sex) &
                                                  (var_data['urban_rural_simple'] == urb)].sort_values('year')
                                if len(subset) > 0:
                                    gender_urban_key = f'{sex}_{urb}'
                                    color = color_map.get(
                                        gender_urban_key, '#808080')

                                    show_legend = gender_urban_key not in legend_groups_used
                                    if show_legend:
                                        legend_groups_used.add(
                                            gender_urban_key)

                                    fig.add_trace(
                                        go.Scatter(
                                            x=subset['year'],
                                            y=subset['Value'],
                                            mode='lines+markers',
                                            name=f'{sex} - {urb}',
                                            marker=dict(color=color, size=7, opacity=0.8,
                                                        line=dict(width=1, color='white')),
                                            line=dict(
                                                color=color, width=2.5, shape='spline'),
                                            legendgroup=gender_urban_key,
                                            showlegend=show_legend,
                                            hovertemplate=f'<b>{var_name}</b><br>Year: %{{x}}<br>Value: %{{y:.2f}}<br>{sex} - {urb}<extra></extra>'
                                        ),
                                        row=row, col=col
                                    )

                        fig.update_xaxes(
                            title_text="Year" if row == n_rows else "",
                            tickmode='array',
                            tickvals=[2002, 2012, 2022],
                            ticktext=['2002', '2012', '2022'],
                            showgrid=True,
                            gridcolor='#e5e5e5',
                            griddash='dot',
                            title_font=dict(
                                family="Arial", color='grey', size=10),
                            tickfont=dict(family="Arial",
                                          color='#2c3e50', size=9),
                            row=row, col=col
                        )
                        fig.update_yaxes(
                            title_text="Average Value" if row == n_rows else "",
                            showgrid=True,
                            gridcolor='#e5e5e5',
                            griddash='dot',
                            title_font=dict(
                                family="Arial", color='grey', size=10),
                            tickfont=dict(family="Arial",
                                          color='#2c3e50', size=9),
                            row=row, col=col
                        )

                        col = 2
                        marker_symbols = {2002: 'circle',
                                          2012: 'x', 2022: 'cross'}

                        for sex in ['Male', 'Female']:
                            for urb in ['Urban', 'Rural']:
                                gender_urban_key = f'{sex}_{urb}'
                                color = color_map.get(
                                    gender_urban_key, '#808080')

                                show_legend = gender_urban_key not in legend_groups_used
                                if show_legend:
                                    legend_groups_used.add(gender_urban_key)

                                all_years_subset = var_data[(var_data['sex'] == sex) &
                                                            (var_data['urban_rural_simple'] == urb)]

                                for year in [2002, 2012, 2022]:
                                    subset = var_data[(var_data['sex'] == sex) &
                                                      (var_data['urban_rural_simple'] == urb) &
                                                      (var_data['year'] == year)]
                                    if len(subset) > 0:
                                        symbol = marker_symbols[year]

                                        fig.add_trace(
                                            go.Scatter(
                                                x=subset['eg_score_norm_mean'],
                                                y=subset['Value'],
                                                mode='markers',
                                                name=f'{sex} - {urb}',
                                                marker=dict(
                                                    color=color,
                                                    size=8,
                                                    opacity=0.7,
                                                    line=dict(
                                                        width=1, color='white'),
                                                    symbol=symbol
                                                ),
                                                legendgroup=gender_urban_key,
                                                showlegend=show_legend,
                                                hovertemplate=f'<b>{var_name}</b><br>EG Score: %{{x:.3f}}<br>Value: %{{y:.2f}}<br>{sex} - {urb} - {year}<extra></extra>'
                                            ),
                                            row=row, col=col
                                        )

                                if len(all_years_subset) >= 2:
                                    try:
                                        z = np.polyfit(
                                            all_years_subset['eg_score_norm_mean'], all_years_subset['Value'], 1)
                                        p = np.poly1d(z)
                                        x_trend = np.linspace(all_years_subset['eg_score_norm_mean'].min(),
                                                              all_years_subset['eg_score_norm_mean'].max(), 50)
                                        fig.add_trace(
                                            go.Scatter(
                                                x=x_trend,
                                                y=p(x_trend),
                                                mode='lines',
                                                name=f'{sex}-{urb} trend',
                                                line=dict(
                                                    color=color, width=1.5, dash='dash'),
                                                legendgroup=gender_urban_key,
                                                showlegend=False,
                                                hoverinfo='skip'
                                            ),
                                            row=row, col=col
                                        )
                                    except:
                                        pass

                        fig.update_xaxes(
                            title_text="EG Score (Normalized)" if row == n_rows else "",
                            showgrid=True,
                            gridcolor='#e5e5e5',
                            griddash='dot',
                            title_font=dict(
                                family="Arial", color='grey', size=10),
                            tickfont=dict(family="Arial",
                                          color='#2c3e50', size=9),
                            row=row, col=col
                        )
                        fig.update_yaxes(
                            title_text="Average Value" if row == n_rows else "",
                            showgrid=True,
                            gridcolor='#e5e5e5',
                            griddash='dot',
                            title_font=dict(
                                family="Arial", color='grey', size=10),
                            tickfont=dict(family="Arial",
                                          color='#2c3e50', size=9),
                            row=row, col=col
                        )

                    year_legend_traces = [
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            name='2002',
                            marker=dict(symbol='circle', size=12, color='#666666', line=dict(
                                width=1.5, color='white')),
                            showlegend=True,
                            legendgroup='years',
                            hoverinfo='skip'
                        ),
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            name='2012',
                            marker=dict(symbol='x', size=12, color='#666666', line=dict(
                                width=1.5, color='white')),
                            showlegend=True,
                            legendgroup='years',
                            hoverinfo='skip'
                        ),
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            name='2022',
                            marker=dict(symbol='cross', size=12, color='#666666', line=dict(
                                width=1.5, color='white')),
                            showlegend=True,
                            legendgroup='years',
                            hoverinfo='skip'
                        )
                    ]
                    for trace in year_legend_traces:
                        fig.add_trace(trace)

                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='lines',
                            name='Trend Line',
                            line=dict(color='#666666', width=2, dash='dash'),
                            showlegend=True,
                            legendgroup='trend_lines',
                            hoverinfo='skip'
                        )
                    )

                    fig.update_layout(
                        template='plotly_white',
                        font=dict(family="Arial", color='#2c3e50'),
                        height=1400 if n_rows == 3 else 1000 if n_rows == 2 else 600,
                        width=1400,
                        title_text="",
                        showlegend=True,
                        legend=dict(
                            title=dict(text='Gender & Location',
                                       font=dict(size=12, color='#2c3e50')),
                            yanchor="top",
                            y=0.98,
                            xanchor="left",
                            x=1.02,
                            bgcolor='rgba(255, 255, 255, 0.95)',
                            bordercolor='#d0d0d0',
                            borderwidth=1.5,
                            font=dict(family="Arial",
                                      color='#2c3e50', size=10),
                            tracegroupgap=20,
                            itemwidth=30,
                            itemsizing='constant'
                        ),
                        hovermode='closest',
                        margin=dict(l=150, r=200, t=180, b=120),
                        annotations=[
                            dict(
                                text="<b>Composite Analysis: Key Variables Over Time & Relationship with Egalitarianism</b><br><span style='font-size:12px;color:grey'>Each row shows one variable: time trends (left) and EG score relationship (right)</span>",
                                xref="paper",
                                yref="paper",
                                x=0.5,
                                y=1.05,
                                xanchor="center",
                                yanchor="top",
                                showarrow=False,
                                font=dict(family="Arial",
                                          color='#2c3e50', size=14)
                            )
                        ]
                    )

                    st.plotly_chart(fig, use_container_width=False)

                    with st.expander("📊 Insights", expanded=False):
                        st.markdown(f"""
                        **Composite Dashboard**: Multi-variable analysis with temporal trends
                        - **Variables**: {', '.join([cols_question_mapping.get(v, v) for v in dashboard_vars])}
                        - **Left Column**: Time trends (2002 → 2012 → 2022) by gender and location
                        - **Right Column**: Relationship with Egalitarianism Score
                        - **Colors**: Gender & Location combinations (4 unique entries)
                        - **Markers**: Year symbols (circle=2002, x=2012, cross=2022)
                        - **Interpretation**: Compare how different variables change over time and relate to egalitarianism
                        """)
                else:
                    st.warning(
                        "⚠️ No data available for the selected variables.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Active Filters:**")
st.sidebar.write(f"**Countries:** {len(selected_countries_filter)} selected")
if len(selected_countries_filter) <= 20:
    st.sidebar.caption(f"Countries: {', '.join(selected_countries_filter)}")
else:
    st.sidebar.caption(
        f"Countries: {', '.join(selected_countries_filter[:20])}...")
st.sidebar.write(
    f"**Gender:** {', '.join(gender_filter) if len(gender_filter) > 0 else 'All'}")
st.sidebar.write(
    f"**Location:** {', '.join(urban_rural_filter) if len(urban_rural_filter) > 0 else 'All'}")
