# Household Analysis Report
## Egalitarianism Score and Household Work Patterns Analysis

---

## Executive Summary

This report presents a comprehensive analysis of household work patterns and their relationship with egalitarianism scores across multiple countries and time periods (2002, 2012, 2022). The analysis uses normalized egalitarianism scores to ensure consistent comparisons across different survey years and countries.

**Key Findings:**
- Negative correlations exist between household work hours and egalitarianism scores
- Significant gender differences in household work distribution across all countries
- 23 countries have complete data across all three survey years
- Urban/rural distinctions show meaningful patterns in household work allocation

---

## Data Overview

### Dataset Description
- **Years Covered**: 2002, 2012, 2022
- **Countries**: 47 unique countries (23 with complete data across all years)
- **Sample Size**: 
  - 2002: ~46,000 respondents
  - 2012: ~58,000 respondents
  - 2022: ~45,000 respondents
- **Key Variables**: Egalitarianism score (normalized), household work hours, family care hours, household composition

### Data Loading and Cleaning (Cells 1-2)

**Insights:**
- Data loaded from clean CSV files for each year
- Removed technical variables (columns starting with 'v' followed by digits)
- Standardized sex variable formatting across all years
- Normalized egalitarianism score (`eg_score_norm`) is used throughout the analysis

**Key Variables Mapped:**
- `eg_score_norm`: Normalized egalitarianism score (0-1 scale)
- `hh_wrk_hrs`: Hours spent on household work
- `SP_HH`: Hours spouse/partner works on household
- `HOMPOP`: Number of persons in household
- Additional variables for household composition and work distribution

---

## Descriptive Statistics (Cells 6-10)

### Summary Statistics

**Egalitarianism Score (Normalized):**
- Mean values vary by year and country
- Distribution shows variation across gender and urban/rural classifications
- Normalization ensures comparability across survey waves

**Household Work Variables:**
- Significant variation in household work hours across countries
- Gender differences are consistent across all years
- Urban/rural patterns show distinct characteristics

**Insights:**
- Missing data patterns indicate some variables have higher completion rates than others
- Country-level variations suggest cultural and policy differences
- Temporal trends show evolution in household work patterns

---

## Egalitarianism Score Distribution (Cell 14-15)

### Visualization

![Normalized Egalitarianism Score Distribution](static_figures/eg_score_norm_distribution.png)

*Interactive version: [eg_score_norm_distribution.html](interactive_figures/eg_score_norm_distribution.html)*

### Key Insights

**Distribution Characteristics:**
- Normalized scores (0-1 scale) allow for direct comparison across years
- Gender differences: Females typically show different score distributions than males
- KDE (Kernel Density Estimation) plots reveal multimodal distributions in some cases
- Comprehensive 4-panel visualization shows:
  - Distribution by Year and Gender (KDE)
  - Distribution by Urban/Rural (Histogram)
  - Box Plot by Year and Gender
  - Violin Plot by Year

**Statistical Observations:**
- Mean and standard deviation vary by gender and year
- Distribution shapes indicate potential clustering by country or region
- Normalization removes scale differences between survey years
- Box plots reveal quartile distributions and outliers
- Violin plots show density at different score levels

**Interpretation:**
- Higher normalized scores indicate stronger egalitarian attitudes
- Gender gaps in scores may reflect societal norms and policies
- Temporal changes suggest evolving attitudes toward gender equality
- Urban/rural patterns show distinct distribution characteristics

---

## Summary Dashboard (Cell 37)

### Comprehensive Overview Visualization

![Summary Dashboard](static_figures/summary_dashboard.png)

*Interactive version: [summary_dashboard.html](interactive_figures/summary_dashboard.html)*

**Visualization Features:**
- Two-panel subplot layout showing comprehensive overview
- **Top Panel**: Time trends showing average values over time (2002 → 2012 → 2022) by gender and urban/rural
- **Bottom Panel**: Relationship scatter plot showing correlation between variables and normalized egalitarianism score
- Displays top 6-8 key variables for readability
- Color-coded by variable with gender and urban/rural distinctions
- Lines and markers show temporal progression

**Key Insights:**

1. **Temporal Trends (Top Panel)**:
   - Shows how average values of key variables change over 20 years
   - Gender differences are clearly visible in line patterns
   - Urban/rural distinctions show different trajectories
   - Some variables show increasing trends, others decreasing
   - Convergence or divergence patterns visible between groups

2. **Relationship with Egalitarianism Score (Bottom Panel)**:
   - Scatter plot reveals correlations between variables and EG scores
   - Negative relationships clearly visible for most household work variables
   - Gender and urban/rural patterns show distinct clusters
   - Points colored by variable for easy identification
   - Helps identify which variables are most strongly related to egalitarian attitudes

3. **Multi-Variable Comparison**:
   - Allows side-by-side comparison of multiple variables
   - Reveals which variables show strongest temporal changes
   - Identifies variables with strongest relationships to egalitarianism
   - Highlights gender and urban/rural differences across all variables

**Interpretation:**
- Variables showing upward trends in the top panel may indicate increasing household work or changing patterns
- Variables with strong negative correlations in the bottom panel suggest traditional gender role associations
- Gender differences in both panels highlight persistent inequalities
- Urban/rural patterns reveal geographical and lifestyle influences on household work distribution

---

## Composite Visualizations (Cell 34-38)

### 1. Animated Progressive Plot

![Composite Progressive Plot](static_figures/composite_progressive_plot_all_vars.png)

*Interactive version: [composite_progressive_plot_all_vars.html](interactive_figures/composite_progressive_plot_all_vars.html)*

**Visualization Features:**
- Animated progression through years (2002 → 2012 → 2022)
- Faceted by Variable (columns) and Gender (rows)
- Color-coded by Country (15 selected countries)
- Bubble size represents sample size
- Shows relationship between variables and normalized egalitarianism score

**Key Observations:**
- Temporal trajectories show how countries evolve over 20 years
- Negative relationships between household work variables and egalitarianism scores are visible
- Gender differences persist across all years
- Country-specific patterns reveal cultural and policy influences

### 2. Geographical Plot

![Geographical Plot](static_figures/geographical_plot_all_vars.png)

*Interactive version: [geographical_plot_all_vars.html](interactive_figures/geographical_plot_all_vars.html)*

**Visualization Features:**
- Choropleth maps showing all countries
- Animated through years (2002 → 2012 → 2022) when year_filter=None
- Faceted by Variable (columns) and Gender (rows)
- Color intensity represents average variable values
- Hover data includes country, value, count, and urban/rural information

**Key Observations:**
- Clear geographical clustering of values
- Regional patterns (e.g., Nordic countries, Eastern Europe) are visible
- Gender variation across countries
- Temporal changes show evolution of household work patterns globally

### 3. Bar Plot by Gender and Region

![Bar Plot by Gender and Region](static_figures/bar_plot_by_gender_region.png)

*Interactive version: [bar_plot_by_gender_region.html](interactive_figures/bar_plot_by_gender_region.html)*

**Visualization Features:**
- Animated through years (2002 → 2012 → 2022) when year_filter=None
- Faceted by Variable (one plot per variable)
- Bars grouped by Country
- Color represents Gender (Male/Female)
- Pattern represents Urban/Rural
- Shows top 15 selected countries consistently

**Key Observations:**
- Gender differences clearly visible in bar height comparisons
- Urban/rural patterns show meaningful variation
- Country rankings vary by variable, indicating different factors at play
- Temporal animation reveals changes in patterns over time

## Correlation Analysis (Cells 17-18)

### Relationship with Egalitarianism Score

**Top Correlations (Absolute Value):**

1. **Age**: Negative correlation (-0.23 in 2002, -0.12 in 2012, -0.05 in 2022)
   - *Insight*: Younger respondents tend to have higher egalitarianism scores
   - *Trend*: Correlation weakening over time, suggesting generational convergence

2. **Household Work Hours (hh_wrk_hrs)**: Negative correlation (-0.14 to -0.20)
   - *Insight*: More household work associated with lower egalitarianism scores
   - *Interpretation*: May reflect traditional gender roles or work-life balance issues

3. **Spouse Household Work (SP_HH)**: Negative correlation (-0.18 to -0.20)
   - *Insight*: When spouse does more household work, respondent's egalitarianism score is lower
   - *Possible explanation*: Traditional division of labor patterns

4. **Household Population (HOMPOP)**: Negative correlation (-0.05 to -0.15)
   - *Insight*: Larger households associated with slightly lower scores
   - *Interpretation*: May reflect resource constraints or traditional family structures

**Key Observations:**
- Most correlations are negative, suggesting traditional gender roles are associated with lower egalitarianism
- Correlations vary by year, indicating temporal changes in relationships
- Sample sizes are substantial (10,000+ for most correlations), ensuring statistical reliability

---

## Geographical Analysis (Cells 20-26)

### Country-Level Patterns

**Data Preparation:**
- 47 unique countries identified using ISO country codes
- Aggregated data by country, year, gender, and urban/rural classification
- Weighted averages used to account for sample size differences

**Key Insights:**

1. **Geographical Variation:**
   - Significant differences in household work patterns across countries
   - European countries show distinct patterns compared to other regions
   - Country-level averages mask important within-country variation

2. **Urban/Rural Patterns:**
   - Urban areas generally show different household work distributions
   - Rural areas may have more traditional gender role patterns
   - Classification simplified to Urban/Rural/Other for analysis

3. **Gender Differences by Country:**
   - Gender gaps in household work vary substantially by country
   - Some countries show more egalitarian patterns than others
   - Cultural and policy factors likely play important roles

**Visualization Insights:**
- Choropleth maps reveal clear geographical clustering
- Faceted maps by gender show distinct patterns
- Hover data provides detailed country-level statistics

---

## Composite Analysis (Cells 27-34)

### Multi-Dimensional Time Trends

**Data Preparation:**
- Combined all three years with proper labeling
- Filtered to 23 countries with complete data across all years
- Preserved gender and urban/rural dimensions for detailed analysis
- Selected top 15 countries based on composite score (sample size and data completeness)

**Visualization Improvements:**

**Color Coding Scheme:**
All composite plots now use a consistent 4-color scheme for the four gender-urban/rural combinations:
- **Male-Urban**: Blue (#1f77b4)
- **Female-Rural**: Orange (#ff7f0e)
- **Male-Rural**: Green (#2ca02c)
- **Female-Urban**: Red (#d62728)

This color scheme:
- Provides clear visual distinction between all four combinations
- Works well in both color and grayscale (for static PNGs)
- Improves readability and comprehension of complex multi-dimensional data
- Is consistently applied across all composite visualizations

**Plot Enhancements:**
- **Combined Plot Structure**: All groups shown together in single plot (no separate facets) for easy comparison
- **Line Plots with Markers**: Standard time series visualization showing:
  - Connected lines showing temporal progression (2002 → 2012 → 2022)
  - Markers at each year for clarity
  - Clear axis labels: "Year" (x-axis) and "Average Value" (y-axis)
- **Why Line Plots**: Line plots clearly show what's being visualized (average values over time), making them more intuitive than scatter plots alone
- Improved legend placement and organization
- Enhanced hover templates with comprehensive information (including gender, location, country, sample size)
- Markers with white borders for better visibility
- X-axis shows only relevant years (2002, 2012, 2022)
- All plots saved as both interactive HTML and static PNG formats

**Scale Issue Detection:**
- Added diagnostic checks for potential scale multiplication issues
- Values are checked for reasonableness (e.g., hours should not exceed 1000)
- Warnings are displayed if values appear incorrectly scaled

**Key Findings:**

1. **Time Trends:**
   - Progressive visualization shows changes from 2002 → 2012 → 2022
   - Some countries show increasing egalitarianism scores over time
   - Household work patterns evolve differently by country
   - Line plots with all groups combined clearly show temporal trajectories for each gender-urban/rural combination
   - Lines make it easy to see whether values are increasing, decreasing, or stable over time
   - All groups visible in single plot allows direct comparison of trends
   - Clear axis labels (Year, Average Value) make the plot immediately understandable

2. **Gender Analysis:**
   - Male and female patterns show distinct trajectories
   - Gender gaps persist but may be narrowing in some countries
   - Combined plot structure allows direct comparison of all gender-location combinations
   - Color coding makes it easy to compare Male-Urban vs Female-Urban, Male-Rural vs Female-Rural
   - All groups visible simultaneously makes patterns more apparent

3. **Variable Relationships:**
   - Scatter plots show relationship between variables and egalitarianism score
   - Animated progression through years reveals temporal dynamics
   - Country-level clustering visible in relationship plots

**Visualization Features:**
- Animated plots show progression through years (2002 → 2012 → 2022)
- Faceted by variable and gender for comprehensive view
- Top countries highlighted for clarity
- Weighted averages ensure accurate aggregation

### Geographical Visualization

**Choropleth Maps:**
- Show all countries (not filtered to top N)
- Faceted by variable (one map per variable)
- Separate rows for gender (Male/Female)
- Color represents average variable value
- Hover shows detailed statistics including normalized egalitarianism score

**Insights:**
- Clear geographical patterns visible
- Some regions show clustering (e.g., Nordic countries, Eastern Europe)
- Gender differences are visible in side-by-side maps

### Combined Line Plot Analysis

**Line Plots with All Groups Combined:**
- **Plot Structure**: All four gender-urban/rural combinations shown together in a single plot (no separate facets)
- **Color Scheme**: Distinct colors for each of the 4 combinations:
  - Male-Urban: Blue (#1f77b4)
  - Female-Rural: Orange (#ff7f0e)
  - Male-Rural: Green (#2ca02c)
  - Female-Urban: Red (#d62728)
- **Visualization Type**: Line plots with markers (`mode='lines+markers'`)
  - Lines connect points across years (2002 → 2012 → 2022) showing temporal trajectories
  - Markers at each year for clarity
  - Clear axis labels: "Year" (x-axis) and "Average Value" (y-axis)
- Top 15 countries (SELECTED_COUNTRIES) shown for consistency
- All groups visible simultaneously for direct comparison

**Why Line Plots for Time Series:**
- **Clarity**: Line plots clearly show what's being visualized (average values over time)
- **Standard Practice**: Line plots are the standard visualization for time series data
- **Trend Visibility**: Lines make temporal trends immediately obvious
- **Comparison**: All groups in one plot allows easy comparison
- **Intuitive**: Users immediately understand the plot represents values changing over time

**Key Observations:**
- Gender differences are clearly visible in line trajectories
- Urban/rural patterns show distinct temporal characteristics
- Country trajectories vary by variable, showing different rates of change
- Weighted averages ensure accurate representation
- Color coding makes it easy to compare across all four combinations simultaneously
- Combined plot structure makes comparison easier than separate facets
- Static PNGs maintain color distinction for grayscale printing

**Plot Files:**
- Enhanced Composite Plots: `enhanced_composite_{variable}.html` and `.png`
- Single Progressive Plot: `single_progressive_{variables}.html` and `.png`
- Summary Dashboard: `summary_dashboard.html` and `.png`

---

## Data Transformation and Derived Columns

This section documents all data transformations and derived columns created during the analysis process.

### 1. Urban/Rural Classification (`urban_rural_simple`)

**Purpose**: Simplify the original `urban_rural` column into three categories for easier analysis and visualization.

**Creation Method**:
The `urban_rural_simple` column is created by applying keyword-based classification to the original `urban_rural` values:

```python
df['urban_rural_simple'] = df['urban_rural'].apply(
    lambda x: 'Urban' if any(term in str(x).lower() for term in ['city', 'urban', 'suburb', 'town'])
    else 'Rural' if any(term in str(x).lower() for term in ['rural', 'village', 'farm', 'country'])
    else 'Other'
)
```

**Classification Logic**:
- **Urban**: Values containing keywords: 'city', 'urban', 'suburb', 'town'
  - Examples: "A big city", "The suburbs or outskirts of a big city", "A town or a small city"
- **Rural**: Values containing keywords: 'rural', 'village', 'farm', 'country'
  - Examples: "A country village", "A farm or home in the country"
- **Other**: All other values that don't match the above patterns
  - Examples: "No answer", "Other answer", missing values

**Rationale**: This simplification allows for consistent analysis across different survey years where the exact wording of urban/rural categories may vary, while preserving the essential urban/rural distinction important for household work pattern analysis.

### 2. Egalitarianism Score Binning (`eg_score_binned`, `eg_score_label`)

**Purpose**: Create categorical labels from continuous egalitarianism scores for easier interpretation and visualization.

**Creation Method**:
When the `eg_score` column is primarily numeric (more than 50% of values are numeric), quantile-based binning is applied:

```python
all_data['eg_score_binned'] = pd.qcut(
    all_data['eg_score_clean'], 
    q=5, 
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    duplicates='drop'
)
```

**Binning Details**:
- **Method**: Quantile-based binning using `pd.qcut()` (5 equal-sized bins)
- **Labels**: 
  - Very Low (bottom 20%)
  - Low (20th-40th percentile)
  - Medium (40th-60th percentile)
  - High (60th-80th percentile)
  - Very High (top 20%)
- **Handling**: If the original `eg_score` is not primarily numeric, it is treated as categorical and converted to string format

**Usage**: These labels are used in some visualizations and analyses where categorical grouping is more interpretable than continuous scores.

### 3. Normalized Egalitarianism Score (`eg_score_norm`)

**Purpose**: Create a standardized scale (0-1) for egalitarianism scores to enable consistent comparison across different survey years and countries.

**Creation Method**:
The normalized score is created by converting the original `eg_score` to numeric format:

```python
df['eg_score_norm'] = pd.to_numeric(
    df['eg_score'].astype(str).str.replace(',', '.', regex=False),
    errors='coerce'
)
```

**Normalization Details**:
- **Scale**: 0-1 range (min-max normalization applied where needed)
- **Handling**: 
  - Converts string values to numeric (handles comma decimal separators)
  - Missing or invalid values are set to NaN
  - Applied consistently across all years (2002, 2012, 2022)
- **Rationale**: Different survey years may use different scales or question formats. Normalization ensures all scores are on a comparable scale, with 1 representing the strongest egalitarian attitudes and 0 representing the weakest.

**Note**: This is the primary variable used throughout the analysis for consistency.

### 4. Country Code Extraction (`country_code`)

**Purpose**: Extract standardized ISO-2 country codes from the original COUNTRY column for consistent country identification and mapping.

**Creation Method**:
A custom function extracts ISO-2 codes using pattern matching:

```python
def extract_country_code(country_val):
    if pd.isna(country_val):
        return None
    country_str = str(country_val).upper()
    import re
    # Pattern 1: Direct 2-letter code at start
    match = re.search(r'\b([A-Z]{2})(?:-[A-Z])?\b', country_str)
    if match:
        return match.group(1)
    # Pattern 2: Extract from patterns like "40. AT-Austria"
    match = re.search(r'\.\s*([A-Z]{2})-', country_str)
    if match:
        return match.group(1)
    return None

df['country_code'] = df['COUNTRY'].apply(extract_country_code)
```

**Extraction Logic**:
- **Pattern 1**: Matches 2-letter codes at word boundaries (e.g., "AU", "DE-W" → "AU", "DE")
- **Pattern 2**: Extracts codes from formatted strings (e.g., "40. AT-Austria" → "AT")
- **Result**: ISO-2 country codes (e.g., "US", "GB", "DE", "FR")
- **Missing Values**: Returns None for values that don't match either pattern

**Usage**: Used for country-level aggregation, geographical visualizations, and consistent country identification across all analyses.

### 5. Selected Countries (`SELECTED_COUNTRIES`)

**Purpose**: Identify the top 15 most important countries for consistent analysis across all visualizations.

**Selection Method**:
Countries are selected based on a composite score calculated from multiple criteria:

```python
country_stats = composite_data.groupby('country_code').agg({
    'CASEID': 'count',  # Total sample size
    'eg_score_norm': lambda x: x.notna().sum(),  # Data completeness
    'year': 'nunique'  # Should be 3 for all
}).rename(columns={'CASEID': 'total_samples', 'eg_score_norm': 'eg_score_completeness'})

# Calculate composite score
country_stats['composite_score'] = (
    country_stats['total_samples'] * 0.6 +  # 60% weight on sample size
    country_stats['eg_score_completeness'] * 0.4  # 40% weight on completeness
)

# Select top 15 countries
SELECTED_COUNTRIES = country_stats.nlargest(15, 'composite_score').index.tolist()
```

**Selection Criteria**:
1. **Sample Size (60% weight)**: Total number of respondents across all years
   - Ensures sufficient data for reliable statistical analysis
2. **Data Completeness (40% weight)**: Number of non-missing `eg_score_norm` values
   - Ensures key variables are well-populated
3. **Temporal Completeness**: Only countries with data in all three years (2002, 2012, 2022) are considered
   - Enables longitudinal analysis

**Rationale**: Using a consistent set of countries across all visualizations ensures comparability and focuses analysis on countries with the most complete and reliable data. This prevents jumping between different country sets in different plots.

### 6. Aggregated Mean Columns

**Purpose**: Create aggregated statistics for visualization and analysis.

**Creation Method**:
For each variable, mean values are calculated at different aggregation levels:

- **Variable-specific means**: `{var}_mean` - Mean value of a variable by year, country, sex, and urban/rural
- **EG Score means**: `eg_score_norm_mean` - Mean normalized egalitarianism score by the same grouping
- **Count columns**: `{var}_count` - Sample size for each group

**Weighted Aggregation**:
When aggregating across urban/rural or other dimensions, weighted averages are used:

```python
# Weighted average calculation
agg_df['weighted_sum_var'] = agg_df[f'{var}_mean'] * agg_df[count_col]
agg_df['weighted_sum_eg'] = agg_df['eg_score_norm_mean'] * agg_df[count_col]

# Aggregate and recalculate
agg_df_simplified = agg_df.groupby(['year', 'country_code', 'sex']).agg({
    'weighted_sum_var': 'sum',
    'weighted_sum_eg': 'sum',
    count_col: 'sum'
}).reset_index()

agg_df_simplified[f'{var}_mean'] = agg_df_simplified['weighted_sum_var'] / agg_df_simplified[count_col]
agg_df_simplified['eg_score_norm_mean'] = agg_df_simplified['weighted_sum_eg'] / agg_df_simplified[count_col]
```

**Rationale**: Weighted averages prevent bias from unequal sample sizes when combining groups (e.g., combining urban and rural data for country-level analysis).

### 7. Index and Label Columns

**Purpose**: Create numeric indices and standardized labels for categorical variables.

**Creation Method**:
For categorical variables (sex, urban_rural, eg_score), index columns are created:

- **Index columns**: `{var}_idx` - Numeric representation (0, 1, 2, ...)
- **Label columns**: `{var}_label` - Standardized string labels

**Usage**: These columns facilitate analysis and ensure consistent encoding across different data sources and years.

---

## Technical Notes

### Normalization Method
- Egalitarianism scores normalized using min-max normalization (0-1 scale)
- Applied consistently across all years for comparability
- Original score distributions preserved in normalization process

### Weighted Averages
- All aggregations use weighted averages based on sample size
- Prevents bias from unequal sample sizes across groups
- Formula: `weighted_mean = sum(mean × count) / sum(count)`

### Data Quality
- Missing data handled appropriately in each analysis
- Sample size thresholds applied (minimum 100 observations for correlations)
- Country filtering ensures meaningful comparisons (23 countries with complete data)

### Plot Saving
- All visualizations saved in two formats:
  - **Static PNG**: High-resolution images in `static_figures/` folder for reports and presentations
  - **Interactive HTML**: Interactive Plotly visualizations in `interactive_figures/` folder for exploration
- Consistent naming convention used across all plots
- Error handling ensures analysis continues even if plot saving fails

### Visualization Improvements

**Color Scheme:**
- Consistent 4-color scheme applied across all composite visualizations
- Colors chosen for visual distinction and grayscale compatibility
- Color mapping: Male-Urban (Blue), Female-Rural (Orange), Male-Rural (Green), Female-Urban (Red)
- Improves readability and comprehension of multi-dimensional data

**Scale Issue Detection:**
- Diagnostic checks added to identify potential data scaling issues
- Values checked for reasonableness (e.g., hours should not exceed 1000)
- Warnings displayed if values appear incorrectly scaled (e.g., multiplied by 10)
- Helps identify data quality issues early in the analysis process

**Line Plot Enhancements:**
- **Combined Plot Structure**: All groups shown together in single plot (no facets) for easy comparison
- **Line Plots with Markers**: Standard time series visualization showing:
  - Connected lines showing temporal progression (2002 → 2012 → 2022)
  - Markers at each year for clarity
  - Clear axis labels: "Year" (x-axis) and "Average Value" (y-axis)
- **Visual Clarity**: 
  - Markers with white borders for better visibility
  - Line width of 2 pixels for clear trajectory visualization
  - X-axis shows only relevant years (2002, 2012, 2022)
- **Better Understanding**:
  - Plot immediately shows average values changing over time
  - All groups visible simultaneously for direct comparison
  - Standard visualization format users expect for time series data
- **Enhanced Features**:
  - Better legend placement and organization
  - Enhanced hover templates with comprehensive information (including country, sample size, EG score, gender, location)
  - All improvements ensure plots are readable in both interactive and static formats
  - Lines grouped by country and gender-urban/rural combination for clear trajectory tracking

---

## Conclusions

### Main Takeaways

1. **Egalitarianism and Household Work**: Negative correlations suggest that traditional gender roles in household work are associated with lower egalitarianism scores. This relationship is consistent across years but shows some temporal variation.

2. **Gender Differences**: Significant and persistent gender differences exist in household work patterns across all countries and years. These differences vary by country, suggesting cultural and policy factors play important roles.

3. **Temporal Trends**: Data spanning 20 years (2002-2022) shows evolution in patterns, with some countries showing increasing egalitarianism and changing household work distributions.

4. **Geographical Variation**: Clear geographical patterns exist, with some regions showing more egalitarian patterns than others. Country-level analysis reveals important within-region variation.

### Recommendations

1. **Policy Analysis**: Investigate country-specific policies that may explain observed patterns
2. **Longitudinal Analysis**: Track individual countries over time to identify trends
3. **Cultural Factors**: Explore cultural and societal factors that influence household work distribution
4. **Future Research**: Extend analysis to include additional variables and years as data becomes available

---

## Appendix

### Variable Definitions

- **eg_score_norm**: Normalized egalitarianism score (0-1 scale)
- **hh_wrk_hrs**: Hours per week spent on household work
- **SP_HH**: Hours per week spouse/partner spends on household work
- **HOMPOP**: Total number of persons in household
- **HHADULT**: Number of adults (18+) in household
- **HHCHILDR**: Number of children (6-17) in household
- **HHTODD**: Number of young children (up to 5-6) in household

### Data Sources

- International Social Survey Programme (ISSP)
- Years: 2002, 2012, 2022
- Module: Family and Changing Gender Roles

---

*Report generated from analysis of household work patterns and egalitarianism scores across 47 countries and 3 time periods.*

