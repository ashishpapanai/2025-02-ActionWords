# Gender Roles and Egalitarian Attitudes: A Longitudinal Analysis

## Project Overview
This repository contains the code and data for a course project built as part of the **Data Literacy Course (ML4102)**  at the **University of Tübingen** . The project analyzes longitudinal survey data to investigate gender roles and egalitarian attitudes. The study focuses on understanding how the **number of children in a family** and **dual-earner household status** influence egalitarian attitudes differently for men and women over time. The analysis is based on survey data collected across three waves: **2002, 2012, and 2022**.

The research is grounded in the findings and methodologies outlined in the paper: [ActionWords_DataLiT_WISE25.pdf](./doc/ActionWords_DataLiT_WISE25.pdf).

---

## Data Pipeline

### 1. Data Source
The primary data for this project comes from longitudinal survey datasets collected in 2002, 2012, and 2022. These datasets capture a wide range of variables related to gender roles, family structure, employment status, and egalitarian attitudes.

### 2. Variable Mapping Script
The repository includes a Python script (`gesis_variable_crawler.py`) that automates the process of crawling and mapping variables across different survey years. This ensures longitudinal compatibility and consistency in variable definitions and formats.

### 3. Data Cleaning & Harmonization
The data cleaning and harmonization process involves:
- Standardizing variable names and formats across the three survey waves.
- Handling missing data and ensuring consistency in variable definitions over two decades.
- Converting variables into categorical formats suitable for statistical analysis.
- Combining the datasets into a single, harmonized dataset for analysis.

The cleaned and harmonized dataset is saved as `combined_dataset_new.csv` in the `data/extras/` directory.

---

## Notebooks

### 1. `factor_analyzer.ipynb`
This notebook performs factor analysis on the cleansed dataset to compute factor scores, referred to as `eg_scores` (Egalitarianism Scores). These scores are used as a measure of egalitarian attitudes in subsequent analyses.

### 2. `respondent_eda.ipynb`
This notebook contains Exploratory Data Analysis (EDA) of the survey participants. It provides insights into the demographic and socioeconomic characteristics of the respondents across the three survey waves.

### 3. `household_analysis.ipynb`
This notebook focuses on hypotheses related to household variables, specifically testing the impact of the number of children and dual-income status on gender-based attitudes. It includes statistical models and visualizations to explore these relationships.

### 4. `cleaning.ipynb`
This notebook outlines the data cleaning process for the survey datasets from 2002, 2012, and 2022. It includes steps for handling missing values, standardizing variable formats, and preparing the data for further analysis. The output of this notebook is the cleaned datasets stored in the `data/cleaned_csv/` directory.

### 5. `childcare_affects.ipynb`
This notebook investigates the impact of childcare responsibilities on egalitarian attitudes. It combines the datasets from 2002, 2012, and 2022, performs data normalization, and maps categorical variables to numerical values for analysis. The notebook also includes statistical models to analyze the relationship between childcare responsibilities, number of children, and gender-based attitudes.

### 6. `generate_grouping.ipynb`
This notebook focuses on creating groupings and clusters within the dataset based on demographic and socioeconomic variables. It uses clustering techniques to identify patterns and group respondents with similar characteristics, which are then used in subsequent analyses to explore variations in egalitarian attitudes.

---

## Statistical Testing
The repository includes scripts and notebooks for conducting statistical tests to examine the following relationships:

1. **Equality Scores vs. Income Control in Marriage**
   - Investigates the association between egalitarian attitudes and the control of income within marriages.

2. **Female Partner’s Employment vs. Equality Score**
   - Examines how the employment status of female partners influences egalitarian attitudes.

3. **Education Levels vs. Equality Score**
   - Analyzes the relationship between education levels and egalitarian attitudes.

---

## Repository Structure
```
├── README.md
├── requirements.txt
├── assets/
│   └── images/
│       └── statistical/
├── data/
│   ├── common_question_mapping.csv
│   ├── cleaned_csv/
│   │   ├── 2002.csv
│   │   ├── 2012.csv
│   │   └── 2022.csv
│   ├── efa_csv/
│   │   ├── 2002.csv
│   │   ├── 2012.csv
│   │   └── 2022.csv
│   ├── extras/
│   │   └── combined_dataset_new.csv
│   └── var_mapping/
│       ├── 2002/
│       │   ├── 2002_variables_short.json
│       │   └── 2002_variables.json
│       ├── 2012/
│       │   ├── 2012_variables_short.json
│       │   └── 2012_variables.json
│       └── 2022/
│           ├── 2022_variables_short.json
│           ├── 2022_variables.json
│           └── country.json
├── doc/
│   ├── bibliography.bib
│   ├── icml2025.bst
│   ├── icml2025.sty
│   ├── report_template.tex
│   └── assets/
├── notebooks/
│   ├── atitude_analysis.ipynb
│   ├── childcare_affects.ipynb
│   ├── cleaning.ipynb
│   ├── eg_score_v2.ipynb
│   ├── eg_score.ipynb
│   ├── factor_analysis.ipynb
│   ├── financial_equality.ipynb
│   ├── household_analysis.ipynb
│   └── respondent_eda.ipynb
├── scripts/
│   ├── CRAWLER.MD
│   └── gesis_variable_crawler.py
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ashishpapanai/ML4102.git
   cd ML4102
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes. Ensure that your code adheres to the repository’s coding standards and is well-documented.

---

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the LICENSE file for details.