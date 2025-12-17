"""
ISSP Data Harmonization Script
Harmonizes variables across 2002, 2012, and 2022 ISSP Family and Gender Roles surveys
for longitudinal analysis.

Based on the project plan: "The Evolution of Domestic Spheres: A Longitudinal 
Strategic Framework for Analyzing Gender Role Trajectories in OECD Nations (2002–2022)"
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False
    print("Warning: pyreadstat not available. SPSS files will require manual conversion.")

try:
    import yaml  # PyYAML
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Define base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CROSSWALK_PATH = DATA_DIR / "harmonized" / "issp_variable_crosswalk.yaml"

def load_variable_mappings():
    """Load variable name mappings from JSON files"""
    mappings = {}
    for year in [2002, 2012, 2022]:
        json_path = DATA_DIR / str(year) / f"ZA{3880 if year == 2002 else (5900 if year == 2012 else 10000)}_variables_short.json"
        with open(json_path, 'r') as f:
            mappings[year] = json.load(f)
    return mappings

def load_crosswalk_yaml(crosswalk_path: Path = DEFAULT_CROSSWALK_PATH):
    """Load harmonization mapping from YAML (single source of truth)."""
    if not HAS_YAML:
        raise ImportError("PyYAML is required to load the YAML crosswalk (pip install pyyaml).")
    if not crosswalk_path.exists():
        raise FileNotFoundError(f"Crosswalk YAML not found: {crosswalk_path}")
    with open(crosswalk_path, "r") as f:
        crosswalk = yaml.safe_load(f)
    if "variables" not in crosswalk:
        raise ValueError("Invalid crosswalk YAML: missing top-level 'variables'")
    return crosswalk

def load_dataset(year):
    """Load dataset for a given year"""
    if year == 2002:
        file_path = DATA_DIR / "2002" / "ZA3880_v1-1-0.dta"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_stata(file_path, convert_categoricals=False)
    elif year == 2012:
        file_path = DATA_DIR / "2012" / "2012.sav"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if HAS_PYREADSTAT:
            df, meta = pyreadstat.read_sav(str(file_path))
        else:
            raise ImportError("pyreadstat required for SPSS files. Install with: pip install pyreadstat")
    elif year == 2022:
        file_path = DATA_DIR / "2022" / "2022.dta"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_stata(file_path, convert_categoricals=False)
    else:
        raise ValueError(f"Unsupported year: {year}")
    
    # Add survey year
    df['survey_year'] = year
    
    # Convert column names to string (some may be numeric)
    df.columns = [str(col) for col in df.columns]
    
    return df

def _harmonization_map_from_crosswalk(crosswalk: dict) -> dict:
    """
    Convert YAML structure into:
      {harmonized_name: {2002: original_or_None, 2012: ..., 2022: ...}}
    """
    out = {}
    for harmonized_name, spec in crosswalk["variables"].items():
        years = spec.get("years", {})
        out[harmonized_name] = {
            2002: (years.get("2002") or {}).get("name"),
            2012: (years.get("2012") or {}).get("name"),
            2022: (years.get("2022") or {}).get("name"),
        }
    return out

def harmonize_variables(df, year, harmonization_map):
    """Rename variables to harmonized names (mapping loaded from YAML)."""
    rename_dict = {}

    for harmonized_name, year_mapping in harmonization_map.items():
        original_name = year_mapping.get(year)
        if original_name is None:
            continue
        if original_name in df.columns:
            rename_dict[original_name] = harmonized_name
        else:
            print(f"Warning: {original_name} not found in {year} dataset for {harmonized_name}")

    df_renamed = df.rename(columns=rename_dict)

    for harmonized_name in harmonization_map.keys():
        if harmonized_name not in df_renamed.columns:
            df_renamed[harmonized_name] = np.nan

    return df_renamed

def select_harmonized_columns(df, harmonization_map):
    """Select only harmonized columns plus survey_year"""
    # Ensure stable order and avoid duplicates (e.g., survey_year can be in mapping AND appended)
    harmonized_cols = list(harmonization_map.keys()) + ["survey_year"]
    seen = set()
    unique_cols = []
    for col in harmonized_cols:
        if col in df.columns and col not in seen:
            unique_cols.append(col)
            seen.add(col)
    return df[unique_cols]

def create_mapping_documentation(harmonization_map, variable_mappings):
    """Create comprehensive documentation of variable mappings"""
    doc = {
        'harmonization_rationale': {
            'description': 'This mapping harmonizes variables across ISSP Family and Gender Roles surveys (2002, 2012, 2022)',
            'source': 'Based on project plan: "The Evolution of Domestic Spheres"',
            'notes': [
                'Variable names changed from uppercase (V36) in 2002 to lowercase (v34) in 2022',
                'Some variables are missing in certain years (e.g., task_small_repairs in 2022)',
                'ISCO occupation codes changed from ISCO88 to ISCO08 between 2012 and 2022'
            ]
        },
        'variable_mappings': {}
    }
    
    for harmonized_name, year_mapping in harmonization_map.items():
        mapping_entry = {
            'harmonized_name': harmonized_name,
            'years': {}
        }
        
        for year, original_name in year_mapping.items():
            if original_name is not None:
                year_entry = {
                    'original_name': original_name,
                    'description': variable_mappings[year].get(original_name, 'Description not found')
                }
                mapping_entry['years'][str(year)] = year_entry
            else:
                mapping_entry['years'][str(year)] = {
                    'original_name': None,
                    'description': 'Variable not available in this year'
                }
        
        doc['variable_mappings'][harmonized_name] = mapping_entry
    
    return doc

def main():
    """Main harmonization pipeline"""
    print("=" * 80)
    print("ISSP Data Harmonization Pipeline")
    print("=" * 80)

    # Load harmonization mapping from YAML
    print("\n1. Loading YAML crosswalk mapping...")
    crosswalk = load_crosswalk_yaml()
    harmonization_map = _harmonization_map_from_crosswalk(crosswalk)
    
    # Load and harmonize datasets
    datasets = {}
    for year in [2002, 2012, 2022]:
        print(f"\n2. Loading {year} dataset...")
        df = load_dataset(year)
        print(f"   Original shape: {df.shape}")
        
        print(f"3. Harmonizing {year} variables...")
        df_harmonized = harmonize_variables(df, year, harmonization_map)
        
        print(f"4. Selecting harmonized columns for {year}...")
        df_final = select_harmonized_columns(df_harmonized, harmonization_map)
        print(f"   Final shape: {df_final.shape}")
        
        datasets[year] = df_final
    
    # Combine datasets
    print("\n5. Combining datasets...")
    df_combined = pd.concat(datasets.values(), ignore_index=True, sort=False)
    print(f"   Combined shape: {df_combined.shape}")
    
    # Save outputs
    output_dir = BASE_DIR / "data" / "harmonized"
    output_dir.mkdir(exist_ok=True)
    
    print("\n6. Saving harmonized dataset (CSV)...")
    csv_file = output_dir / "issp_harmonized_2002_2012_2022.csv"
    df_combined.to_csv(csv_file, index=False)
    print(f"   Saved to: {csv_file}")
    
    # Create summary statistics
    print("\n7. Generating summary statistics...")
    summary = {
        'total_records': len(df_combined),
        'records_by_year': df_combined['survey_year'].value_counts().to_dict(),
        'variables_available': len([col for col in harmonization_map.keys() if col in df_combined.columns]),
        'missing_variables_by_year': {}
    }
    
    for year in [2002, 2012, 2022]:
        df_year = df_combined[df_combined['survey_year'] == year]
        missing = {}
        for var in harmonization_map.keys():
            if var in df_year.columns:
                missing_pct = (df_year[var].isna().sum() / len(df_year)) * 100
                if missing_pct == 100:
                    missing[var] = "100% missing (variable not in this year)"
        summary['missing_variables_by_year'][str(year)] = missing
    
    summary_file = output_dir / "harmonization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Summary: {summary_file}")
    
    print("\n" + "=" * 80)
    print("Harmonization Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total records: {summary['total_records']:,}")
    print(f"  Records by year: {summary['records_by_year']}")
    print(f"  Harmonized variables: {summary['variables_available']}")
    
    return df_combined, summary

if __name__ == "__main__":
    df_combined, summary = main()

