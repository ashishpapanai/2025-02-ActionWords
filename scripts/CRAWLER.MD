# GESIS Variable Crawler
## Running the Crawling Script

To collect variable information from the GESIS dataset, use the `gesis_variable_crawler.py` script.

### Prerequisites
```bash
pip install -r requirements.txt
```

### Usage
```bash
python gesis_variable_crawler.py --study_number <STUDY_NUMBER> [OPTIONS]
```

### Arguments
- `--study_number` (required): The GESIS study number (e.g., `ZA10000` for ISSP 2022)
- `--output_dir` (optional): Directory to save the crawled data (default: `./data/`)
- `--format` (optional): Output format - `csv`, `json`, or `excel` (default: `csv`)
- `--verbose` (optional): Enable detailed logging

### Example
```bash
python gesis_variable_crawler.py --study_number ZA10000 --output_dir ./data --format csv --verbose
```

This will download all variable metadata from the ISSP 2022 dataset and save it to the specified directory.

---