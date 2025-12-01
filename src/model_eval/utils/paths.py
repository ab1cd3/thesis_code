# src/model_eval/utils/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"
DATA_RAW = DATA_ROOT / "raw"
DATA_STD = DATA_ROOT / "standardized"
DATA_PROCESSED = DATA_ROOT / "processed"
DATA_METADATA = DATA_ROOT / "metadata"

# Raw paths
DATA_RAW_GT = DATA_RAW / "gt"
DATA_RAW_PRED = DATA_RAW / "predictions"
DATA_RAW_VIDEOS = DATA_RAW / "videos"

# Std paths
DATA_STD_GT = DATA_STD / "gt"
DATA_STD_PRED = DATA_STD / "predictions"

# Reports (outputs)
REPORTS_ANALYSIS = PROJECT_ROOT / "reports" / "analysis"
REPORTS_ANALYSIS_TABLES = REPORTS_ANALYSIS / "tables"
REPORTS_ANALYSIS_FIGURES = REPORTS_ANALYSIS / "figures"
