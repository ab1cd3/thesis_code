# src/model_eval/preprocessing/pred_standardization.py

from pathlib import Path
from typing import Tuple
import pandas as pd

from model_eval.config import (
    DATA_RAW_PRED,
    DATA_STD_PRED,
    TARGET_WIDTH,
    TARGET_HEIGHT
)

def parse_original_size(line: str) -> Tuple[float, float]:
    """
    Parse the first line 'original_size = 3840, 2160;' and return (width, height).
    """
    line = line.strip()
    try:
        size_part = line.split("=", 1)[1].strip().rstrip(";")
        width_str, height_str = [s.strip() for s in size_part.split(",")]
        return float(width_str), float(height_str)
    except Exception as e:
        raise ValueError(f"Cannot parse original size from line: {line}") from e


def standardize_predictions_for_series(video_series: str) -> None:
    """
    Standardize predictions for one video series:
    - Read raw txt file
    - Parse original size from first line
    - Parse prediction lines
    - Scale bbox to target size
    - Save standardized CSV in data/standardized/predictions
    """
    raw_path = DATA_RAW_PRED / f"{video_series}.txt"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw predictions file not found: {raw_path}")

    DATA_STD_PRED.mkdir(parents=True, exist_ok=True)

    rows = []
    with raw_path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    # First line: original size
    orig_w, orig_h = parse_original_size(first_line)

    # Remaining lines: predictions
    df = pd.read_csv(
        raw_path,
        sep=";",
        skiprows=1,
        header=None,
        names=["frame", "confidence", "track_id", "bbox"]
    )

    if df.empty:
        raise ValueError(f"Raw predictions file empty for series: {video_series}")

    bbox_cols = df["bbox"].str.split(",", expand=True)
    if bbox_cols.shape[1] != 4:
        raise ValueError(
            f"Unexpected bbox format in {raw_path}. "
            f"Expected 4 values, got {bbox_cols.shape[1]}."
        )
    df[["x_min", "y_min", "x_max", "y_max"]] = bbox_cols.astype(float)
    df = df.drop(columns=["bbox"])

    # Scale
    sx = TARGET_WIDTH / orig_w
    sy = TARGET_HEIGHT / orig_h
    df["x_min"] *= sx
    df["y_min"] *= sy
    df["x_max"] *= sx
    df["y_max"] *= sy

    df["video_series"] = video_series

    df = df[["video_series", "frame", "confidence", "track_id", "x_min", "y_min", "x_max", "y_max"]]

    # Save
    out_path = DATA_STD_PRED / f"{video_series}.csv"
    df.to_csv(out_path, index=False)
    print(f"[PRED] Saved standardized predictions to {out_path}")


def standardize_all_predictions() -> None:
    """
    Standardize predictions for all video series found in data/raw/predictions (format: .txt).
    """
    if not DATA_RAW_PRED.exists():
        raise FileNotFoundError(f"Predictions raw directory not found: {DATA_RAW_PRED}.")

    for txt_path in sorted(DATA_RAW_PRED.glob("*.txt")):
        video_series = txt_path.stem
        print(f"[PRED] Standardizing series: {video_series}")
        standardize_predictions_for_series(video_series)