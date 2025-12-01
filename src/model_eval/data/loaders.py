# src/model_eval/data/loaders.py

from pathlib import Path
from typing import List
import pandas as pd

from model_eval.config import DATA_STD_PRED, DATA_STD_GT, DATA_PROCESSED


def get_std_gt_series_list() -> List[str]:
    """
    Return list of video_series names that have standardized GT data.
    Assumes one folder per series under DATA_STD_GT.
    """
    if not DATA_STD_GT.exists():
        return []

    return sorted(
        d.name for d in DATA_STD_GT.iterdir() if d.is_dir()
    )


def get_std_pred_series_list() -> List[str]:
    """
    Return list of video_series names that have standardized predictions.
    Assumes one <series>.csv per series under DATA_STD_PRED.
    """
    if not DATA_STD_PRED.exists():
        return []

    return sorted(
        p.stem for p in DATA_STD_PRED.glob("*.csv")
    )


def load_std_gt_for_series(video_series: str) -> pd.DataFrame:
    """
    Load standardized GT for one series from:
        DATA_STD_GT / <series> / <series>_all.csv
    """
    series_dir = DATA_STD_GT / video_series
    csv_path = series_dir / f"{video_series}_all.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Standardized GT file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_std_pred_for_series(video_series: str) -> pd.DataFrame:
    """
    Load standardized predictions for one series from:
        DATA_STD_PRED / <series>.csv
    """
    csv_path = DATA_STD_PRED / f"{video_series}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Standardized predictions file not found: {csv_path}")

    return pd.read_csv(csv_path)


def load_std_gt_all_series() -> pd.DataFrame:
    """
    Load standardized GT for all series and concatenate.
    """
    series_list = get_std_gt_series_list()
    dfs = []
    for series in series_list:
        df = load_std_gt_for_series(series)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_std_pred_all_series() -> pd.DataFrame:
    """
    Load standardized predictions for all series and concatenate.
    """
    series_list = get_std_pred_series_list()
    dfs = []
    for series in series_list:
        df = load_std_pred_for_series(series)
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def load_gt_pred_all() -> pd.DataFrame:
    """
    Load processed combined GT+PRED csv file.
    """
    csv_path = DATA_PROCESSED / "gt_pred_all.csv"
    return pd.read_csv(csv_path)


def load_pred_iou() -> pd.DataFrame:
    """
    Load precomputed IoU table for predictions: data/processed/pred_iou.csv
    """
    path = DATA_PROCESSED / "pred_iou.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"pred_iou.csv not found at {path}. Run scripts/build_pred_iou.py first."
        )
    return pd.read_csv(path)