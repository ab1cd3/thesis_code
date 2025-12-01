# src/model_eval/preprocessing/gt_pred_join.py

import pandas as pd

from model_eval.data.loaders import (
    load_std_gt_all_series,
    load_std_pred_all_series
)
from model_eval.config import DATA_PROCESSED, GT_SOURCE, PRED_SOURCE

def build_processed_gt_pred() -> pd.DataFrame:
    """
    Build processed dataframe combining standardized GT and Pred files.
    Adds source labels but does NOT add derived metrics.
    Saves to data/processed.
    """
    df_gt = load_std_gt_all_series()
    df_pred = load_std_pred_all_series()

    if df_gt.empty:
        raise ValueError("No standardized GT data loaded.")

    if df_pred.empty:
        raise ValueError("No standardized prediction data loaded.")

    df_gt["source"] = GT_SOURCE
    df_pred["source"] = PRED_SOURCE

    df_all = pd.concat([df_gt, df_pred], ignore_index=True)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "gt_pred_all.csv"
    df_all.to_csv(out_path, index=False)

    print(f"[PROCESSED] Saved combined GT+Pred dataset to {out_path}")
    return df_all