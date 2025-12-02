# src/model_eval/utils/metrics.py

from typing import Iterable
import numpy as np
import pandas as pd

from model_eval.utils.bbox import iou_xyxy
from model_eval.config import IOU_THRESHOLD


def match_frame_predictions(
    gt_boxes: np.ndarray,
    df_pred_f: pd.DataFrame,
    iou_thresh: float = IOU_THRESHOLD
) -> pd.DataFrame:
    """
    Given GT boxes and a predictions dataframe for a single frame,
    perform greedy one-to-one matching (sorted by confidence).

    Returns a new dataframe with the same rows as df_pred_f plus:
      - best_iou: best IoU vs any GT in this frame
      - is_tp: True if matched to a GT (IoU >= iou_thresh), else False
    """
    n_gt = gt_boxes.shape[0]

    # If no preds: just return empty with the right columns
    if df_pred_f.empty:
        out = df_pred_f.copy()
        out["best_iou"] = np.nan
        out["is_tp"] = False
        return out

    # Sort preds by confidence descending
    df_pred_f = df_pred_f.sort_values("confidence", ascending=False).reset_index(drop=True)
    pred_boxes = df_pred_f[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
    n_pred = pred_boxes.shape[0]

    gt_matched = np.zeros(n_gt, dtype=bool)
    best_ious = np.zeros(n_pred, dtype=float)
    is_tp = np.zeros(n_pred, dtype=bool)

    for p_idx in range(n_pred):
        p_box = pred_boxes[p_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for g_idx in range(n_gt):
            if gt_matched[g_idx]:
                continue
            g_box = gt_boxes[g_idx]
            iou = iou_xyxy(g_box, p_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        best_ious[p_idx] = best_iou
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            is_tp[p_idx] = True
            gt_matched[best_gt_idx] = True

    out = df_pred_f.copy()
    out["best_iou"] = best_ious
    out["is_tp"] = is_tp
    return out


def normalize_filter_arg(arg: str | Iterable[str] | None) -> list[str] | None:
    """
    Helper to allow a single string or list as filter argument.
    """
    if arg is None:
        return None
    if isinstance(arg, str):
        return [arg]
    return list(arg)


def add_pr_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add precision and recall columns to a TP/FP/FN table.

    precision = tp / (tp + fp)  (NaN if tp+fp == 0)
    recall  = tp / (tp + fn)  (NaN if tp+fn == 0)
    """
    df = df.copy()
    tp = df["tp"].astype(float)
    fp = df["fp"].astype(float)
    fn = df["fn"].astype(float)

    prec_den = tp + fp
    rec_den = tp + fn

    df["precision"] = np.where(prec_den > 0, tp / prec_den, np.nan)
    df["recall"] = np.where(rec_den > 0, tp / rec_den, np.nan)

    return df