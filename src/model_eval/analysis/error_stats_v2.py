# src/model_eval/analysis/error_stats_v2.py

from typing import Tuple
import pandas as pd
import numpy as np

from model_eval.data.loaders import load_gt_pred_all
from model_eval.config import (
    GT_SOURCE,
    PRED_SOURCE,
    IOU_THRESHOLD
)
from model_eval.analysis.error_stats import compute_error_counts_per_series


def compute_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:

    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)

    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0

    return inter_area / denom


def match_boxes_by_frame(
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray,
        iou_thresh: float = IOU_THRESHOLD
) -> Tuple[int, int, int]:

    n_gt = gt_boxes.shape[0]
    n_pred = pred_boxes.shape[0]

    gt_matched = np.zeros(n_gt, dtype=bool)
    pred_matched = np.zeros(n_pred, dtype=bool)

    for pred_idx in range(n_pred):
        pred_box = pred_boxes[pred_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx in range(n_gt):
            if gt_matched[gt_idx]:
                continue

            gt_box = gt_boxes[gt_idx]

            iou = compute_iou_xyxy(gt_box, pred_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx > -1:
            gt_matched[best_gt_idx] = True
            pred_matched[pred_idx] = True

    tp = int(pred_matched.sum())    # predictions matched 1-to-1 to a GT box (IoU ≥ threshold)
    fp = int((~pred_matched).sum()) # predictions not matched to any GT (IoU < threshold or duplicates)
    fn = int((~gt_matched).sum())   # GT boxes not matched by any prediction

    return tp, fp, fn


def compute_error_count_by_event(
        aggregate_by_series: bool = True,
        iou_thresh: float = IOU_THRESHOLD
):

    df = load_gt_pred_all()

    df_pred = df[df["source"] == PRED_SOURCE]
    df_gt = df[df["source"] == GT_SOURCE]

    records: list[dict] = []

    for (series, segment), df_event_gt in df_gt.groupby(["video_series", "segment_id"]):

        frames = df_event_gt["frame"].unique()

        df_event_pred = df_pred[
            (df_pred["video_series"] == series) &
            (df_pred["frame"].isin(frames))
        ]

        tp_total = fp_total = fn_total = 0

        for frame in sorted(frames):
            gt_boxes = df_event_gt[df_event_gt["frame"] == frame][[
                "x_min", "y_min", "x_max", "y_max"]].to_numpy()
            pred_frame = df_event_pred[df_event_pred["frame"] == frame]

            n_gt = gt_boxes.shape[0]

            # No PRED (GT unmatched) -> FN * num gt boxes
            if pred_frame.empty:
                fn_total += n_gt
                continue

            # PRED (GT matched)
            pred_boxes = pred_frame.sort_values("confidence", ascending=False)[
                ["x_min", "y_min", "x_max", "y_max"]].to_numpy()

            # Compute TP and FP count for frame
            n_tp, n_fp, n_fn = match_boxes_by_frame(gt_boxes, pred_boxes, iou_thresh)

            tp_total += n_tp
            fp_total += n_fp
            fn_total += n_fn

        records.append({
            "video_series": series,
            "segment_id": segment,
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total
        }
        )

    df_stats = pd.DataFrame(records)

    if aggregate_by_series:
        return df_stats.groupby("video_series")[["tp", "fp", "fn"]].sum().reset_index()
    else:
        return df_stats


def validate_error_stats(
        df_v1: pd.DataFrame,
        df_v2: pd.DataFrame
):
    merged = df_v2.merge(
        df_v1,
        on="video_series",
        how="outer",
        suffixes=("_v2", "_v1"),
        validate="one_to_one"
    )

    for col in ["tp", "fp", "fn"]:
        merged[f"{col}_diff"] = abs(merged[f"{col}_v2"] - merged[f"{col}_v1"])

    diffs = merged[
        (merged["tp_diff"] != 0) |
        (merged["fp_diff"] != 0) |
        (merged["fn_diff"] != 0)
        ]

    if diffs.empty:
        print("[OK] ERROR STATS V1 and V2 match for all series.")
    else:
        print("[MISMATCH] Differences found between V1 and V2:")
        print(diffs)
