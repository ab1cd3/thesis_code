# src/model_eval/analysis/error_stats.py

from typing import Tuple, Iterable
import numpy as np
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.data.metadata import apply_series_event_labels, apply_series_labels
from model_eval.utils.bbox import iou_xyxy
from model_eval.config import GT_SOURCE, PRED_SOURCE, IOU_THRESHOLD


def _match_frame_predictions(
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


def _normalize_filter_arg(arg: str | Iterable[str] | None) -> list[str] | None:
    """
    Helper to allow a single string or list as filter argument.
    """
    if arg is None:
        return None
    if isinstance(arg, str):
        return [arg]
    return list(arg)


def _add_pr_columns(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_error_counts_per_event(
    iou_thresh: float = IOU_THRESHOLD,
    video_series: str | Iterable[str] | None = None,
    segment_id: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Compute TP/FP/FN per (video_series, segment_id).

    Events are defined from GT only:
      - Take GT rows -> groups of (video_series, segment_id)
      - For each event, find PRED rows in same series and same frames.
    Predictions on frames with no GT are ignored here (not counted).

    Optional filters:
      - video_series: single series name or list
      - segment_id: single segment_id or list (e.g. '1470_0')
    """
    df = load_gt_pred_all()
    if df.empty:
        raise ValueError("No data loaded from gt_pred_all.csv.")

    required = {
        "video_series",
        "segment_id",
        "frame",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gt_pred_all.csv: {sorted(missing)}")

    vs_filter = _normalize_filter_arg(video_series)
    seg_filter = _normalize_filter_arg(segment_id)

    # Split GT and PRED
    df_gt = df[df["source"] == GT_SOURCE]
    df_pred = df[df["source"] == PRED_SOURCE]

    if vs_filter is not None:
        df_gt = df_gt[df_gt["video_series"].isin(vs_filter)]
        df_pred = df_pred[df_pred["video_series"].isin(vs_filter)]

    if seg_filter is not None:
        df_gt = df_gt[df_gt["segment_id"].isin(seg_filter)]

    # Ensure GT has valid segments
    df_gt = df_gt[df_gt["segment_id"].notna()]

    if df_gt.empty:
        raise ValueError("No GT data left after filtering (series/segment).")

    records: list[dict] = []

    # Events defined from GT only
    for (series, seg), df_event_gt in df_gt.groupby(["video_series", "segment_id"]):
        frames = df_event_gt["frame"].unique()

        # PRED for same series and same frames
        df_event_pred = df_pred[
            (df_pred["video_series"] == series)
            & (df_pred["frame"].isin(frames))
        ]

        tp_total = fp_total = fn_total = 0

        for frame in sorted(frames):
            gt_f = df_event_gt[df_event_gt["frame"] == frame]
            pred_f = df_event_pred[df_event_pred["frame"] == frame]

            gt_boxes = gt_f[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
            n_gt = gt_boxes.shape[0]

            # No preds in this frame -> all GT are FN
            if pred_f.empty:
                fn_total += n_gt
                continue

            # Use shared helper
            matched = _match_frame_predictions(
                gt_boxes=gt_boxes,
                df_pred_f=pred_f,
                iou_thresh=iou_thresh,
            )

            n_tp = int(matched["is_tp"].sum())
            n_fp = int((~matched["is_tp"]).sum())
            n_fn = n_gt - n_tp

            tp_total += n_tp
            fp_total += n_fp
            fn_total += n_fn

        records.append(
            {
                "video_series": series,
                "segment_id": seg,
                "tp": tp_total,
                "fp": fp_total,
                "fn": fn_total
            }
        )

    df_events = pd.DataFrame(records)
    df_events = apply_series_event_labels(df_events)

    # Add precision - recall per event
    df_events = _add_pr_columns(df_events)

    return df_events


def compute_error_counts_per_series(
    iou_thresh: float = IOU_THRESHOLD,
    video_series: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Aggregate TP/FP/FN counts to per-series totals.
    Optional filter:
      - video_series: single name or list.
    """
    df_events = compute_error_counts_per_event(
        iou_thresh=iou_thresh,
        video_series=video_series,
        segment_id=None
    )

    df_series = (
        df_events.groupby("video_series")[["tp", "fp", "fn"]]
        .sum()
        .reset_index()
    )

    df_series = apply_series_labels(df_series)

    # Add precision / recall per series
    df_series = _add_pr_columns(df_series)

    return df_series


def compute_pred_labels_on_gt_frames(
    iou_thresh: float = IOU_THRESHOLD,
    video_series: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """
    For predictions on frames that contain GT, compute:
      - best_iou: best IoU vs any GT in that frame
      - error_type: "TP" (matched) or "FP" (unmatched)

    Uses the same greedy matching as compute_error_counts_per_event.
    """
    df = load_gt_pred_all()
    if df.empty:
        raise ValueError("No data loaded from gt_pred_all.csv.")

    required = {
        "video_series",
        "frame",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "confidence"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gt_pred_all.csv: {sorted(missing)}")

    vs_filter = _normalize_filter_arg(video_series)

    df_gt = df[df["source"] == GT_SOURCE].copy()
    df_pred = df[df["source"] == PRED_SOURCE].copy()

    if vs_filter is not None:
        df_gt = df_gt[df_gt["video_series"].isin(vs_filter)]
        df_pred = df_pred[df_pred["video_series"].isin(vs_filter)]

    records: list[dict] = []

    # Loop by (series, frame) where there ARE predictions
    for (series, frame), df_pred_f in df_pred.groupby(["video_series", "frame"]):
        df_gt_f = df_gt[
            (df_gt["video_series"] == series)
            & (df_gt["frame"] == frame)
        ]
        # Only frames that have GT
        if df_gt_f.empty:
            continue

        gt_boxes = df_gt_f[["x_min", "y_min", "x_max", "y_max"]].to_numpy()

        matched = _match_frame_predictions(
            gt_boxes=gt_boxes,
            df_pred_f=df_pred_f,
            iou_thresh=iou_thresh
        )

        for _, row in matched.iterrows():
            records.append(
                {
                    "video_series": series,
                    "frame": frame,
                    "track_id": row.get("track_id", None),
                    "confidence": float(row["confidence"]),
                    "iou": float(row["best_iou"]),
                    "error_type": "tp" if row["is_tp"] else "fp"
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "video_series",
                "frame",
                "track_id",
                "confidence",
                "iou",
                "error_type"
            ]
        )

    return pd.DataFrame(records)