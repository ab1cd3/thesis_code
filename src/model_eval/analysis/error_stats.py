# src/model_eval/analysis/error_stats.py

from typing import Tuple, Iterable
import numpy as np
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.data.metadata import apply_series_event_labels, apply_series_labels
from model_eval.utils.metrics import (
    match_frame_predictions,
    normalize_filter_arg,
    add_pr_columns,
)
from model_eval.config import GT_SOURCE, PRED_SOURCE, IOU_THRESHOLD


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

    vs_filter = normalize_filter_arg(video_series)
    seg_filter = normalize_filter_arg(segment_id)

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
            matched = match_frame_predictions(
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
    df_events = add_pr_columns(df_events)

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
    df_series = add_pr_columns(df_series)

    return df_series