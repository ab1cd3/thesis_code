# src/model_eval/analysis/error_labels.py

from typing import Iterable
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.config import GT_SOURCE, PRED_SOURCE, IOU_THRESHOLD
from model_eval.utils.metrics import (
    normalize_filter_arg,
    match_frame_predictions
)


def compute_pred_labels_on_gt_frames(
    iou_thresh: float = IOU_THRESHOLD,
    video_series: str | Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Global view. For predictions on frames that contain GT, compute:
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

    vs_filter = normalize_filter_arg(video_series)

    df_gt = df[df["source"] == GT_SOURCE]
    df_pred = df[df["source"] == PRED_SOURCE]

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

        matched = match_frame_predictions(
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


def compute_event_frame_errors(
    video_series: str,
    segment_id: str,
    iou_thresh: float = IOU_THRESHOLD
) -> pd.DataFrame:
    """
    For a given event (video_series, segment_id), compute per-frame error labels.

    Returns one row per prediction, plus one synthetic FN point per GT frame
    that has at least one FN with confidence=0.0:
      video_series, segment_id, frame, confidence, best_iou, error_type
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
        "y_max",
        "confidence"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gt_pred_all.csv: {sorted(missing)}")

    # GT for this event (series + segment)
    df_gt_ev = df[
        (df["video_series"] == video_series)
        & (df["segment_id"] == segment_id)
        & (df["source"] == GT_SOURCE)
    ]

    if df_gt_ev.empty:
        raise ValueError(
            f"No GT data found for series={video_series}, segment_id={segment_id}."
        )

    # ALL predictions for this series (segment_id may be NaN on PRED rows)
    df_pred_series = df[
        (df["video_series"] == video_series)
        & (df["source"] == PRED_SOURCE)
    ]

    frames = sorted(df_gt_ev["frame"].unique())
    records: list[dict] = []

    for frame in frames:
        gt_f = df_gt_ev[df_gt_ev["frame"] == frame]
        pred_f = df_pred_series[df_pred_series["frame"] == frame]

        gt_boxes = gt_f[["x_min", "y_min", "x_max", "y_max"]].to_numpy()
        n_gt = gt_boxes.shape[0]

        # No preds in this frame -> all GT are FN
        if pred_f.empty:
            records.append(
                {
                    "video_series": video_series,
                    "segment_id": segment_id,
                    "frame": frame,
                    "confidence": 0.0,
                    "best_iou": 0.0,
                    "error_type": "fn"
                }
            )
            continue

        matched = match_frame_predictions(
            gt_boxes=gt_boxes,
            df_pred_f=pred_f,
            iou_thresh=iou_thresh
        )

        n_tp = int(matched["is_tp"].sum())
        n_fn = n_gt - n_tp

        # Add TP/FP predictions
        for _, r in matched.iterrows():
            records.append(
                {
                    "video_series": video_series,
                    "segment_id": segment_id,
                    "frame": frame,
                    "confidence": float(r["confidence"]),
                    "best_iou": float(r["best_iou"]),
                    "error_type": "tp" if r["is_tp"] else "fp"
                }
            )

        # If there are FNs in this frame, add FN marker at confidence=0
        if n_fn > 0:
            records.append(
                {
                    "video_series": video_series,
                    "segment_id": segment_id,
                    "frame": frame,
                    "confidence": 0.0,
                    "best_iou": 0.0,
                    "error_type": "fn"
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "video_series",
                "segment_id",
                "frame",
                "confidence",
                "best_iou",
                "error_type"
            ]
        )

    return pd.DataFrame(records)