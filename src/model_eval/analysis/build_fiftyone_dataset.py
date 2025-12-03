# src/model_eval/analysis/build_fiftyone_dataset.py

from pathlib import Path

import fiftyone as fo
import fiftyone.core.labels as fol
import pandas as pd

from model_eval.data.loaders import load_gt_pred_all
from model_eval.data.metadata import apply_series_event_labels
from model_eval.config import (
    DATA_RAW_VIDEOS,
    GT_SOURCE,
    PRED_SOURCE,
    TARGET_WIDTH,
    TARGET_HEIGHT,
    FIFTYONE_DATASET_NAME,
    FIFTYONE_CLASS_NAME
)


def convert_xyxy_to_coco_rel(row) -> list[float]:
    """Convert absolute xyxy to relative [x, y, w, h] in [0, 1]."""
    w = TARGET_WIDTH
    h = TARGET_HEIGHT

    # Normalize X and Y by width and height
    x_min = row["x_min"] / w
    y_min = row["y_min"] / h
    # Normalize box width and height
    box_w = (row["x_max"] - row["x_min"]) / w
    box_h = (row["y_max"] - row["y_min"]) / h

    return [x_min, y_min, box_w, box_h]


def frame_to_filepath(row: pd.Series) -> Path:
    """
    Map (video_series, segment_id, frame) -> image path.

    Assumes:
      segment_id like '1470_0', global frame:
        1470 -> frame_0000.jpg, 1471 -> frame_0001.jpg, ...
      under data/raw/videos/<series>/<segment_id>/images/
    """
    series = row["video_series"]
    segment_id = row["segment_id"]

    start_frame = int(str(segment_id).split("_")[0])
    local_idx = int(row["frame"]) - start_frame

    return (
        DATA_RAW_VIDEOS
        / str(series)
        / str(segment_id)
        / "images"
        / f"frame_{local_idx:04d}.jpg"
    )


def build_fiftyone_dataset(
    name: str = FIFTYONE_DATASET_NAME,
) -> fo.Dataset:
    """
    Build a persistent FiftyOne dataset with:
      - sample fields: video_series, segment_id, frame, event_id, full_label
      - label fields:  gt (Detections), predictions (Detections)
    """
    df = load_gt_pred_all()

    required = {
        "video_series",
        "segment_id",
        "frame",
        "source",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in gt_pred_all.csv: {sorted(missing)}")

    # Keep only frames that have GT, and propagate GT segment_id to PRED
    df_gt_frames = (
        df[df["source"] == GT_SOURCE][["video_series", "frame", "segment_id"]]
        .dropna(subset=["segment_id"])
        .drop_duplicates()
    )

    df = df.merge(
        df_gt_frames,
        on=["video_series", "frame"],
        how="inner",
        suffixes=("", "_gt"),
    )
    df["segment_id"] = df["segment_id_gt"]
    df = df.drop(columns=["segment_id_gt"])

    # Add label metadata: series_label, event_label, full_label
    meta = df[["video_series", "segment_id"]].drop_duplicates()
    meta = apply_series_event_labels(meta)

    df = df.merge(
        meta[["video_series", "segment_id", "full_label"]],
        on=["video_series", "segment_id"],
        how="left",
    )

    # event_id = "<series>_<segment_id>" for easier filtering in the app
    df["event_id"] = df["video_series"].astype(str) + "_" + df["segment_id"].astype(str)

    # Add filepath to each row
    df["filepath"] = df.apply(frame_to_filepath, axis=1)

    # Recreate dataset
    if name in fo.list_datasets():
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    dataset.persistent = True

    # One sample per unique image path
    for fp, group in df.groupby("filepath"):
        fp = Path(fp)
        first = group.iloc[0]

        sample = fo.Sample(filepath=str(fp))

        # Sample-level metadata for filtering
        sample["video_series"] = str(first["video_series"])
        sample["segment_id"] = str(first["segment_id"])
        sample["frame"] = int(first["frame"])
        sample["event_id"] = str(first["event_id"]) # <video_series>_<segment_id>
        # Event label: "V1E1"
        if "full_label" in first and not pd.isna(first["full_label"]):
            sample["event_label"] = str(first["full_label"])

        # GT detections
        gt_rows = group[group["source"] == GT_SOURCE]
        gt_dets = [
            fol.Detection(
                label=FIFTYONE_CLASS_NAME,
                bounding_box=convert_xyxy_to_coco_rel(r),
            )
            for _, r in gt_rows.iterrows()
        ]
        sample["gt"] = fol.Detections(detections=gt_dets)

        # Prediction detections
        pred_rows = group[group["source"] == PRED_SOURCE]
        pred_dets = [
            fol.Detection(
                label=FIFTYONE_CLASS_NAME,
                bounding_box=convert_xyxy_to_coco_rel(r),
                confidence=float(r.get("confidence", 0.0)),
                track_id=(
                    int(r["track_id"])
                    if "track_id" in r and not pd.isna(r["track_id"])
                    else None
                ),
            )
            for _, r in pred_rows.iterrows()
        ]
        sample["predictions"] = fol.Detections(detections=pred_dets)

        dataset.add_sample(sample)

    dataset.save()
    return dataset