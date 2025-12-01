# src/model_eval/analysis/plot_total_frames_gt_vs_pred.py

from pathlib import Path

import matplotlib.pyplot as plt

from model_eval.data.loaders import load_gt_pred_all
from model_eval.data.metadata import apply_series_labels
from model_eval.config import (
    REPORTS_ANALYSIS_FIGURES,
    GT_SOURCE,
    PRED_SOURCE,
    GT_COLOR,
    PRED_COLOR
)


def plot_total_frames_gt_vs_pred_by_series(
    rename_labels: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Side-by-side bars of total frames with GT vs PRED detections per series.

    X-axis: series (raw or renamed).
    Y-axis: number of frames that contain at least one bbox (GT or PRED).
    """
    df = load_gt_pred_all()
    if df.empty:
        raise ValueError("No data loaded from gt_pred_all.csv.")

    # Count unique frames per (series, source)
    counts = (
        df.groupby(["video_series", "source"])["frame"]
        .nunique()
        .reset_index(name="frame_count")
    )

    # Add series labels (series_idx, series_label)
    if rename_labels:
        counts = apply_series_labels(counts)
        x_col = "series_label"
        order_col = "series_idx"
    else:
        x_col = "video_series"
        order_col = x_col

    # Pivot to wide format: index = series, columns = source (GT/PRED)
    wide = counts.pivot(index=x_col, columns="source", values="frame_count").fillna(0)

    # Order series
    series_order = (
        counts[[x_col, order_col]]
        .drop_duplicates()
        .sort_values(order_col)[x_col]
        .tolist()
    )
    wide = wide.reindex(series_order)

    x_labels = list(wide.index)
    x = list(range(len(x_labels)))
    width = 0.4

    gt_vals = wide.get(GT_SOURCE, 0)
    pred_vals = wide.get(PRED_SOURCE, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 0.8), 5))

    ax.bar(
        [i - width / 2 for i in x],
        gt_vals,
        width=width,
        color=GT_COLOR,
        alpha=0.8,
        label="GT",
    )
    ax.bar(
        [i + width / 2 for i in x],
        pred_vals,
        width=width,
        color=PRED_COLOR,
        alpha=0.8,
        label="PRED",
    )

    x_rotation = 0 if rename_labels else 90
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=x_rotation)
    ax.set_xlabel("Video series")
    ax.set_ylabel("Frames with detections")
    ax.set_title("Total frames with GT vs PRED bboxes per series")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Save
    if save_path is None:
        REPORTS_ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_ANALYSIS_FIGURES / "total_frames_gt_vs_pred_by_series.png"
    else:
        if save_path.is_dir():
            save_path = save_path / "total_frames_gt_vs_pred_by_series.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved total GT vs PRED frame count barplot to {save_path}")