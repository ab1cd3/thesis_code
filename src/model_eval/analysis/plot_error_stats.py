# src/model_eval/analysis/error_plots.py

from pathlib import Path
import matplotlib.pyplot as plt

from model_eval.analysis.error_stats import compute_error_counts_per_series
from model_eval.config import (
    REPORTS_ANALYSIS_FIGURES,
    ERROR_COLORS,
    IOU_THRESHOLD
)


def plot_error_type_stacked_by_series(
    rename_labels: bool = True,
    iou_thresh: float = IOU_THRESHOLD,
    save_path: Path | None = None
) -> None:
    """
    Stacked barplot of normalized TP/FP/FN per series.

    X-axis: series (raw or series_label if rename_labels=True)
    Y-axis: proportion (TP+FP+FN = 1 per series)
    """
    df = compute_error_counts_per_series(iou_thresh=iou_thresh)

    if rename_labels and "series_label" in df.columns:
        x_col = "series_label"
        order_col = "series_idx" if "series_idx" in df.columns else x_col
    else:
        x_col = "video_series"
        order_col = x_col

    # Sort and set index to the label column
    df = df.sort_values(order_col).set_index(x_col)

    # Keep only tp/fp/fn and fill missing with 0
    counts = df[["tp", "fp", "fn"]].fillna(0)

    # Normalize rows so each series sums to 1
    totals = counts.sum(axis=1).replace(0, 1)
    counts_norm = counts.div(totals, axis=0)

    # Colors in fixed order
    cols = ["tp", "fp", "fn"]
    color_list = [ERROR_COLORS[k] for k in cols]

    ax = counts_norm[cols].plot(
        kind="bar",
        stacked=True,
        color=color_list,
        figsize=(max(6, int(len(counts_norm) * 0.8)), 5)
    )

    x_rotation = 0 if rename_labels else 90
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation, ha="right")
    ax.set_xlabel("Video series")
    ax.set_ylabel("Proportion of detections")
    ax.set_ylim(0, 1)
    ax.set_title(f"TP / FP / FN proportions per series (IoU ≥ {iou_thresh})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(["TP", "FP", "FN"], title="Type", loc="upper right")

    # Save
    if save_path is None:
        REPORTS_ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
        iou_str = f"{iou_thresh:.2f}".replace(".", "_")
        save_path = REPORTS_ANALYSIS_FIGURES / f"error_types_stacked_by_series_iou_{iou_str}.png"
    else:
        if save_path.is_dir():
            iou_str = f"{iou_thresh:.2f}".replace(".", "_")
            save_path = save_path / f"error_types_stacked_by_series_iou_{iou_str}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved error type stacked barplot to {save_path}")