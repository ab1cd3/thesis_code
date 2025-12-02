# scr/model_eval/analysis/plot_iou_vs_confidence_by_error.py

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from model_eval.analysis.error_labels import compute_pred_labels_on_gt_frames
from model_eval.config import (
    REPORTS_ANALYSIS_FIGURES,
    IOU_THRESHOLD,
    ERROR_COLORS
)


def plot_iou_vs_confidence_by_error(
        iou_thresh: float = IOU_THRESHOLD,
        save_path: Path | None = None
):
    """
    Scatterplot of IoU vs confidence for predictions on GT frames,
    colored by error type (TP / FP).
     - TP: IoU >= iou_thresh
     - FP: IoU < iou_thresh
    """
    df = compute_pred_labels_on_gt_frames(iou_thresh=iou_thresh)
    if df.empty:
        raise ValueError("No prediction labels found for IoU plot.")

    colors = df["error_type"].str.lower().map(ERROR_COLORS).fillna("gray")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        df["confidence"],
        df["iou"],
        c=colors,
        s=10,
        alpha=0.5
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("IoU (best vs GT per frame)")
    ax.set_title(f"IoU vs confidence by error type (TP: IoU ≥ {iou_thresh:.2f},  FP: IoU < {iou_thresh:.2f})")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Save
    if save_path is None:
        REPORTS_ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_ANALYSIS_FIGURES / f"iou_vs_confidence_by_error_iou_{iou_thresh:.2f}.png"
    else:
        if save_path.is_dir():
            save_path = save_path / f"iou_vs_confidence_by_error_iou_{iou_thresh:.2f}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved IoU vs confidence by error type scatterplot to {save_path}")