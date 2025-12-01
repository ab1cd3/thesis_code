# src/model_eval/analysis/plot_iou_vs_confidence.py

from pathlib import Path

import matplotlib.pyplot as plt

from model_eval.data.loaders import load_pred_iou
from model_eval.config import REPORTS_ANALYSIS_FIGURES


def plot_iou_vs_confidence(
    save_path: Path | None = None,
    drop_zero_iou: bool = False
) -> None:
    """
    Scatterplot of IoU vs confidence for all predictions in pred_iou.csv.

    X-axis: confidence
    Y-axis: IoU (best IoU vs any GT in that frame)

    If drop_zero_iou=True, filter out rows with iou == 0.0
    (frames with no overlap / no GT).
    """
    df = load_pred_iou()
    if df.empty:
        raise ValueError("pred_iou.csv is empty or not found. Build it first.")

    df = df[["confidence", "iou"]].dropna()

    if drop_zero_iou:
        df = df[df["iou"] > 0.0]

    if df.empty:
        raise ValueError("No valid (confidence, iou) pairs after filtering.")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        df["confidence"],
        df["iou"],
        s=10,
        alpha=0.3
    )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("IoU (best vs GT per frame)")
    ax.set_title("IoU vs confidence for all predictions")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    # Save
    if save_path is None:
        REPORTS_ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
        save_path = REPORTS_ANALYSIS_FIGURES / "iou_vs_confidence.png"
    else:
        if save_path.is_dir():
            save_path = save_path / "iou_vs_confidence.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved IoU vs confidence scatterplot to {save_path}")