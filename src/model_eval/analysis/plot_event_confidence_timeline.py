# src/model_eval/analysis/plot_event_confidence_timeline.py

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from matplotlib.pyplot import legend

from model_eval.analysis.error_labels import compute_event_frame_errors
from model_eval.data.metadata import apply_series_event_labels
from model_eval.config import ERROR_COLORS, REPORTS_ANALYSIS_FIGURES, IOU_THRESHOLD


def plot_event_confidence_timeline(
    video_series: str,
    segment_id: str,
    iou_thresh: float = IOU_THRESHOLD,
    rename_labels: bool = True,
    save_path: Path | None = None
) -> None:
    """
    Confidence timeline plot for one event (video_series, segment_id).
    - x-axis: actual frame number
    - y-axis: confidence for TP/FP, small negative value for FN
    - color: ERROR_COLORS["tp"/"fp"/"fn"]
    """
    df = compute_event_frame_errors(
        video_series=video_series,
        segment_id=segment_id,
        iou_thresh=iou_thresh,
    )
    if df.empty:
        raise ValueError(
            f"No event frame errors for series={video_series}, segment_id={segment_id}."
        )

    # Title: optional renaming
    title_label = f"{video_series} / {segment_id}"
    if rename_labels:
        meta = apply_series_event_labels(
            df[["video_series", "segment_id"]].drop_duplicates()
        )
        full_label = meta.get("full_label", None)
        if full_label is not None:
            title_label = full_label.iloc[0]

    # FN slightly below zero for visibility
    df_plot = df.copy()
    fn_level = 0.05

    df_plot.loc[df_plot["error_type"] == "fn", "confidence"] = fn_level

    # Map error_type -> color
    colors = df_plot["error_type"].map(ERROR_COLORS).fillna("gray")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        df_plot["frame"],
        df_plot["confidence"],
        width=1.0,
        color=colors,
        alpha=0.8
    )

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Confidence")
    ax.set_title(f"Event timeline: {title_label} (IoU ≥ {iou_thresh:.2f})")
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        Patch(color=ERROR_COLORS["tp"], label="TP"),
        Patch(color=ERROR_COLORS["fp"], label="FP"),
        Patch(color=ERROR_COLORS["fn"], label="FN")
    ]
    ax.legend(handles=legend_handles, title="Error type")

    # Save
    if save_path is None:
        REPORTS_ANALYSIS_FIGURES.mkdir(parents=True, exist_ok=True)
        save_path = (
            REPORTS_ANALYSIS_FIGURES
            / f"event_conf_timeline_{video_series}_{segment_id}_iou_{iou_thresh:.2f}.png"
        )
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

    print(f"[PLOT] Saved event timeline confidence plot to {save_path}")