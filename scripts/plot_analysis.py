# scripts/plot_analysis.py

from model_eval.analysis.plot_bbox_area import plot_bbox_area_violin_by_series
from model_eval.analysis.plot_total_frames_by_gt_events import plot_total_frames_by_gt_events
from model_eval.analysis.plot_total_frames_gt_vs_pred import plot_total_frames_gt_vs_pred_by_series
from model_eval.analysis.plot_error_stats import plot_error_type_stacked_by_series
from model_eval.analysis.plot_iou_vs_confidence import plot_iou_vs_confidence
from model_eval.analysis.plot_precision_recall import plot_precision_recall_lines_by_series
from model_eval.analysis.plot_iou_vs_confidence_by_error import plot_iou_vs_confidence_by_error
from model_eval.analysis.plot_event_confidence_timeline import plot_event_confidence_timeline

if __name__ == "__main__":
    plot_bbox_area_violin_by_series()
    plot_total_frames_by_gt_events()
    plot_total_frames_gt_vs_pred_by_series()
    plot_error_type_stacked_by_series()
    plot_iou_vs_confidence()
    plot_precision_recall_lines_by_series()
    plot_iou_vs_confidence_by_error()
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="2880_0",
        iou_thresh=0.3
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="2580_0",
        iou_thresh=0.3
    )
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="3480_0",
        iou_thresh=0.3
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="810_0",
        iou_thresh=0.3
    )
    plot_event_confidence_timeline(
        video_series="SC01032019-A",
        segment_id="5460_0",
        iou_thresh=0.3
    )
    plot_event_confidence_timeline(
        video_series="DC20190403_B",
        segment_id="450_0",
        iou_thresh=0.3
    )