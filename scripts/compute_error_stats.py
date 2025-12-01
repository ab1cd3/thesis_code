# scripts/compute_error_stats.py

import argparse

from model_eval.analysis.error_stats import (
    compute_error_counts_per_event,
    compute_error_counts_per_series,
)
from model_eval.config import IOU_THRESHOLD


def main():
    parser = argparse.ArgumentParser(
        description="Compute TP/FP/FN error counts for MODEL vs GT."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=IOU_THRESHOLD,
        help=f"IoU threshold for TP (default: {IOU_THRESHOLD})"
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional video series name(s) to filter (e.g. DC13112018-B)."
    )
    parser.add_argument(
        "--segment",
        type=str,
        nargs="*",
        default=None,
        help="Optional segment_id(s) to filter when level='event' (e.g. 1470_0)."
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["event", "series"],
        default="series",
        help="Aggregation level: 'event' or 'series' (default: series)."
    )

    args = parser.parse_args()

    if args.level == "event":
        df = compute_error_counts_per_event(
            iou_thresh=args.iou,
            video_series=args.series,
            segment_id=args.segment
        )
    else:
        df = compute_error_counts_per_series(
            iou_thresh=args.iou,
            video_series=args.series
        )

    print(f"\nERROR STATS (IoU = {args.iou})")
    print(df)


if __name__ == "__main__":
    main()