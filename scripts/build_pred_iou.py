# scripts/build_pred_iou.py

import argparse
from model_eval.analysis.build_pred_iou import save_pred_iou_table


def main():
    parser = argparse.ArgumentParser(
        description="Compute IoU for each prediction and save as pred_iou.csv."
    )
    parser.add_argument(
        "--series",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of video_series to restrict (e.g. DC13112018-B).",
    )
    args = parser.parse_args()

    save_pred_iou_table(video_series=args.series)

if __name__ == "__main__":
    main()