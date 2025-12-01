# scripts/standardize_pred.py

import argparse

from model_eval.preprocessing.pred_standardization import (
    standardize_predictions_for_series,
    standardize_all_predictions
)


def main():
    parser = argparse.ArgumentParser(
        description="Standardize prediction txt files into CSV per video series."
    )
    parser.add_argument(
        "--series",
        type=str,
        default=None,
        help="Video series name (e.g. DC13112018-B). If omitted, process all.",
    )
    args = parser.parse_args()

    if args.series:
        standardize_predictions_for_series(args.series)
    else:
        standardize_all_predictions()


if __name__ == "__main__":
    main()