from model_eval.validation.check_data import (
    check_series_have_gt_and_pred,
    check_gt_segments_have_video_folders,
)


def main():
    print("Running data checks...\n")

    check_series_have_gt_and_pred()
    check_gt_segments_have_video_folders()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()