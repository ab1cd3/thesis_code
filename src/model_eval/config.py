# src/model_eval/config.py

from .utils.paths import (
    DATA_RAW,
    DATA_STD,
    DATA_PROCESSED,
    DATA_METADATA,
    DATA_RAW_PRED,
    DATA_RAW_GT,
    DATA_RAW_VIDEOS,
    DATA_STD_PRED,
    DATA_STD_GT,
    REPORTS_ANALYSIS_FIGURES,
    REPORTS_ANALYSIS_TABLES
)

from .utils.constants import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    GT_SOURCE,
    PRED_SOURCE,
    IOU_THRESHOLD,
    FIFTYONE_DATASET_NAME,
    FIFTYONE_CLASS_NAME
)

from .utils.colors import (
    GT_COLOR,
    PRED_COLOR,
    OVERLAY_GT_COLOR,
    OVERLAY_PRED_COLOR,
    ERROR_COLORS
)
