# scripts/compute_error_stats_2.py

from model_eval.analysis.error_stats_2 import (
    compute_error_count_by_event,
    validate_error_stats
)
from model_eval.analysis.error_stats import compute_error_counts_per_series

def main():
    df_v2 = compute_error_count_by_event()
    print(df_v2)

    df_v1 = compute_error_counts_per_series()

    validate_error_stats(df_v1, df_v2)

if __name__ == "__main__":
    main()