# scripts/build_fiftyone_dataset.py

from model_eval.analysis.build_fiftyone_dataset import build_fiftyone_dataset


if __name__ == "__main__":
    ds = build_fiftyone_dataset()
    print(ds)