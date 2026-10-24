
"""Evaluate the existing UPI scam detector on an Indian multilingual dataset."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.predict import predict

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "models" / "indian_dataset_benchmark.json"
DATASET_NAME = "karanverma19/Indian_Multilingual_Scam_Message_Dataset"


def evaluate(df, dataset_name):
    """Evaluate unique messages or all rows using the existing prediction function."""
    y_true = df["label"].map({"legit": 0, "scam": 1}).astype(int).tolist()
    y_pred = []
    scores = []
    sources = Counter()

    for message in df["message"].fillna("").astype(str):
        result = predict(message, sender="")
        y_pred.append(1 if result["is_scam"] else 0)
        scores.append(float(result["score"]))
        sources[result["model_source"]] += 1

    return {
        "dataset": dataset_name,
        "rows": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["legit", "scam"],
            output_dict=True,
            zero_division=0,
        ),
        "model_sources": dict(sources),
        "per_language": {},
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    df = dataset["train"].to_pandas()

    required = {"message", "label", "language"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[df["label"].isin(["scam", "legit"])].copy()
    df["message"] = df["message"].fillna("").astype(str).str.strip()
    df = df[df["message"] != ""].copy()

    print("Rows:", len(df))
    print("Label distribution:", dict(Counter(df["label"])))
    print("Language distribution:", dict(Counter(df["language"])))
    print("Unique messages:", df["message"].nunique())
    print("Duplicate rows:", len(df) - df["message"].nunique())

    # Full dataset: duplicates remain, so repeated templates affect the metrics.
    full_results = evaluate(df, "all_rows")

    # Unique messages only: keep one occurrence of each exact message.
    unique_df = df.drop_duplicates(subset=["message"]).copy()
    unique_results = evaluate(unique_df, "unique_messages")

    for language, group in df.groupby("language"):
        y_true = group["label"].map({"legit": 0, "scam": 1}).astype(int).tolist()
        preds = [
            1 if predict(message, sender="")["is_scam"] else 0
            for message in group["message"]
        ]
        full_results["per_language"][language] = {
            "rows": len(group),
            "accuracy": round(accuracy_score(y_true, preds), 4),
            "recall": round(recall_score(y_true, preds, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, preds, zero_division=0), 4),
        }

    output = {
        "dataset_name": DATASET_NAME,
        "note": (
            "Exploratory benchmark only. The dataset is small and contains "
            "duplicate messages. Unique-message results remove exact duplicates "
            "but do not establish real-world performance."
        ),
        "total_rows": len(df),
        "unique_messages": int(df["message"].nunique()),
        "duplicate_rows": int(len(df) - df["message"].nunique()),
        "full_dataset": full_results,
        "unique_messages_only": unique_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for result in (full_results, unique_results):
        print(f"\n=== {result['dataset']} ===")
        print("Rows:", result["rows"])
        print("Accuracy:", result["accuracy"])
        print("Precision:", result["precision"])
        print("Recall:", result["recall"])
        print("F1-score:", result["f1_score"])
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(result["confusion_matrix"])
        print("Model sources:", result["model_sources"])

    print("\n=== Per-language results (all rows) ===")
    for language, metrics in full_results["per_language"].items():
        print(language, metrics)

    print(f"\nMetrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
