"""
Model training pipeline for UPI scam detector.

Steps:
  1. Load synthetic and holdout data (data/messages.csv, data/real_holdout.csv)
  2. Split: stratified train / test split
  3. Feature extraction:
     - TF-IDF on normalized text (n-grams 1-2)
     - Hand-engineered URL, sender, and signal features via numeric_features_from_df
  4. Train: base classifier (LogisticRegression / LinearSVC)
  5. Calibrate: probability calibration via CalibratedClassifierCV
  6. Evaluate: compute Accuracy, Precision, Recall, F1, ROC-AUC, Brier score
  7. Save: serialize pipeline to models/model.joblib and metrics to models/model_metrics.json
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

# Support running either as module or standalone
try:
    from src.features import normalize_text, numeric_features_from_df
except ImportError:
    from features import normalize_text, numeric_features_from_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def build_pipeline() -> Pipeline:
    """
    Construct the full end-to-end classification pipeline:
      - ColumnTransformer combining TF-IDF and hand-crafted features
      - CalibratedClassifierCV wrapping LogisticRegression
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=(1, 2),
                    max_features=3500,
                    sublinear_tf=True
                ),
                "message"
            ),
            (
                "numeric",
                FunctionTransformer(numeric_features_from_df, validate=False),
                ["message", "sender"]
            )
        ],
        remainder="drop"
    )

    base_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=5
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", calibrated_clf)
    ])

    return pipeline


def evaluate_model(pipeline: Pipeline, df_test: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive evaluation and calibration metrics."""
    X_test = df_test[["message", "sender"]]
    y_test = df_test["label"].to_numpy(dtype=int)

    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True, target_names=["SAFE", "SCAM"])

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probs)), 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }
    return metrics


def train():
    """Execute data loading, split, training, calibration, evaluation and export."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    messages_path = DATA_DIR / "messages.csv"

    if not messages_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {messages_path}. Run generate_synthetic_data.py first!"
        )

    logger.info(f"Loading training data from {messages_path}...")
    df = pd.read_csv(messages_path)
    df["message"] = df["message"].fillna("").astype(str)
    df["sender"] = df["sender"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)

    logger.info(f"Loaded {len(df)} records. Class distribution:\n{df['label'].value_counts().to_dict()}")

    # 1. Stratified Split
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    logger.info(f"Split completed: Train={len(df_train)}, Test={len(df_test)}")

    # 2. Build and Train Pipeline
    pipeline = build_pipeline()
    logger.info("Fitting feature preprocessor and calibrated classifier...")
    pipeline.fit(df_train[["message", "sender"]], df_train["label"])

    # 3. Evaluate on Test Split
    logger.info("Evaluating model on validation test set...")
    test_metrics = evaluate_model(pipeline, df_test)

    # 4. Evaluate on Real Holdout Set (if available)
    holdout_path = DATA_DIR / "real_holdout.csv"
    holdout_metrics = None
    if holdout_path.exists():
        logger.info(f"Evaluating model on real holdout set ({holdout_path})...")
        df_holdout = pd.read_csv(holdout_path)
        df_holdout["message"] = df_holdout["message"].fillna("").astype(str)
        df_holdout["sender"] = df_holdout["sender"].fillna("").astype(str)
        df_holdout["label"] = df_holdout["label"].astype(int)
        holdout_metrics = evaluate_model(pipeline, df_holdout)

    # 5. Save Artifacts
    model_save_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipeline, model_save_path)
    logger.info(f"Saved trained model to {model_save_path}")

    metrics_save_path = MODELS_DIR / "model_metrics.json"
    full_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "validation_metrics": test_metrics,
        "holdout_metrics": holdout_metrics
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_save_path}")

    print("\n" + "=" * 50)
    print("      UPI SCAM DETECTOR: TRAINING REPORT      ")
    print("=" * 50)
    print(f"Accuracy   : {test_metrics['accuracy']:.4f}")
    print(f"Precision  : {test_metrics['precision']:.4f}")
    print(f"Recall     : {test_metrics['recall']:.4f}")
    print(f"F1 Score   : {test_metrics['f1']:.4f}")
    print(f"ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    print(f"Brier Loss : {test_metrics['brier_score']:.4f} (calibration quality)")
    print("=" * 50)
    if holdout_metrics:
        print(f"Holdout Benchmark Accuracy : {holdout_metrics['accuracy']:.4f} (N={len(df_holdout)})")
        print(f"Holdout Benchmark F1 Score : {holdout_metrics['f1']:.4f}")
        print("=" * 50)

    return pipeline, full_metrics


if __name__ == "__main__":
    train()
