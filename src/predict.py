"""
Inference module for the UPI scam detector.

Exposes:
  predict(message, sender) -> {"label": "SCAM"|"SAFE", "score": float, "reasons": list, ...}
"""
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import joblib

# Support running either as package (src.predict) or standalone script
try:
    from src.features import compute_features, reasons_from_features
    from src.security import mask_sender
except ImportError:
    from features import compute_features, reasons_from_features
    from security import mask_sender

logger = logging.getLogger(__name__)

# Default model location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# In-memory model cache
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None
_MODEL_LOOKUP_WARNED = False


def load_model(model_path: Optional[str] = None):
    """
    Load and cache the trained ML pipeline from disk.
    Returns None if the model file does not exist yet.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH, _MODEL_LOOKUP_WARNED
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == target_path:
        return _CACHED_MODEL

    if not target_path.exists():
        if not _MODEL_LOOKUP_WARNED:
            logger.info(f"Trained model not found at {target_path}. Using feature heuristic.")
            _MODEL_LOOKUP_WARNED = True
        return None

    try:
        model = joblib.load(target_path)
        _CACHED_MODEL = model
        _CACHED_MODEL_PATH = target_path
        logger.info(f"Loaded trained model from {target_path}")
        return _CACHED_MODEL
    except Exception as e:
        logger.error(f"Failed to load model from {target_path}: {e}")
        return None


def _heuristic_score(feats: Dict[str, Any]) -> float:
    """
    Fallback heuristic scorer used when model.joblib is not yet trained/available.
    Weights domain spoofing, urgency+threat, and credential harvesting patterns.
    """
    score = 0.15  # baseline prior probability

    # Critical scam signals
    if feats.get("bank_msg_from_mobile"):
        score += 0.35
    if feats.get("url_brand_spoof"):
        score += 0.35
    if feats.get("url_lookalike"):
        score += 0.30
    if feats.get("sig_asks_secret_and_link"):
        score += 0.30
    if feats.get("sig_urgency_and_threat"):
        score += 0.25
    if feats.get("url_suspicious_tld"):
        score += 0.20
    if feats.get("url_is_shortener"):
        score += 0.20
    if feats.get("sig_credential_request"):
        score += 0.20
    if feats.get("sig_collect_trick"):
        score += 0.25
    if feats.get("sig_job_bait"):
        score += 0.25
    if feats.get("sig_fee_request"):
        score += 0.20
    if feats.get("sig_kyc"):
        score += 0.15
    if feats.get("sig_reward"):
        score += 0.25
    if feats.get("sender_is_mobile"):
        score += 0.10
    if feats.get("sender_unknown_header"):
        score += 0.10
    if feats.get("sig_phone_in_body"):
        score += 0.10

    # Safe signals (reduce score)
    if feats.get("sender_known_header"):
        score -= 0.30
    if feats.get("url_official"):
        score -= 0.35
    if feats.get("sig_safety_warning"):
        score -= 0.25
    if feats.get("sig_txn_alert"):
        score -= 0.20

    return float(np.clip(score, 0.01, 0.99))


def predict(
    message: str,
    sender: str = "",
    model_path: Optional[str] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict whether a UPI SMS / notification is a SCAM or SAFE.

    Args:
        message: The message body text.
        sender: Sender ID or phone number (e.g., 'SBIINB', '+919876543210').
        model_path: Optional custom path to model.joblib.
        threshold: Decision boundary probability for SCAM (default: 0.5).

    Returns:
        Dict with keys:
          - label: 'SCAM' | 'SAFE'
          - score: probability of scam (0.0 to 1.0)
          - reasons: list of human-readable explanation sentences
          - is_scam: bool
          - model_source: 'ml_model' | 'heuristic_fallback'
    """
    message = str(message or "").strip()
    sender = str(sender or "").strip()

    # 1. Feature extraction & explanations
    feats = compute_features(message, sender)
    bad_reasons, good_reasons = reasons_from_features(feats)

    # 2. Try ML model inference
    model = load_model(model_path)
    score: float
    model_source: str

    if model is not None:
        try:
            df = pd.DataFrame([{"message": message, "sender": sender}])
            probs = model.predict_proba(df)[0]

            # Find index corresponding to scam (class 1 or 'SCAM'/'scam')
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                scam_idx = classes.index(1)
            elif "SCAM" in classes:
                scam_idx = classes.index("SCAM")
            elif "scam" in classes:
                scam_idx = classes.index("scam")
            else:
                scam_idx = 1 if len(classes) > 1 else 0

            score = float(probs[scam_idx])
            model_source = "ml_model"
        except Exception as e:
            logger.warning(f"ML inference error ({type(e).__name__}). Falling back to heuristic.")
            score = _heuristic_score(feats)
            model_source = "heuristic_fallback"
    else:
        score = _heuristic_score(feats)
        model_source = "heuristic_fallback"

    # 3. Decision & explanation selection
    is_scam = score >= threshold
    label = "SCAM" if is_scam else "SAFE"

    if is_scam:
        reasons = bad_reasons if bad_reasons else ["Suspicious message pattern detected"]
    else:
        reasons = good_reasons if good_reasons else ["No common scam patterns detected"]

    return {
        "label": label,
        "score": round(score, 4),
        "reasons": reasons,
        "is_scam": is_scam,
        "masked_sender": mask_sender(sender),
        "model_source": model_source,
        "safe_reasons": good_reasons,
        "scam_reasons": bad_reasons,
    }


if __name__ == "__main__":
    samples = [
        (
            "SBI ALERT: Dear customer, your YONO account is blocked today due to pending KYC. Update immediately: http://sbi-kyc-verify.xyz/login",
            "+919876543210"
        ),
        (
            "Dear SBI User, your A/C ending 4321 is credited with INR 5,000.00 on 23-Oct. Avl Bal: INR 25,430.00. UPI Ref: 3298472918. Report at 1930 if not done by you.",
            "SBIINB"
        ),
        (
            "Congratulations! You won Rs.25,000 lottery from KBC! Click here to claim your reward: bit.ly/kbc-reward-claim",
            "9123456780"
        ),
        (
            "Sent Rs. 1500 to Rahul via Google Pay. UPI transaction ID: 409823190283. Never share your UPI PIN with anyone.",
            "GPAYIN"
        ),
    ]

    print("=== UPI Scam Detector: Inference Test ===\n")
    for msg, snd in samples:
        res = predict(msg, snd)
        print(f"Sender : {snd}")
        print(f"Message: {msg}")
        print(f"Result : [{res['label']}] (Score: {res['score']:.4f}, Source: {res['model_source']})")
        print(f"Reasons: {res['reasons']}")
        print("-" * 70)
