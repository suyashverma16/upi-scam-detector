"""
FastAPI application for UPI Scam Detector with privacy-first architecture.

Features:
  - POST /predict: Evaluates SMS fraud risk while keeping raw input strictly in-memory.
  - POST /feedback: Collects user feedback with mandatory PII sanitization.
  - GET  /health: Service and model health status.
  - SQLite persistence:
      * Complete message bodies are NEVER stored by default.
      * Only sanitized metadata (masked sender, pseudonymous hash, score, label) is persisted.
      * Feedback text is sanitized before persistence.
  - PII-safe logging: Request bodies and raw inputs are never emitted to log streams.
"""
import time
import uuid
import sqlite3
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Local imports
from src.predict import predict
from src.security import (
    mask_sender,
    mask_sensitive_data,
    pseudonymize_identifier
)

# --------------------------------------------------------------------------- Logging configuration
# Set logging format that avoids capturing any payload data
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("upi_scam_detector.api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "audit.db"


# --------------------------------------------------------------------------- SQLite Storage
def init_db(db_path: Path = DB_PATH):
    """Initialize SQLite tables for privacy-compliant prediction and feedback metadata."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        # 1. Prediction Metadata: Minimal audit record (NO complete message body stored)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_audit (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                masked_sender TEXT,
                sender_pseudonym TEXT,
                label TEXT NOT NULL,
                score REAL NOT NULL,
                is_scam INTEGER NOT NULL,
                model_source TEXT NOT NULL,
                reasons_count INTEGER NOT NULL,
                latency_ms REAL NOT NULL
            );
        """)
        
        # 2. Feedback Table: Only sanitized user comments are stored
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_records (
                id TEXT PRIMARY KEY,
                prediction_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                is_correct INTEGER NOT NULL,
                sanitized_comment TEXT,
                FOREIGN KEY (prediction_id) REFERENCES prediction_audit(id)
            );
        """)
        conn.commit()


def save_prediction_audit(
    pred_id: str,
    timestamp: str,
    masked_sender: str,
    sender_pseudonym: str,
    label: str,
    score: float,
    is_scam: bool,
    model_source: str,
    reasons_count: int,
    latency_ms: float,
    db_path: Path = DB_PATH
):
    """Store minimum required prediction metadata. Complete message is NEVER stored."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_audit (
                id, timestamp, masked_sender, sender_pseudonym,
                label, score, is_scam, model_source, reasons_count, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            pred_id,
            timestamp,
            masked_sender,
            sender_pseudonym,
            label,
            score,
            int(is_scam),
            model_source,
            reasons_count,
            round(latency_ms, 2)
        ))
        conn.commit()


def save_feedback_record(
    feedback_id: str,
    prediction_id: str,
    timestamp: str,
    is_correct: bool,
    sanitized_comment: Optional[str],
    db_path: Path = DB_PATH
):
    """Store feedback with strictly sanitized comments."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback_records (
                id, prediction_id, timestamp, is_correct, sanitized_comment
            ) VALUES (?, ?, ?, ?, ?);
        """, (
            feedback_id,
            prediction_id,
            timestamp,
            int(is_correct),
            sanitized_comment
        ))
        conn.commit()


# --------------------------------------------------------------------------- Lifecycle & App
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup DB on startup
    init_db()
    logger.info("Database initialized with privacy-compliant metadata tables.")
    yield


app = FastAPI(
    title="UPI Scam Detector API",
    version="1.0.0",
    description="Privacy-preserving real-time UPI and SMS phishing detection service.",
    lifespan=lifespan
)


# --------------------------------------------------------------------------- Schemas
class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="SMS text content")
    sender: Optional[str] = Field(default="", max_length=100, description="Sender ID or phone number")

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Dear customer, your SBI account is blocked today due to pending KYC. Update now: http://sbi-kyc.xyz",
                "sender": "+919876543210"
            }
        }
    }


class PredictResponse(BaseModel):
    prediction_id: str
    label: str
    score: float
    is_scam: bool
    reasons: List[str]
    masked_sender: str
    model_source: str
    # Note: raw message, raw sender, and sensitive details are explicitly omitted


class FeedbackRequest(BaseModel):
    prediction_id: str = Field(..., description="UUID of the prediction being evaluated")
    is_correct: bool = Field(..., description="Whether the user agreed with the prediction")
    comment: Optional[str] = Field(default=None, max_length=1000, description="Optional user comment")


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str


# --------------------------------------------------------------------------- Endpoints
@app.post("/predict", response_model=PredictResponse)
def handle_predict(req: PredictRequest):
    start_time = time.perf_counter()
    pred_id = str(uuid.uuid4())
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. In-memory ML inference using raw input (preserves feature detection fidelity)
    try:
        result = predict(message=req.message, sender=req.sender or "")
    except Exception as exc:
        # Avoid logging exc or request body which might leak raw message/phone
        logger.error(f"Inference execution error for request id={pred_id}: {type(exc).__name__}")
        raise HTTPException(status_code=500, detail="Inference processing failed.")

    latency_ms = (time.perf_counter() - start_time) * 1000

    # 2. Privacy-first sanitization for storage
    masked_sender_val = mask_sender(req.sender or "")
    sender_pseudo = pseudonymize_identifier(req.sender or "")

    # 3. Persist only minimum audit metadata (RAW MESSAGE IS NOT STORED)
    try:
        save_prediction_audit(
            pred_id=pred_id,
            timestamp=now_iso,
            masked_sender=masked_sender_val,
            sender_pseudonym=sender_pseudo,
            label=result["label"],
            score=result["score"],
            is_scam=result["is_scam"],
            model_source=result["model_source"],
            reasons_count=len(result["reasons"]),
            latency_ms=latency_ms
        )
    except Exception as exc:
        logger.error(f"Failed to persist prediction audit id={pred_id}: {type(exc).__name__}")

    # 4. Construct privacy-safe response (raw message & raw secrets omitted)
    return PredictResponse(
        prediction_id=pred_id,
        label=result["label"],
        score=result["score"],
        is_scam=result["is_scam"],
        reasons=result["reasons"],
        masked_sender=masked_sender_val,
        model_source=result["model_source"]
    )


@app.post("/feedback", response_model=FeedbackResponse)
def handle_feedback(req: FeedbackRequest):
    feedback_id = str(uuid.uuid4())
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Sanitize user feedback comment before storage
    sanitized_comment = mask_sensitive_data(req.comment) if req.comment else None

    try:
        save_feedback_record(
            feedback_id=feedback_id,
            prediction_id=req.prediction_id,
            timestamp=now_iso,
            is_correct=req.is_correct,
            sanitized_comment=sanitized_comment
        )
    except Exception as exc:
        logger.error(f"Failed to store feedback id={feedback_id}: {type(exc).__name__}")
        raise HTTPException(status_code=500, detail="Failed to record feedback.")

    return FeedbackResponse(
        feedback_id=feedback_id,
        status="success",
        message="Feedback securely recorded."
    )


@app.get("/health")
def handle_health():
    model_file = PROJECT_ROOT / "models" / "model.joblib"
    db_file = DB_PATH
    return {
        "status": "healthy",
        "model_loaded": model_file.exists(),
        "database_connected": db_file.exists(),
        "privacy_protection": "active",
        "message_storage_policy": "zero_body_retention"
    }
