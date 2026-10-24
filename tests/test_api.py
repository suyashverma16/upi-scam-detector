"""
Integration and security tests for FastAPI endpoints (app/main.py).

Verifies:
  - Prediction accuracy on known scam and safe samples.
  - Zero-leakage API responses (no raw message, no raw sender phone).
  - SQLite persistence privacy: zero message body storage, masked sender, pseudonymized hash.
  - User feedback comment sanitization before database insertion.
"""
import sqlite3
import pytest
from fastapi.testclient import TestClient

from app.main import app, DB_PATH, init_db


@pytest.fixture(scope="module")
def client():
    # Ensure fresh DB schema
    init_db(DB_PATH)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["privacy_protection"] == "active"
    assert data["message_storage_policy"] == "zero_body_retention"


def test_predict_known_scam(client):
    payload = {
        "message": "SBI ALERT: Dear customer, your YONO account is blocked today due to pending KYC. Update immediately: http://sbi-kyc-verify.xyz/login",
        "sender": "+919876543210"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # ML Prediction validation
    assert data["label"] == "SCAM"
    assert data["is_scam"] is True
    assert data["score"] >= 0.5
    assert len(data["reasons"]) > 0

    # Privacy Protection verification in API Response
    assert data["masked_sender"] == "+91 98765*****"
    assert "+919876543210" not in str(data)  # Full number NEVER leaked
    assert "message" not in data  # Raw message NEVER echoed back


def test_predict_known_safe(client):
    payload = {
        "message": "Dear SBI User, your A/C ending 4321 is credited with INR 5,000.00 on 23-Oct. Avl Bal: INR 25,430.00. UPI Ref: 3298472918. Report at 1930 if not done by you.",
        "sender": "SBIINB"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["label"] == "SAFE"
    assert data["is_scam"] is False
    assert data["score"] < 0.5
    assert data["masked_sender"] == "SBIINB"  # Preserved public bank header


def test_sqlite_persistence_privacy(client):
    """Verify that complete SMS messages are NOT stored in SQLite and metadata is sanitized."""
    sensitive_phone = "+919123456789"
    sensitive_message = "Urgent: Approve collect request of Rs 4999 on PhonePe. Enter OTP 765432 for refund."
    
    payload = {
        "message": sensitive_message,
        "sender": sensitive_phone
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    pred_id = res.json()["prediction_id"]

    # Directly inspect SQLite database records
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 1. Verify table columns - ensure no 'message' or 'body' column exists
        cursor.execute("PRAGMA table_info(prediction_audit);")
        columns = [row["name"] for row in cursor.fetchall()]
        assert "message" not in columns
        assert "sms" not in columns
        assert "body" not in columns

        # 2. Verify the saved record
        cursor.execute("SELECT * FROM prediction_audit WHERE id = ?", (pred_id,))
        record = cursor.fetchone()
        assert record is not None
        
        # Masked sender should have asterisks
        assert record["masked_sender"] == "+91 91234*****"
        assert sensitive_phone not in record["masked_sender"]
        
        # Sender pseudonym should be a hashed identifier
        assert record["sender_pseudonym"].startswith("id_")
        assert sensitive_phone not in record["sender_pseudonym"]

        # Ensure the sensitive message content is not found anywhere in the record fields
        record_values = " ".join([str(v) for v in record])
        assert "765432" not in record_values
        assert sensitive_phone not in record_values
        assert "collect request" not in record_values


def test_feedback_sanitization(client):
    """Verify feedback comments containing PII are sanitized before storage in SQLite."""
    # First make a prediction to get an ID
    res = client.post("/predict", json={"message": "Meeting at 5 pm", "sender": "9876543210"})
    pred_id = res.json()["prediction_id"]

    # Submit feedback containing sensitive OTP, phone number, and account number
    feedback_payload = {
        "prediction_id": pred_id,
        "is_correct": False,
        "comment": "The fraudster called from +919876543210 asking for OTP 987123 for A/C no. 50100234567891"
    }
    fb_res = client.post("/feedback", json=feedback_payload)
    assert fb_res.status_code == 200
    fb_id = fb_res.json()["feedback_id"]

    # Verify SQLite record is sanitized
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback_records WHERE id = ?", (fb_id,))
        fb_record = cursor.fetchone()
        assert fb_record is not None
        
        stored_comment = fb_record["sanitized_comment"]
        # Raw PII MUST NOT be in the stored comment
        assert "9876543210" not in stored_comment
        assert "987123" not in stored_comment
        assert "50100234567891" not in stored_comment
        
        # Masked tokens MUST be present
        assert "+91 98765*****" in stored_comment
        assert "[OTP_MASKED]" in stored_comment
        assert "7891" in stored_comment


def test_invalid_predict_request(client):
    # Empty message should fail validation
    res = client.post("/predict", json={"message": "", "sender": "123"})
    assert res.status_code == 422
