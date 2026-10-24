"""
Unit tests for privacy protection and PII masking (src/security.py).
"""
import pytest
from src.security import (
    mask_phone,
    mask_phone_numbers_in_text,
    mask_bank_accounts_in_text,
    mask_upi_ids_in_text,
    mask_otp_and_pins_in_text,
    mask_sensitive_data,
    mask_sender,
    pseudonymize_identifier
)


def test_mask_standalone_phone():
    assert mask_phone("+919876543210") == "+91 98765*****"
    assert mask_phone("9876543210") == "98765*****"
    assert mask_phone("+91-8765432109") == "+91 87654*****"
    # Preserves registered alphanumeric business headers
    assert mask_phone("SBIINB") == "SBIINB"
    assert mask_phone("HDFCBK") == "HDFCBK"
    assert mask_phone("GPAYIN") == "GPAYIN"
    # Empty handling
    assert mask_phone("") == ""
    assert mask_phone(None) == ""


def test_mask_phone_in_text():
    text = "Call bank officer at +919876543210 or 9123456780 immediately"
    masked = mask_phone_numbers_in_text(text)
    assert "+91 98765*****" in masked
    assert "91234*****" in masked
    assert "9876543210" not in masked
    assert "9123456780" not in masked


def test_preserve_helpline_and_toll_free():
    text = "Report cyber fraud at 1930 or call toll-free 18001080 immediately."
    masked = mask_phone_numbers_in_text(text)
    assert "1930" in masked
    assert "18001080" in masked


def test_mask_bank_account_numbers():
    # Long full account numbers
    text1 = "Your A/C no. 50100234567891 has been credited."
    masked1 = mask_bank_accounts_in_text(text1)
    assert "50100234567891" not in masked1
    assert "7891" in masked1
    assert "*" in masked1

    # Preserves existing masked format (e.g., A/C ending 4321)
    text2 = "A/C ending 4321 has avl bal Rs 5000"
    masked2 = mask_bank_accounts_in_text(text2)
    assert "4321" in masked2


def test_mask_upi_handles():
    text = "Send money to rahul.sharma@okaxis or 9876543210@paytm or support@ybl"
    masked = mask_upi_ids_in_text(text)
    assert "rahul.sharma@okaxis" not in masked
    assert "9876543210@paytm" not in masked
    assert "r***a@okaxis" in masked
    assert "98765*****@paytm" in masked
    assert "s***t@ybl" in masked


def test_mask_otp_and_pins():
    # Prefix format
    text1 = "Your OTP for login is 482910. Do not share."
    masked1 = mask_otp_and_pins_in_text(text1)
    assert "482910" not in masked1
    assert "[OTP_MASKED]" in masked1

    # Suffix format
    text2 = "392810 is your secret OTP for transaction."
    masked2 = mask_otp_and_pins_in_text(text2)
    assert "392810" not in masked2
    assert "[OTP_MASKED]" in masked2

    # UPI PIN
    text3 = "Enter your UPI PIN 4321 to approve request."
    masked3 = mask_otp_and_pins_in_text(text3)
    assert "4321" not in masked3
    assert "[PIN_MASKED]" in masked3


def test_preserve_ordinary_financial_text():
    # Standard transaction alert without explicit OTP/full account
    text = "Dear SBI User, your A/C ending 4321 is credited with INR 5,000.00 on 23-Oct. Avl Bal: INR 25,430.00. UPI Ref: 3298472918. Report at 1930 if not done by you."
    sanitized = mask_sensitive_data(text)
    
    # Verify non-PII terms remain untouched
    assert "Dear SBI User" in sanitized
    assert "INR 5,000.00" in sanitized
    assert "on 23-Oct" in sanitized
    assert "INR 25,430.00" in sanitized
    assert "UPI Ref: 3298472918" in sanitized
    assert "1930" in sanitized
    assert "ending 4321" in sanitized


def test_comprehensive_mask_sensitive_data():
    raw_scam = "Urgent: Transfer Rs 500 to 9876543210@paytm or call +919876543210. Your OTP is 123456 and A/C 12345678901234 is blocked."
    sanitized = mask_sensitive_data(raw_scam)
    
    # Nothing sensitive should remain in plaintext
    assert "9876543210" not in sanitized
    assert "123456" not in sanitized
    assert "12345678901234" not in sanitized
    
    # Masking tokens should be in place
    assert "[OTP_MASKED]" in sanitized
    assert "98765*****@paytm" in sanitized
    assert "+91 98765*****" in sanitized


def test_pseudonymize_identifier():
    h1 = pseudonymize_identifier("+919876543210")
    h2 = pseudonymize_identifier("+919876543210")
    h3 = pseudonymize_identifier("+919123456780")
    
    assert h1.startswith("id_")
    assert h1 == h2  # Deterministic for same sender
    assert h1 != h3  # Unique for different senders
    assert "9876543210" not in h1  # Non-reversible hash prefix
