"""
Privacy and PII Protection module for UPI Scam Detector.

Provides robust, production-grade masking for sensitive Indian financial data:
  1. Indian mobile phone numbers (+91 / 10-digit mobile)
  2. Bank account numbers (9-18 digits)
  3. UPI IDs / VPAs (e.g. user@bank, 9876543210@paytm)
  4. Explicitly identified OTP, PIN, and CVV values

Rules:
  - In-memory ML inference preserves raw text for feature extraction.
  - Storage (SQLite, audit logs, user feedback, error logs) MUST strictly sanitize
    sensitive data before persistence.
  - Complete SMS bodies are NEVER stored by default; only minimal prediction
    metadata and sanitized feedback text are retained.
  - No encryption claims: Salted SHA-256 hashes are used for pseudonymous correlation.
"""
import re
import hashlib
from typing import Optional, Dict, Any

# --------------------------------------------------------------------------- Regex Patterns
# Indian Mobile numbers: Optional +91 / 0, starts with 6-9, followed by 9 digits
# Excludes 1800 toll-free numbers and 1930 helpline
PHONE_REGEX = re.compile(
    r"(?<!\d)(?:(?:\+91[\s-]?)?|0?)([6-9]\d{4})[\s-]?(\d{5})(?!\d)"
)

# Standalone 10-digit mobile number for sender field
EXACT_MOBILE_REGEX = re.compile(
    r"^(?:\+?91[\s-]?)?0?([6-9]\d{4})(\d{5})$"
)

# Full Bank account numbers (9 to 18 digits) preceded by account indicators or isolated sequences
ACCOUNT_INDICATOR_REGEX = re.compile(
    r"(?i)\b(?:a/?c(?:\s*(?:no\.?|ending|number))?|acct|account(?:\s*no\.?)?|khata)\s*[:#-]?\s*(\d{5,14})(\d{4})\b"
)
LONG_DIGIT_ACCOUNT_REGEX = re.compile(
    r"(?<![\d/.-])\b(\d{5,14})(\d{4})\b(?![\d/.-])"
)

# UPI ID / VPA: identifier@handle
UPI_VPA_REGEX = re.compile(
    r"\b([a-zA-Z0-9.\-_]{2,})@([a-zA-Z]{2,})\b"
)

# Explicitly identified OTP / Verification Code (4 to 8 digits)
# Pattern 1: Number before or after otp/code keywords
OTP_PREFIX_REGEX = re.compile(
    r"(?i)\b(?:otp|one[\s-]?time[\s-]?password|verification[\s-]?code|secret[\s-]?code|passcode)\b(?:[^\d\n]{0,35}?)\b(?:is|:|[-=])?\s*(\d{4,8})\b"
)
OTP_SUFFIX_REGEX = re.compile(
    r"(?i)\b(\d{4,8})\b\s*(?:is\s*(?:your|the)?\s*(?:secret\s*)?(?:otp|one[\s-]?time[\s-]?password|verification[\s-]?code|passcode))\b"
)

# Explicitly identified PIN or MPIN or CVV (3 to 6 digits)
PIN_EXPLICIT_REGEX = re.compile(
    r"(?i)\b(?:upi[\s-]?pin|mpin|atm[\s-]?pin|cvv2?|secret[\s-]?pin)\s*(?:is|:|[-=])?\s*(\d{3,6})\b"
)

# Known institutional short helpline numbers to preserve (never mask as PII)
PRESERVED_NUMBERS = {"1930", "112", "100", "1800"}


# --------------------------------------------------------------------------- Core Masking Functions
def mask_phone(phone: Optional[str]) -> str:
    """
    Mask a standalone phone number or sender ID.
    Examples:
      '+919876543210' -> '+91 98765*****'
      '9876543210'    -> '98765*****'
      'SBIINB'        -> 'SBIINB' (public business header, preserved)
    """
    if not phone:
        return ""
    clean = str(phone).strip()
    match = EXACT_MOBILE_REGEX.match(clean)
    if match:
        prefix = "+91 " if "+91" in clean else ""
        first_half = match.group(1)
        return f"{prefix}{first_half}*****"
    return clean


def mask_phone_numbers_in_text(text: str) -> str:
    """
    Mask all 10-digit Indian phone numbers found in free-form text.
    Retains the first 5 digits for traceability while obscuring the last 5 digits.
    Ignores official helpline 1930 or 1800 toll-free numbers.
    """
    if not text:
        return ""

    def _replace_phone(m: re.Match) -> str:
        first_five = m.group(1)
        full_match = m.group(0)
        has_plus_91 = "+91" in full_match
        prefix = "+91 " if has_plus_91 else ""
        return f"{prefix}{first_five}*****"

    return PHONE_REGEX.sub(_replace_phone, text)


def mask_bank_accounts_in_text(text: str) -> str:
    """
    Mask bank account numbers (9-18 digits).
    Preserves existing 4-digit masking formats (e.g., A/C ending 4321 or XX3456)
    while fully redacting raw 9-18 digit account numbers down to their last 4 digits.
    """
    if not text:
        return ""

    def _replace_acct_with_indicator(m: re.Match) -> str:
        prefix_chars = m.group(0)[:m.start(1) - m.start(0)]
        first_part = m.group(1)
        last_four = m.group(2)
        masked_part = "*" * len(first_part)
        return f"{prefix_chars}{masked_part}{last_four}"

    def _replace_isolated_digits(m: re.Match) -> str:
        full_digits = m.group(0)
        # Avoid masking 10-digit phone numbers (starts with 6-9) which are handled by phone masker
        if len(full_digits) == 10 and full_digits[0] in "6789":
            return full_digits
        # Avoid masking 12-digit phone numbers with country code (91[6-9]...)
        if len(full_digits) == 12 and full_digits.startswith("91") and full_digits[2] in "6789":
            return full_digits
        # Avoid masking 12-digit transaction references if preceded by ref/txn/upi
        start = max(0, m.start() - 25)
        context = text[start:m.start()].lower()
        if any(term in context for term in ["ref", "txn", "rrn", "utr", "id"]):
            return full_digits
        first_part = m.group(1)
        last_four = m.group(2)
        return f"{'*' * len(first_part)}{last_four}"

    # First pass: Explicit account indicators
    result = ACCOUNT_INDICATOR_REGEX.sub(_replace_acct_with_indicator, text)
    # Second pass: Long numeric sequences (11-18 digits typical of full Indian bank A/C)
    result = LONG_DIGIT_ACCOUNT_REGEX.sub(_replace_isolated_digits, result)
    return result


def mask_upi_ids_in_text(text: str) -> str:
    """
    Mask UPI handles (Virtual Payment Addresses).
    Examples:
      'rahul@okaxis'      -> 'r***l@okaxis'
      '9876543210@paytm'  -> '98765*****@paytm'
      'helpdesk@ybl'      -> 'h***k@ybl'
    """
    if not text:
        return ""

    def _replace_vpa(m: re.Match) -> str:
        user = m.group(1)
        handle = m.group(2)
        # If the username is a phone number
        if re.fullmatch(r"[6-9]\d{9}", user):
            masked_user = f"{user[:5]}*****"
        elif len(user) <= 2:
            masked_user = f"{user[0]}*"
        elif len(user) == 3:
            masked_user = f"{user[0]}*{user[-1]}"
        else:
            masked_user = f"{user[0]}***{user[-1]}"
        return f"{masked_user}@{handle}"

    return UPI_VPA_REGEX.sub(_replace_vpa, text)


def mask_otp_and_pins_in_text(text: str) -> str:
    """
    Redact explicitly stated OTPs, verification codes, and PINs.
    Preserves general dates, monetary amounts, and time references.
    """
    if not text:
        return ""

    # Replace 'OTP is 482910' -> 'OTP is [OTP_MASKED]'
    def _replace_otp_prefix(m: re.Match) -> str:
        full = m.group(0)
        otp = m.group(1)
        if otp in PRESERVED_NUMBERS:
            return full
        return full[:m.start(1) - m.start(0)] + "[OTP_MASKED]"

    # Replace '482910 is your OTP' -> '[OTP_MASKED] is your OTP'
    def _replace_otp_suffix(m: re.Match) -> str:
        full = m.group(0)
        otp = m.group(1)
        if otp in PRESERVED_NUMBERS:
            return full
        return "[OTP_MASKED]" + full[len(otp):]

    # Replace 'UPI PIN: 1234' -> 'UPI PIN: [PIN_MASKED]'
    def _replace_pin(m: re.Match) -> str:
        full = m.group(0)
        return full[:m.start(1) - m.start(0)] + "[PIN_MASKED]"

    result = OTP_PREFIX_REGEX.sub(_replace_otp_prefix, text)
    result = OTP_SUFFIX_REGEX.sub(_replace_otp_suffix, result)
    result = PIN_EXPLICIT_REGEX.sub(_replace_pin, result)
    return result


def mask_sensitive_data(text: str) -> str:
    """
    Comprehensive sanitization pipeline that runs all PII masking sequentially.
    Safe for storing in database logs, feedback comments, or debugging output.
    """
    if not text:
        return ""
    # 1. Redact OTPs and PINs first (before account or phone numbers can mistake them)
    text = mask_otp_and_pins_in_text(text)
    # 2. Redact UPI IDs
    text = mask_upi_ids_in_text(text)
    # 3. Redact Indian Mobile Numbers
    text = mask_phone_numbers_in_text(text)
    # 4. Redact Bank Account Numbers
    text = mask_bank_accounts_in_text(text)
    return text


def mask_sender(sender: str) -> str:
    """Convenience alias for masking sender IDs / phone numbers."""
    return mask_phone(sender)


def pseudonymize_identifier(value: str, salt: str = "upi_scam_detector_salt") -> str:
    """
    Deterministic pseudo-anonymous correlation identifier (truncated SHA-256).
    Allows grouping recurring malicious senders without ever storing their real identity.
    NOT encryption: strictly one-way cryptographic hashing.
    """
    if not value:
        return ""
    hasher = hashlib.sha256(f"{salt}:{value.strip().lower()}".encode("utf-8"))
    return f"id_{hasher.hexdigest()[:16]}"


def sanitize_log_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary prior to logging or storing.
    Ensures message, sender, or any nested string fields are masked.
    """
    sanitized = {}
    for k, v in data.items():
        if isinstance(v, str):
            if k in {"sender", "sender_id", "from"}:
                sanitized[k] = mask_sender(v)
            elif k in {"message", "text", "body", "sms", "feedback", "comment"}:
                sanitized[k] = mask_sensitive_data(v)
            else:
                sanitized[k] = mask_sensitive_data(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_log_dict(v)
        else:
            sanitized[k] = v
    return sanitized
