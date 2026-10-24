"""
Feature engineering for the UPI scam detector.

Three groups of hand-made features (plus TF-IDF, which lives in train.py):
  1. URL features      -> url_features()
  2. Sender features   -> sender_features()
  3. Signal features   -> signal_features()   (urgency, threat, OTP/PIN requests, English + Hinglish)

Everything is computed from just two raw inputs: `message` and `sender`,
so training and inference use exactly the same code path.
"""
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- constants
SHORTENERS = {"bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "t.ly", "is.gd", "goo.gl", "ow.ly"}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "online", "site", "info", "live", "tk", "ml", "ga", "cf", "gq", "icu", "buzz"}

# brand keyword -> used to spot lookalike / spoofed domains
BRAND_KEYWORDS = ["sbi", "hdfc", "icici", "axis", "pnb", "kotak", "baroda", "paytm", "phonepe", "gpay", "googlepay",
                  "npci", "irctc", "incometax", "epfo"]
OFFICIAL_DOMAINS = {"sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com", "pnbindia.in",
                    "kotak.com", "bankofbaroda.in", "paytm.com", "phonepe.com", "pay.google.com", "google.com",
                    "npci.org.in", "irctc.co.in", "incometax.gov.in", "epfindia.gov.in"}

# SMS sender IDs (headers) that real banks use. Extend this list with ones you see in your own inbox.
KNOWN_HEADER_CODES = {"SBIINB", "SBIPSG", "HDFCBK", "ICICIB", "AXISBK", "PNBSMS", "KOTAKB", "BOBSMS",
                      "PAYTMB", "PHONEPE", "GPAYIN", "IRCTCI", "UIDAIK"}
BANK_WORDS = ["sbi", "hdfc", "icici", "axis", "pnb", "kotak", "bank of baroda", "paytm", "phonepe", "google pay", "gpay"]

URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+|(?:bit\.ly|tinyurl\.com|cutt\.ly|rb\.gy|t\.ly|is\.gd|t\.me|wa\.me)/[^\s]+", re.I)
PHONE_RE = re.compile(r"(?<![\d,])(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")
MONEY_RE = re.compile(r"(?:rs\.?|inr|₹)\s?[\d,]+", re.I)

LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "@": "a", "$": "s"})

# --------------------------------------------------------------------------- keyword lists
# Matched on the *normalised* text (lowercase, leetspeak undone). Add your own words freely.
SIGNALS = {
    "urgency": [r"immediately", r"urgent", r"today", r"tonight", r"within \d+ ?(?:hours|hrs)", r"last chance",
                r"right now", r"act fast", r"turant", r"abhi", r"jaldi", r"aaj raat", r"\baaj\b", r"ghante"],
    "threat": [r"block", r"suspend", r"deactivat", r"disconnect", r"account (?:will be )?clos", r"legal action",
               r"band ho", r"kat jayeg", r"kat jayega", r"cut today", r"will be cut", r"\bsuspended\b"],
    "reward": [r"refund", r"cashback", r"prize", r"\bwon\b", r"lottery", r"lucky draw", r"reward", r"congratulations",
               r"badhai", r"jeet", r"\bkbc\b"],
    "credential_request": [r"\botp\b", r"\bpin\b", r"\bcvv\b", r"password", r"card number", r"debit card",
                           r"\bpan\b", r"aadhaar", r"aadhar", r"upi pin"],
    "kyc": [r"\bkyc\b", r"verify", r"verification", r"update your", r"update karo", r"update karein"],
    "call_to_action": [r"\bcall\b", r"whatsapp", r"telegram", r"click", r"claim", r"sampark", r"join"],
    "collect_trick": [r"collect request", r"approve", r"accept the money", r"by mistake", r"galti se", r"request accept"],
    "job_bait": [r"part ?time", r"work from home", r"earn rs", r"daily rs", r"per day", r"roz kamaye", r"ghar baithe",
                 r"/day"],
    "fee_request": [r"processing fee", r"registration fee", r"pay rs"],
    # signals that push toward SAFE
    "safety_warning": [r"do not share", r"never share", r"don'?t share", r"never asks", r"nahi maangta", r"savdhan",
                       r"beware", r"report at 1930"],
    "txn_alert": [r"debited", r"credited", r"\ba/c\b", r"avl bal", r"upi ref", r"\bdebit hue\b", r"is your otp"],
}
_COMPILED = {k: [re.compile(p) for p in v] for k, v in SIGNALS.items()}


# --------------------------------------------------------------------------- helpers
def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_urls(text: str) -> list:
    return [u.rstrip(".,)!?;:") for u in URL_RE.findall(text or "")]


def normalize_text(text: str) -> str:
    """Lowercase, undo leetspeak (bl0cked -> blocked), and replace URLs / numbers by placeholders.
    Used both for TF-IDF and for keyword matching."""
    text = str(text or "").lower()
    text = URL_RE.sub(" urltoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    out = []
    for tok in text.split():
        bare = tok.strip(".,:;!?()[]'\"-")
        if re.fullmatch(r"x{2,}\d+", bare):
            out.append("acctnum")
        elif re.fullmatch(r"[\d,./:-]+", bare) and any(c.isdigit() for c in bare):
            out.append("numtoken")
        elif sum(c.isalpha() for c in bare) >= 3 and any(c in "0136@$" for c in bare):
            out.append(bare.translate(LEET))
        else:
            out.append(tok)
    return " ".join(out)


# --------------------------------------------------------------------------- feature groups
def url_features(message: str) -> dict:
    urls = extract_urls(message)
    f = dict(has_url=0, n_urls=len(urls), url_len=0, url_is_shortener=0, url_suspicious_tld=0, url_n_hyphens=0,
             url_n_digits=0, url_subdomain_depth=0, url_no_https=0, url_is_ip=0, url_brand_spoof=0,
             url_lookalike=0, url_official=0, url_n_keywords=0)
    if not urls:
        return f
    u = urls[0]
    full = u if re.match(r"^https?://", u, re.I) else "http://" + u
    try:
        p = urlparse(full)
        host = (p.hostname or "").lower()
    except ValueError:
        host, p = "", urlparse("http://invalid")
    parts = host.split(".")
    tld = parts[-1] if parts else ""
    core = parts[-2] if len(parts) >= 2 else host
    official = any(host == d or host.endswith("." + d) for d in OFFICIAL_DOMAINS)
    brand_in_host = any(b in host for b in BRAND_KEYWORDS)
    f.update(
        has_url=1,
        url_len=len(u),
        url_is_shortener=int(host in SHORTENERS),
        url_suspicious_tld=int(tld in SUSPICIOUS_TLDS),
        url_n_hyphens=host.count("-"),
        url_n_digits=sum(c.isdigit() for c in host),
        url_subdomain_depth=max(len(parts) - 2, 0),
        url_no_https=int(p.scheme != "https"),
        url_is_ip=int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        url_official=int(official),
        url_brand_spoof=int(brand_in_host and not official),
        url_lookalike=int(any(0 < edit_distance(core, b) <= 2 and len(core) > 3 for b in BRAND_KEYWORDS)
                          and not official),
        url_n_keywords=sum(k in full.lower() for k in ["login", "verify", "update", "secure", "kyc", "refund", "claim"]),
    )
    return f


def sender_features(sender: str, message: str = "") -> dict:
    s = str(sender or "").strip().upper()
    is_mobile = bool(re.fullmatch(r"(?:\+?91)?[6-9]\d{9}", s))
    header = re.fullmatch(r"[A-Z]{2}-([A-Z0-9]{3,8})", s)
    code = header.group(1) if header else ""
    known = code in KNOWN_HEADER_CODES
    mentions_bank = any(b in str(message).lower() for b in BANK_WORDS)
    return dict(
        sender_missing=int(s == ""),
        sender_is_mobile=int(is_mobile),
        sender_known_header=int(known),
        sender_unknown_header=int(bool(header) and not known),
        # "SBI" message coming from a random number is a classic scam pattern
        bank_msg_from_mobile=int(is_mobile and mentions_bank),
    )


def signal_features(message: str) -> dict:
    norm = normalize_text(message)
    raw = str(message or "")
    f = {f"sig_{name}": int(any(p.search(norm) for p in pats)) for name, pats in _COMPILED.items()}
    letters = [c for c in raw if c.isalpha()]
    f.update(
        sig_phone_in_body=int(bool(PHONE_RE.search(raw))),
        sig_has_money=int(bool(MONEY_RE.search(raw))),
        sig_caps_ratio=round(sum(c.isupper() for c in letters) / len(letters), 3) if letters else 0.0,
        sig_exclaims=min(raw.count("!"), 5),
        sig_msg_len=min(len(raw), 600) / 600,
        # combos that are especially suspicious
        sig_urgency_and_threat=int(f["sig_urgency"] and f["sig_threat"]),
        sig_asks_secret_and_link=int(f["sig_credential_request"] and bool(URL_RE.search(raw))),
    )
    return f


def compute_features(message: str, sender: str = "") -> dict:
    feats = {}
    feats.update(url_features(message))
    feats.update(sender_features(sender, message))
    feats.update(signal_features(message))
    return feats


NUMERIC_COLS = list(compute_features("hello", "").keys())


def numeric_features_from_df(df: pd.DataFrame) -> np.ndarray:
    """Used inside the sklearn pipeline. Expects columns: message, sender."""
    rows = [compute_features(m, s) for m, s in zip(df["message"], df["sender"])]
    return pd.DataFrame(rows, columns=NUMERIC_COLS).to_numpy(dtype=float)


# --------------------------------------------------------------------------- human-readable reasons
REASONS = {
    "url_is_shortener": "Link uses a URL shortener that hides the real destination",
    "url_brand_spoof": "Link mentions a bank/brand name but is not the official domain",
    "url_lookalike": "Link domain looks like a misspelled bank or brand name",
    "url_suspicious_tld": "Link uses a domain ending commonly seen in scams (.xyz, .top, .click ...)",
    "url_no_https": "Link is not secured with HTTPS",
    "sender_is_mobile": "Sender is a personal mobile number, not a registered sender ID",
    "bank_msg_from_mobile": "Message claims to be from a bank but came from a personal number",
    "sender_unknown_header": "Sender ID does not match any known bank header",
    "sig_urgency_and_threat": "Combines urgency with a threat (e.g. 'account blocked today')",
    "sig_urgency": "Creates artificial urgency",
    "sig_threat": "Threatens account block, suspension or disconnection",
    "sig_reward": "Offers a refund, cashback or prize",
    "sig_credential_request": "Asks for OTP, PIN, card or ID details",
    "sig_asks_secret_and_link": "Asks for sensitive details and includes a link",
    "sig_kyc": "Asks you to verify or update KYC",
    "sig_collect_trick": "Pushes you to 'approve' a request or accept money (you may actually be paying)",
    "sig_job_bait": "Offers easy money for a part-time/work-from-home job",
    "sig_fee_request": "Asks for an upfront fee",
    "sig_phone_in_body": "Asks you to call/contact a number inside the message",
}
SAFE_REASONS = {
    "sender_known_header": "Sent from a known bank sender ID",
    "url_official": "Link points to an official domain",
    "sig_safety_warning": "Contains a standard 'do not share OTP' safety warning",
    "sig_txn_alert": "Matches a normal transaction-alert format",
}


def reasons_from_features(feats: dict, max_items: int = 5):
    """Return (scam_reasons, safe_reasons) lists of readable sentences."""
    bad = [txt for key, txt in REASONS.items() if feats.get(key)]
    good = [txt for key, txt in SAFE_REASONS.items() if feats.get(key)]
    return bad[:max_items], good[:3]


if __name__ == "__main__":
    demo = "SBI ALERT: KYC expired. Your account will be bl0cked today. Verify now http://sbi-kyc-update.xyz/login"
    print(normalize_text(demo))
    f = compute_features(demo, "9876543210")
    print({k: v for k, v in f.items() if v})
    print(reasons_from_features(f))
