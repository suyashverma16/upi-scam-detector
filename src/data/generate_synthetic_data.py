"""
Synthetic data generator for the UPI scam detector.

Generates realistic Indian UPI scam and safe SMS messages with realistic
sender IDs, URLs, phone numbers, urgency/threat keywords, Hinglish expressions,
and legitimate bank alert formats.

Outputs:
  - data/messages.csv: synthetic training dataset (message, sender, label)
  - data/real_holdout.csv: curated benchmark test cases
"""
import os
import random
from pathlib import Path
import pandas as pd

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Banks & Brands
BANKS = [
    ("SBI", "SBIINB", "sbi.co.in", ["sbi-kyc-portal.xyz", "sbi-yono-update.top", "onlinesbi-secure.online", "sbi-verify.site"]),
    ("HDFC", "HDFCBK", "hdfcbank.com", ["hdfc-kyc-verify.xyz", "hdfc-netbanking.top", "hdfc-reward.live", "hdfc-security.click"]),
    ("ICICI", "ICICIB", "icicibank.com", ["icici-kyc-update.xyz", "icici-alert.online", "icicibank-secure.top", "icici-login.site"]),
    ("Axis", "AXISBK", "axisbank.com", ["axis-kyc-verify.top", "axisbank-support.online", "axis-netbanking.xyz"]),
    ("PNB", "PNBSMS", "pnbindia.in", ["pnb-kyc-online.xyz", "pnb-secure-verify.site", "pnb-update.top"]),
    ("Kotak", "KOTAKB", "kotak.com", ["kotak-811-kyc.xyz", "kotak-verify.online", "kotak-support.click"]),
    ("Paytm", "PAYTMB", "paytm.com", ["paytm-cashback-claim.xyz", "paytm-kyc-portal.online", "paytm-reward.top"]),
    ("PhonePe", "PHONEPE", "phonepe.com", ["phonepe-reward-claim.xyz", "phonepe-cashback.top", "phonepe-support.online"]),
    ("Google Pay", "GPAYIN", "pay.google.com", ["gpay-reward-claim.xyz", "googlepay-cashback.top", "gpay-bonus.site"]),
]

SHORTENERS = ["bit.ly/3xY9aK", "tinyurl.com/y4bc8x", "cutt.ly/upikyc", "rb.gy/9a2b1", "t.ly/sbi99"]
SUSPICIOUS_DOMAINS = [
    "http://kyc-update-fast.xyz/login",
    "http://pan-aadhaar-link.online/verify",
    "http://claim-cashback-now.site/reward",
    "http://electricity-bill-pay.top/update",
    "http://fast-loan-approved.live/apply",
]

MOBILE_NUMBERS = [f"+91{random.randint(6000000000, 9999999999)}" for _ in range(50)]


def generate_scam_messages(n: int = 1200) -> list:
    scams = []

    # 1. Bank KYC & Account Block Threats
    for _ in range(n // 4):
        bank, _, _, spoof_urls = random.choice(BANKS)
        mobile = random.choice(MOBILE_NUMBERS)
        spoof_url = random.choice(spoof_urls)
        templates = [
            f"Dear {bank} User, your account has been suspended due to pending KYC update. Click immediately {spoof_url} to avoid permanent deactivation.",
            f"{bank} ALERT: Your NetBanking access will be bl0cked today. Turant KYC update karein: {spoof_url}",
            f"Dear customer, your {bank} A/C will be closed within 24 hours. Update PAN & Aadhaar now: {spoof_url}",
            f"{bank} Urgent: Your debit card is deactivated. Update details immediately: {spoof_url} or call {random.choice(MOBILE_NUMBERS)}",
            f"Aapka {bank} khata band ho jayega aaj raat. Link par click karke update karein: {spoof_url}",
            f"{bank} Security Alert: Unauthorized login attempt detected. Verify your identity now: {spoof_url}",
        ]
        scams.append({"message": random.choice(templates), "sender": mobile, "label": 1})

    # 2. Collect Request / Accidental Money Tricks
    for _ in range(n // 4):
        mobile = random.choice(MOBILE_NUMBERS)
        amount = random.randint(1500, 25000)
        templates = [
            f"Collect request of Rs.{amount} received. Enter UPI PIN to receive money in your bank account.",
            f"Dear user, Rs.{amount} cashback approved! Click approve on collect request in PhonePe to accept the money.",
            f"Galti se Rs.{amount} aapke account me bhej diye hai. Please request accept karke refund karein.",
            f"You have received a payment request of Rs.{amount}. Open Paytm and enter UPI PIN to deposit into your account.",
            f"Refund authorization: Approve the collect request of Rs.{amount} to get your failed recharge refund instantly.",
        ]
        scams.append({"message": random.choice(templates), "sender": mobile, "label": 1})

    # 3. Lottery, Cashback, Prize, Tax Refund Scams
    for _ in range(n // 4):
        mobile = random.choice(MOBILE_NUMBERS)
        shortener = random.choice(SHORTENERS)
        amount = random.choice(["25,000", "50,000", "5,00,000", "25,00,000"])
        templates = [
            f"Congratulations! You won Rs.{amount} lottery in KBC lucky draw. Claim your prize immediately: {shortener}",
            f"Dear Customer, Income Tax refund of Rs. 14,850 approved. Update bank account details now: {shortener}",
            f"Diwali bumper reward! You won Rs.{amount} cashback from Google Pay. Click to claim: {shortener}",
            f"Badhai ho! Aapka number 25 Lakh ke lucky draw me chuna gaya hai. Sampark karein WhatsApp: {random.choice(MOBILE_NUMBERS)}",
            f"Exclusive reward: Claim your Amazon cashback of Rs. 3,500 today: {shortener}",
            f"You have 4,500 unredeemed reward points worth Rs. 4,500 expiring tonight. Redeem now: {shortener}",
        ]
        scams.append({"message": random.choice(templates), "sender": mobile, "label": 1})

    # 4. Part-time Job, Utility Disconnection, Loan Scams
    for _ in range(n // 4):
        mobile = random.choice(MOBILE_NUMBERS)
        pay = random.choice(["2000-5000", "3000", "1500-3000"])
        templates = [
            f"Work from home! Part time job offer: Earn Rs {pay} daily by liking YouTube videos. Ghar baithe roz kamaye. WhatsApp: {random.choice(MOBILE_NUMBERS)}",
            f"Dear consumer, your electricity power will be disconnected tonight at 9:30 PM due to unpaid bill. Bijli kat jayega. Call officer: {random.choice(MOBILE_NUMBERS)}",
            f"Pre-approved personal loan of Rs 5,00,000 at 0% interest. Pay Rs 999 registration fee to disburse: {random.choice(SUSPICIOUS_DOMAINS)}",
            f"Earn Rs 2500 per day working 1 hour on telegram. Daily payment via UPI. Join now: t.me/daily_upi_earn",
            f"Immediate job opening: Review products & earn daily rs 4000. No experience needed. Contact: {random.choice(MOBILE_NUMBERS)}",
        ]
        scams.append({"message": random.choice(templates), "sender": mobile, "label": 1})

    return scams


def generate_safe_messages(n: int = 1200) -> list:
    safe = []

    # 1. Genuine Bank Transaction Alerts (Credit/Debit)
    for _ in range(n // 3):
        bank, header, _, _ = random.choice(BANKS[:6])
        acct = random.randint(1000, 9999)
        amount = f"{random.randint(50, 45000):,}.00"
        bal = f"{random.randint(1000, 250000):,}.00"
        ref = random.randint(300000000000, 499999999999)
        date = f"{random.randint(1, 28)}-Oct-26"
        templates = [
            f"Dear Customer, your A/C ending {acct} is credited with INR {amount} on {date}. Avl Bal: INR {bal}. UPI Ref: {ref}. -{bank}",
            f"{bank} Alert: INR {amount} debited from A/C *{acct} on {date} to SWIGGY. Avl Bal: INR {bal}. UPI Txn: {ref}. Call 18001080 if not done by you.",
            f"Your A/C {acct} debited by Rs {amount} on {date}. UPI/P2A/{ref}/Merchant. Avl Bal Rs: {bal}. Report at 1930 if unauthorized. -{bank}",
            f"Dear {bank} user, INR {amount} transferred to Rahul via UPI ref {ref}. Available balance: INR {bal}.",
        ]
        safe.append({"message": random.choice(templates), "sender": header, "label": 0})

    # 2. Genuine OTPs & Security Warnings
    for _ in range(n // 3):
        bank, header, domain, _ = random.choice(BANKS)
        otp = random.randint(100000, 999999)
        templates = [
            f"{otp} is your secret OTP for UPI registration. Never share OTP or UPI PIN with anyone, including bank staff. {bank} never asks for OTP.",
            f"Do not share this OTP with anyone. Your OTP for login to {bank} NetBanking is {otp}. Valid for 5 mins.",
            f"{otp} is your verification code for {bank}. Bank never asks for your password, PIN or OTP. Savdhan rahein.",
            f"Dear Customer, {otp} is your OTP to link bank account. NEVER share your UPI PIN or OTP to receive money.",
            f"Your one time password is {otp}. Please do not disclose it to anyone. Report cyber fraud at cybercrime.gov.in or 1930.",
        ]
        safe.append({"message": random.choice(templates), "sender": header, "label": 0})

    # 3. Genuine Bank Info, Official Links & Normal Peer Messages
    for _ in range(n // 3):
        bank, header, domain, _ = random.choice(BANKS)
        mobile = random.choice(MOBILE_NUMBERS)
        templates = [
            (f"Dear Customer, update your KYC safely at your nearest branch or via our official portal https://{domain}. Beware of fraudulent links.", header),
            (f"{bank} Alert: Always verify merchant details before making payment. Never enter UPI PIN to receive money.", header),
            (f"Cheque no {random.randint(100000, 999999)} for Rs 8,500 has been cleared in your {bank} account. Avl Bal: Rs 54,200.", header),
            ("Bhai lunch ke paise gpay kar diye hai check kar lena.", mobile),
            ("Can you send me 500 on UPI? Will return tomorrow morning.", mobile),
            ("Payment done for electricity bill via Google Pay. Ref: 409823910283.", mobile),
            ("Dinner split: You owe 450 for yesterday. Send whenever free.", mobile),
        ]
        msg, snd = random.choice(templates)
        safe.append({"message": msg, "sender": snd, "label": 0})

    return safe


def generate_benchmark_holdout() -> list:
    """Curated, high-fidelity real-world test cases covering edge cases."""
    return [
        # Phishing with bank lookalike
        {"message": "SBI ALERT: Your YONO account is blocked today due to pending KYC. Update immediately: http://sbi-kyc-verify.xyz/login", "sender": "+919876543210", "label": 1},
        # Legitimate bank credit alert
        {"message": "Dear SBI User, your A/C ending 4321 is credited with INR 5,000.00 on 23-Oct. Avl Bal: INR 25,430.00. UPI Ref: 3298472918. Report at 1930 if not done by you.", "sender": "SBIINB", "label": 0},
        # Fake lottery reward scam
        {"message": "Congratulations! You won Rs.25,000 lottery from KBC! Click here to claim your reward: bit.ly/kbc-reward-claim", "sender": "9123456780", "label": 1},
        # Authentic Google Pay alert with warning
        {"message": "Sent Rs. 1500 to Rahul via Google Pay. UPI transaction ID: 409823190283. Never share your UPI PIN with anyone.", "sender": "GPAYIN", "label": 0},
        # Collect request scam
        {"message": "You received a payment request of Rs. 3,500 on PhonePe. Click approve and enter UPI PIN to receive cashback.", "sender": "+919811223344", "label": 1},
        # Real bank OTP
        {"message": "384910 is your OTP for transaction at Amazon. Never share OTP or UPI PIN with anyone. Bank never calls asking for OTP.", "sender": "HDFCBK", "label": 0},
        # Electricity bill disconnection threat
        {"message": "Dear customer, your electricity power will be cut tonight at 9:30 PM. Bijli kat jayega. Immediately pay bill: http://power-bill-update.online", "sender": "+919822334455", "label": 1},
        # Normal chat
        {"message": "Hey Rahul, sent 300 on paytm for the movie ticket. Please check.", "sender": "+919988776655", "label": 0},
        # Work from home job bait
        {"message": "Part time work from home: Earn Rs 2500 per day by rating hotels online. Daily payout via UPI. Contact telegram @hotel_reviews_hr", "sender": "+919711223344", "label": 1},
        # Legitimate awareness notification
        {"message": "Savdhan! Never enter your UPI PIN to receive money. Fraudsters send collect requests posing as buyers on OLX. -NPCI", "sender": "UIDAIK", "label": 0},
    ]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic dataset...")
    scam_data = generate_scam_messages(n=1200)
    safe_data = generate_safe_messages(n=1200)
    all_data = scam_data + safe_data
    random.shuffle(all_data)

    df_train = pd.DataFrame(all_data)
    messages_path = DATA_DIR / "messages.csv"
    df_train.to_csv(messages_path, index=False)
    print(f"Saved {len(df_train)} records to {messages_path} (Scam: {len(scam_data)}, Safe: {len(safe_data)})")

    # Real holdout benchmark
    holdout_path = DATA_DIR / "real_holdout.csv"
    df_holdout = pd.DataFrame(generate_benchmark_holdout())
    df_holdout.to_csv(holdout_path, index=False)
    print(f"Saved {len(df_holdout)} benchmark holdout cases to {holdout_path}")


if __name__ == "__main__":
    main()
