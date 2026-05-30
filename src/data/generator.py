"""
Generates synthetic merchant + transaction data for the Risk Copilot demo.
100 merchants across 5 MCC codes, 90 days of transactions (~300K rows).
Injects hero anomaly: Café Aurora (M007) with 10x volume spike in last 3 days.
"""
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

DB_PATH = Path(__file__).parent.parent.parent / "data" / "transactions.db"

MCC_PROFILES = {
    "5812": {"desc": "Eating Places, Restaurants", "avg_ticket": 12, "daily_txns": 80, "std_mult": 0.3, "count": 30},
    "5411": {"desc": "Grocery Stores", "avg_ticket": 45, "daily_txns": 200, "std_mult": 0.2, "count": 25},
    "5541": {"desc": "Service Stations (Gas)", "avg_ticket": 35, "daily_txns": 150, "std_mult": 0.25, "count": 20},
    "5999": {"desc": "Miscellaneous Retail", "avg_ticket": 28, "daily_txns": 60, "std_mult": 0.35, "count": 15},
    "7995": {"desc": "Gambling, Betting", "avg_ticket": 75, "daily_txns": 40, "std_mult": 0.5, "count": 10},
}

CITIES = [
    ("Austin", "TX"), ("Miami", "FL"), ("New York", "NY"),
    ("Chicago", "IL"), ("Los Angeles", "CA"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Denver", "CO"), ("Seattle", "WA"), ("Atlanta", "GA"),
]

HERO_MERCHANT_ID = "M007"
HERO_NAME = "Café Aurora"
HERO_MCC = "5812"
HERO_CITY = ("Austin", "TX")
SPIKE_DAYS = 3
SPIKE_MULTIPLIER = 10

SUSPICIOUS_MERCHANTS = {
    "M089": {"type": "new_merchant_spike", "desc": "New merchant, sudden high volume"},
    "M067": {"type": "cnp_foreign_spike", "desc": "CNP + foreign card surge"},
    "M045": {"type": "ticket_spike", "desc": "Ticket size 5x normal"},
    "M078": {"type": "gambling_new", "desc": "New gambling merchant, irregular hours"},
    "M091": {"type": "few_customers_many_txns", "desc": "Few unique customers, many txns"},
}


def create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS merchants;

        CREATE TABLE merchants (
            merchant_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            mcc_code TEXT NOT NULL,
            mcc_description TEXT NOT NULL,
            city TEXT NOT NULL,
            state TEXT NOT NULL,
            registered_date TEXT NOT NULL,
            avg_monthly_volume INTEGER,
            risk_tier TEXT DEFAULT 'LOW'
        );

        CREATE TABLE transactions (
            txn_id TEXT PRIMARY KEY,
            merchant_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            card_type TEXT NOT NULL,
            is_card_present INTEGER NOT NULL,
            customer_id TEXT NOT NULL,
            is_international INTEGER NOT NULL,
            FOREIGN KEY (merchant_id) REFERENCES merchants(merchant_id)
        );

        CREATE INDEX idx_txn_merchant ON transactions(merchant_id);
        CREATE INDEX idx_txn_timestamp ON transactions(timestamp);
    """)


def generate_merchants() -> list[dict]:
    merchants = []
    merchant_idx = 1

    for mcc, profile in MCC_PROFILES.items():
        for i in range(profile["count"]):
            mid = f"M{merchant_idx:03d}"
            city, state = random.choice(CITIES)

            if mid == HERO_MERCHANT_ID:
                name = HERO_NAME
                city, state = HERO_CITY
                mcc_override = HERO_MCC
            else:
                name = fake.company()
                mcc_override = mcc

            reg_days_ago = random.randint(180, 1800)
            if mid in SUSPICIOUS_MERCHANTS and SUSPICIOUS_MERCHANTS[mid]["type"] in ("new_merchant_spike", "gambling_new"):
                reg_days_ago = random.randint(15, 60)

            merchants.append({
                "merchant_id": mid,
                "name": name,
                "mcc_code": mcc_override,
                "mcc_description": MCC_PROFILES[mcc_override]["desc"] if mcc_override in MCC_PROFILES else profile["desc"],
                "city": city,
                "state": state,
                "registered_date": (datetime.now() - timedelta(days=reg_days_ago)).strftime("%Y-%m-%d"),
                "avg_monthly_volume": int(profile["daily_txns"] * 30 * random.uniform(0.8, 1.2)),
                "risk_tier": "HIGH" if mcc == "7995" else "MEDIUM" if mcc == "5999" else "LOW",
            })
            merchant_idx += 1

    return merchants


def _generate_normal_day_txns(merchant: dict, date: datetime, profile: dict, txn_counter: list) -> list[dict]:
    base_daily = profile["daily_txns"]
    std = base_daily * profile["std_mult"]
    day_of_week = date.weekday()
    seasonal = 1.0 + 0.15 * (day_of_week >= 5)

    n_txns = max(5, int(random.gauss(base_daily * seasonal, std)))
    txns = []
    customer_pool_size = max(int(n_txns * 0.85), 10)

    for _ in range(n_txns):
        txn_counter[0] += 1
        hour = random.choices(range(24), weights=[
            1, 0, 0, 0, 0, 1, 3, 8, 12, 14, 15, 16,
            15, 14, 13, 14, 15, 16, 14, 10, 6, 4, 2, 1
        ])[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = date.replace(hour=hour, minute=minute, second=second)

        amount = max(0.50, random.gauss(profile["avg_ticket"], profile["avg_ticket"] * 0.4))

        txns.append({
            "txn_id": f"T{txn_counter[0]:07d}",
            "merchant_id": merchant["merchant_id"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "card_type": random.choice(["VISA", "MASTERCARD", "AMEX", "DISCOVER"]),
            "is_card_present": 1 if random.random() < 0.92 else 0,
            "customer_id": f"C{random.randint(1, customer_pool_size):05d}",
            "is_international": 1 if random.random() < 0.03 else 0,
        })
    return txns


def _generate_hero_spike_txns(merchant: dict, date: datetime, profile: dict, txn_counter: list) -> list[dict]:
    n_txns = int(profile["daily_txns"] * SPIKE_MULTIPLIER * random.uniform(0.9, 1.1))
    txns = []
    customer_pool_size = int(n_txns * 0.9)

    for _ in range(n_txns):
        txn_counter[0] += 1
        hour = random.choices(range(24), weights=[
            0, 0, 0, 0, 0, 2, 5, 12, 18, 20, 22, 22,
            20, 18, 16, 18, 20, 22, 18, 12, 6, 3, 1, 0
        ])[0]
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        ts = date.replace(hour=hour, minute=minute, second=second)

        amount = max(0.50, random.gauss(profile["avg_ticket"] * 0.85, profile["avg_ticket"] * 0.3))

        txns.append({
            "txn_id": f"T{txn_counter[0]:07d}",
            "merchant_id": merchant["merchant_id"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "card_type": random.choice(["VISA", "MASTERCARD", "AMEX", "DISCOVER"]),
            "is_card_present": 1,
            "customer_id": f"C{random.randint(1, customer_pool_size):05d}",
            "is_international": 0,
        })
    return txns


def _generate_suspicious_txns(merchant: dict, date: datetime, profile: dict, sus_type: str, txn_counter: list) -> list[dict]:
    txns = []
    base = profile["daily_txns"]

    if sus_type == "new_merchant_spike":
        n_txns = int(base * 6)
        customer_pool = max(int(n_txns * 0.7), 10)
    elif sus_type == "cnp_foreign_spike":
        n_txns = int(base * 3)
        customer_pool = max(int(n_txns * 0.4), 5)
    elif sus_type == "ticket_spike":
        n_txns = base
        customer_pool = max(int(n_txns * 0.8), 10)
    elif sus_type == "gambling_new":
        n_txns = int(base * 4)
        customer_pool = max(int(n_txns * 0.3), 5)
    elif sus_type == "few_customers_many_txns":
        n_txns = int(base * 8)
        customer_pool = 5
    else:
        return _generate_normal_day_txns(merchant, date, profile, txn_counter)

    for _ in range(n_txns):
        txn_counter[0] += 1
        hour = random.randint(0, 23)
        ts = date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

        if sus_type == "ticket_spike":
            amount = max(1.0, random.gauss(profile["avg_ticket"] * 5, profile["avg_ticket"] * 2))
        else:
            amount = max(0.50, random.gauss(profile["avg_ticket"], profile["avg_ticket"] * 0.4))

        is_cp = 1
        is_intl = 0
        if sus_type == "cnp_foreign_spike":
            is_cp = 1 if random.random() < 0.3 else 0
            is_intl = 1 if random.random() < 0.45 else 0

        txns.append({
            "txn_id": f"T{txn_counter[0]:07d}",
            "merchant_id": merchant["merchant_id"],
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "card_type": random.choice(["VISA", "MASTERCARD", "AMEX", "DISCOVER"]),
            "is_card_present": is_cp,
            "customer_id": f"C{random.randint(1, customer_pool):05d}",
            "is_international": is_intl,
        })
    return txns


def generate_transactions(merchants: list[dict], days: int = 90) -> list[dict]:
    all_txns = []
    txn_counter = [0]
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)

    for merchant in merchants:
        mid = merchant["merchant_id"]
        mcc = merchant["mcc_code"]
        profile = MCC_PROFILES[mcc]

        for day_offset in range(days):
            date = start_date + timedelta(days=day_offset)
            is_spike_period = day_offset >= (days - SPIKE_DAYS)

            if mid == HERO_MERCHANT_ID and is_spike_period:
                txns = _generate_hero_spike_txns(merchant, date, profile, txn_counter)
            elif mid in SUSPICIOUS_MERCHANTS and is_spike_period:
                sus_type = SUSPICIOUS_MERCHANTS[mid]["type"]
                txns = _generate_suspicious_txns(merchant, date, profile, sus_type, txn_counter)
            else:
                txns = _generate_normal_day_txns(merchant, date, profile, txn_counter)

            all_txns.extend(txns)

    return all_txns


def save_to_db(merchants: list[dict], transactions: list[dict]):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    create_tables(conn)

    conn.executemany(
        "INSERT INTO merchants VALUES (?,?,?,?,?,?,?,?,?)",
        [(m["merchant_id"], m["name"], m["mcc_code"], m["mcc_description"],
          m["city"], m["state"], m["registered_date"], m["avg_monthly_volume"], m["risk_tier"])
         for m in merchants]
    )

    conn.executemany(
        "INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?)",
        [(t["txn_id"], t["merchant_id"], t["timestamp"], t["amount"],
          t["card_type"], t["is_card_present"], t["customer_id"], t["is_international"])
         for t in transactions]
    )

    conn.commit()
    conn.close()


def main():
    print("Generating merchants...")
    merchants = generate_merchants()
    print(f"  Created {len(merchants)} merchants across {len(MCC_PROFILES)} MCC codes")

    print("Generating transactions (90 days)...")
    transactions = generate_transactions(merchants)
    print(f"  Created {len(transactions):,} transactions")

    print(f"Saving to {DB_PATH}...")
    save_to_db(merchants, transactions)

    conn = sqlite3.connect(str(DB_PATH))
    hero_count = conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE merchant_id = ? AND timestamp >= date('now', '-3 days')",
        (HERO_MERCHANT_ID,)
    ).fetchone()[0]
    total_merchants = conn.execute("SELECT COUNT(*) FROM merchants").fetchone()[0]
    conn.close()

    print(f"\nHero anomaly: {HERO_NAME} ({HERO_MERCHANT_ID}) has {hero_count} txns in last 3 days")
    print(f"Suspicious merchants injected: {list(SUSPICIOUS_MERCHANTS.keys())}")
    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    main()
