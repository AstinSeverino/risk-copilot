"""Database query helpers for the Risk Copilot."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent.parent / "data" / "transactions.db"


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH))


def get_merchant(merchant_id: str) -> dict:
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM merchants WHERE merchant_id = ?", (merchant_id,)).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Merchant {merchant_id} not found")
    return dict(row)


def get_transactions(merchant_id: str, days: int = 90) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT * FROM transactions
           WHERE merchant_id = ?
             AND timestamp >= datetime((SELECT MAX(timestamp) FROM transactions WHERE merchant_id = ?), ?)
           ORDER BY timestamp""",
        conn,
        params=(merchant_id, merchant_id, f"-{days} days"),
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_peer_merchants(mcc_code: str, exclude_merchant: str = None) -> list[str]:
    conn = get_connection()
    if exclude_merchant:
        rows = conn.execute(
            "SELECT merchant_id FROM merchants WHERE mcc_code = ? AND merchant_id != ?",
            (mcc_code, exclude_merchant),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT merchant_id FROM merchants WHERE mcc_code = ?", (mcc_code,)
        ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_peer_daily_volumes(mcc_code: str, exclude_merchant: str = None, days: int = 7) -> pd.DataFrame:
    conn = get_connection()
    exclude_clause = f"AND t.merchant_id != '{exclude_merchant}'" if exclude_merchant else ""
    df = pd.read_sql_query(
        f"""SELECT t.merchant_id, DATE(t.timestamp) as day, COUNT(*) as txn_count,
                   AVG(t.amount) as avg_amount, SUM(t.amount) as total_amount
            FROM transactions t
            JOIN merchants m ON t.merchant_id = m.merchant_id
            WHERE m.mcc_code = ?
              AND t.timestamp >= datetime((SELECT MAX(timestamp) FROM transactions), ?)
            {exclude_clause}
            GROUP BY t.merchant_id, DATE(t.timestamp)""",
        conn,
        params=(mcc_code, f"-{days} days"),
    )
    conn.close()
    return df


def get_all_merchants() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM merchants", conn)
    conn.close()
    return df


def get_flagged_merchants(risk_scores: dict, threshold: float = 0.5) -> list[dict]:
    flagged = []
    for mid, score in risk_scores.items():
        if score >= threshold:
            merchant = get_merchant(mid)
            merchant["risk_score"] = score
            flagged.append(merchant)
    return sorted(flagged, key=lambda x: x["risk_score"], reverse=True)
