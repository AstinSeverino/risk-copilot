"""Node 1: Collects merchant data, transaction history, and peer stats. Pure Python — no LLM."""
import pandas as pd

from src.data.database import get_merchant, get_transactions, get_peer_daily_volumes
from src.agents.state import AgentState
from src.agents.observability import traced


@traced(name="data_collector")
def data_collector(state: AgentState) -> dict:
    merchant_id = state["merchant_id"]
    merchant = get_merchant(merchant_id)
    txns = get_transactions(merchant_id, days=90)
    peer_volumes = get_peer_daily_volumes(merchant["mcc_code"], exclude_merchant=merchant_id, days=7)

    now = txns["timestamp"].max() if not txns.empty else pd.Timestamp.now()
    txns_24h = txns[txns["timestamp"] >= now - pd.Timedelta(hours=24)]
    txns_7d = txns[txns["timestamp"] >= now - pd.Timedelta(days=7)]

    transactions_summary = {
        "total_txns_90d": len(txns),
        "txn_count_24h": len(txns_24h),
        "txn_count_7d": len(txns_7d),
        "avg_amount_24h": round(float(txns_24h["amount"].mean()), 2) if len(txns_24h) > 0 else 0.0,
        "total_amount_24h": round(float(txns_24h["amount"].sum()), 2) if len(txns_24h) > 0 else 0.0,
        "unique_customers_24h": int(txns_24h["customer_id"].nunique()) if len(txns_24h) > 0 else 0,
        "pct_card_present": round(float(txns_24h["is_card_present"].mean()), 3) if len(txns_24h) > 0 else 1.0,
        "pct_international": round(float(txns_24h["is_international"].mean()), 3) if len(txns_24h) > 0 else 0.0,
        "date_range": f"{txns['timestamp'].min().date()} to {txns['timestamp'].max().date()}" if not txns.empty else "N/A",
    }

    if not peer_volumes.empty:
        peer_avg_daily = peer_volumes.groupby("merchant_id")["txn_count"].mean()
        peer_stats = {
            "peer_count": int(peer_volumes["merchant_id"].nunique()),
            "peer_avg_daily_txns": round(float(peer_avg_daily.mean()), 1),
            "peer_std_daily_txns": round(float(peer_avg_daily.std()), 1),
            "peer_median_daily_txns": round(float(peer_avg_daily.median()), 1),
        }
    else:
        peer_stats = {"peer_count": 0, "peer_avg_daily_txns": 0, "peer_std_daily_txns": 0, "peer_median_daily_txns": 0}

    return {
        "merchant_info": merchant,
        "transactions_summary": transactions_summary,
        "peer_stats": peer_stats,
        "reasoning_trace": [
            f"[DataCollector] Merchant: {merchant['name']} (MCC {merchant['mcc_code']}, {merchant['city']}). "
            f"Pulled {len(txns)} txns (90d), {len(txns_24h)} in last 24h. "
            f"{peer_stats['peer_count']} peers in same MCC."
        ],
    }
