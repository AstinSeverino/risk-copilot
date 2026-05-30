"""Feature engineering for merchant risk scoring."""
import numpy as np
import pandas as pd

from src.data.database import get_transactions, get_peer_daily_volumes, get_merchant

FEATURE_NAMES = [
    "txn_count_24h", "txn_count_7d", "txn_count_30d", "volume_growth_ratio",
    "avg_ticket_24h", "max_ticket_24h", "total_amount_24h",
    "unique_customers_24h", "customer_txn_ratio", "pct_card_present", "pct_international",
    "peer_volume_zscore", "peer_amount_zscore",
    "hour_entropy", "business_age_days",
]


def calculate_features(merchant_id: str) -> dict:
    merchant = get_merchant(merchant_id)
    txns = get_transactions(merchant_id, days=90)

    if txns.empty:
        return {f: 0.0 for f in FEATURE_NAMES}

    now = txns["timestamp"].max()
    txns_24h = txns[txns["timestamp"] >= now - pd.Timedelta(hours=24)]
    txns_7d = txns[txns["timestamp"] >= now - pd.Timedelta(days=7)]
    txns_30d = txns[txns["timestamp"] >= now - pd.Timedelta(days=30)]

    txn_count_24h = len(txns_24h)
    txn_count_7d = len(txns_7d)
    txn_count_30d = len(txns_30d)

    avg_daily_30d = txn_count_30d / 30.0 if txn_count_30d > 0 else 1.0
    volume_growth_ratio = txn_count_24h / avg_daily_30d if avg_daily_30d > 0 else 0.0

    avg_ticket_24h = txns_24h["amount"].mean() if len(txns_24h) > 0 else 0.0
    max_ticket_24h = txns_24h["amount"].max() if len(txns_24h) > 0 else 0.0
    total_amount_24h = txns_24h["amount"].sum() if len(txns_24h) > 0 else 0.0

    unique_customers_24h = txns_24h["customer_id"].nunique() if len(txns_24h) > 0 else 0
    customer_txn_ratio = unique_customers_24h / txn_count_24h if txn_count_24h > 0 else 0.0

    pct_card_present = txns_24h["is_card_present"].mean() if len(txns_24h) > 0 else 1.0
    pct_international = txns_24h["is_international"].mean() if len(txns_24h) > 0 else 0.0

    peer_vol_z, peer_amt_z = _calculate_peer_zscores(merchant_id, merchant["mcc_code"], txn_count_24h, avg_ticket_24h)

    hour_entropy = _calculate_hour_entropy(txns_24h)

    from datetime import datetime
    reg_date = datetime.strptime(merchant["registered_date"], "%Y-%m-%d")
    business_age_days = (datetime.now() - reg_date).days

    features = {
        "txn_count_24h": float(txn_count_24h),
        "txn_count_7d": float(txn_count_7d),
        "txn_count_30d": float(txn_count_30d),
        "volume_growth_ratio": float(volume_growth_ratio),
        "avg_ticket_24h": float(avg_ticket_24h),
        "max_ticket_24h": float(max_ticket_24h),
        "total_amount_24h": float(total_amount_24h),
        "unique_customers_24h": float(unique_customers_24h),
        "customer_txn_ratio": float(customer_txn_ratio),
        "pct_card_present": float(pct_card_present),
        "pct_international": float(pct_international),
        "peer_volume_zscore": float(peer_vol_z),
        "peer_amount_zscore": float(peer_amt_z),
        "hour_entropy": float(hour_entropy),
        "business_age_days": float(business_age_days),
    }
    return features


def _calculate_peer_zscores(merchant_id: str, mcc_code: str, txn_count_24h: int, avg_ticket_24h: float) -> tuple:
    peer_volumes = get_peer_daily_volumes(mcc_code, exclude_merchant=merchant_id, days=1)

    if peer_volumes.empty or len(peer_volumes) < 2:
        return 0.0, 0.0

    peer_mean_vol = peer_volumes["txn_count"].mean()
    peer_std_vol = peer_volumes["txn_count"].std()
    peer_mean_amt = peer_volumes["avg_amount"].mean()
    peer_std_amt = peer_volumes["avg_amount"].std()

    vol_z = (txn_count_24h - peer_mean_vol) / peer_std_vol if peer_std_vol > 0 else 0.0
    amt_z = (avg_ticket_24h - peer_mean_amt) / peer_std_amt if peer_std_amt > 0 else 0.0

    return vol_z, amt_z


def _calculate_hour_entropy(txns: pd.DataFrame) -> float:
    if txns.empty:
        return 0.0
    hours = txns["timestamp"].dt.hour
    counts = hours.value_counts(normalize=True)
    entropy = -np.sum(counts * np.log2(counts + 1e-10))
    return float(entropy)


def calculate_all_merchant_features(merchant_ids: list[str]) -> pd.DataFrame:
    rows = []
    for mid in merchant_ids:
        feats = calculate_features(mid)
        feats["merchant_id"] = mid
        rows.append(feats)
    return pd.DataFrame(rows)
