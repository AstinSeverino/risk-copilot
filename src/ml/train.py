"""Train IsolationForest + XGBoost risk models."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.database import get_all_merchants
from src.ml.features import calculate_all_merchant_features, FEATURE_NAMES

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    merchants = get_all_merchants()
    merchant_ids = merchants["merchant_id"].tolist()
    print(f"Calculating features for {len(merchant_ids)} merchants...")

    features_df = calculate_all_merchant_features(merchant_ids)
    X = features_df[FEATURE_NAMES].values

    print("Training IsolationForest...")
    iso_forest = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(X)

    if_scores = iso_forest.decision_function(X)
    if_predictions = iso_forest.predict(X)
    anomalies = np.sum(if_predictions == -1)
    print(f"  IsolationForest detected {anomalies} anomalous merchants")

    print("Creating weak labels for XGBoost...")
    is_anomaly_if = (if_predictions == -1).astype(int)
    high_volume = (features_df["volume_growth_ratio"] > 3.0).astype(int)
    low_customer_ratio = (features_df["customer_txn_ratio"] < 0.3).astype(int)
    high_peer_z = (features_df["peer_volume_zscore"] > 3.0).astype(int)

    weak_labels = np.clip(is_anomaly_if + high_volume + high_peer_z + low_customer_ratio, 0, 1)
    weak_labels = (weak_labels >= 1).astype(int)
    print(f"  Weak labels: {np.sum(weak_labels)} positive / {len(weak_labels) - np.sum(weak_labels)} negative")

    print("Training XGBoost...")
    pos_count = max(np.sum(weak_labels), 1)
    neg_count = max(len(weak_labels) - pos_count, 1)
    scale_pos = neg_count / pos_count

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False,
    )

    if len(np.unique(weak_labels)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, weak_labels, test_size=0.2, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        print(f"  XGBoost PR-AUC: {pr_auc:.3f}")
        y_pred = (y_prob >= 0.5).astype(int)
        print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"], zero_division=0))
    else:
        xgb_model.fit(X, weak_labels)
        print("  Warning: only one class in labels, skipping eval")

    if_path = MODELS_DIR / "isolation_forest.joblib"
    xgb_path = MODELS_DIR / "xgboost_risk.joblib"
    joblib.dump(iso_forest, if_path)
    joblib.dump(xgb_model, xgb_path)
    print(f"\nModels saved to {MODELS_DIR}")

    print("\nFeature importances (XGBoost):")
    importances = dict(zip(FEATURE_NAMES, xgb_model.feature_importances_))
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat}: {imp:.4f}")

    print("\nFlagged merchants:")
    all_probs = xgb_model.predict_proba(X)[:, 1]
    for i, mid in enumerate(merchant_ids):
        if all_probs[i] >= 0.5:
            name = merchants[merchants["merchant_id"] == mid]["name"].values[0]
            mcc = merchants[merchants["merchant_id"] == mid]["mcc_code"].values[0]
            print(f"  {mid} ({name}, MCC {mcc}): risk={all_probs[i]:.3f}")


if __name__ == "__main__":
    train()
