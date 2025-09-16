#########################
# Input데이타 
#      2.에서 생성된 "최종merge.csv"
# Output데이타
#      scikit-learn에서 제공해주는 random forest 모델 데이타
#        - flight.joblib (모델 파이프라인)
#        - flight.metrics.json (정확도/클래스별 지표/혼동행렬)
#        - flight.feature_importances.csv (피처 중요도)
#        - flight.feature_importances.png (피처 중요도 그래프)
#     
#########################


#### 사전준비
# pip install argparse
# pip install matplotlib
# pip install scikit-learn
# pip install joblib

# #### 실행방법
# python 3.training_model.py
# 

"""
Train a RandomForest model for flight departure status (출발/지연/취소)
from weather features, and save the model + evaluation artifacts.

Default usage (no args):
    python train_rf_flight_status.py

Defaults:
    --csv ./middle_data/최종merge.csv
    --out_model ./model/flight.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURE_COLS = ["풍속(m/s)", "풍향(deg)", "시정(m)", "강수량(mm)", "순간풍속(m/s)"]
TARGET_COL = "상태"
VALID_LABELS = {"출발", "지연", "취소"}

def load_data(csv_path: str) -> pd.DataFrame:
    # Try common encodings
    last_err = None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception as e:
        last_err = e
        df = None
        
    if df is None:
        raise last_err

    # Cast features to numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only valid labels
    df = df[df[TARGET_COL].isin(VALID_LABELS)].copy()
    # Drop missing target just in case
    df = df[df[TARGET_COL].notna()].copy()
    return df

def build_pipeline(n_estimators: int, min_samples_leaf: int, random_state: int):
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pre = ColumnTransformer([("num", num_tf, FEATURE_COLS)], remainder="drop")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocess", pre), ("rf", rf)])

def evaluate_and_save(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, out_prefix: Path):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    labels = list(pipe.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Save metrics JSON
    metrics_path = out_prefix.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "classification_report": report,
                "labels": labels,
                "confusion_matrix": cm.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Save feature importances
    rf = pipe.named_steps["rf"]
    importances = rf.feature_importances_
    fi = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    fi_path = out_prefix.with_suffix(".feature_importances.csv")
    fi.to_csv(fi_path, index=False, encoding="utf-8-sig")

    # Plot (single chart, no color/style customizations)
    plt.figure(figsize=(7, 4))
    plt.bar(fi["feature"], fi["importance"])
    plt.title("Feature Importances (RandomForest)")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig_path = out_prefix.with_suffix(".feature_importances.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()

    return acc, metrics_path, fi_path, fig_path, labels, cm

#### MAIN시작
p = argparse.ArgumentParser()
p.add_argument("--csv", default="./middle_data/최종merge.csv", help="Path to merged CSV")
p.add_argument("--out_model", default="./model/flight.joblib", help="Output .joblib model path")
p.add_argument("--test_size", type=float, default=0.2)
p.add_argument("--n_estimators", type=int, default=400)
p.add_argument("--min_samples_leaf", type=int, default=2)
p.add_argument("--random_seed", type=int, default=42)  # name spelled 'random_seed' for clarity
args = p.parse_args()

csv_path = args.csv
out_model = Path(args.out_model)
out_model.parent.mkdir(parents=True, exist_ok=True)

print("[1/4] Loading data:", csv_path)
df = load_data(csv_path)
if df.empty:
    print("ERROR: No valid rows after filtering labels {출발, 지연, 취소}.", file=sys.stderr)
    sys.exit(1)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print("[2/4] Train/test split ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
)

print("[3/4] Build & train pipeline ...")
pipe = build_pipeline(args.n_estimators, args.min_samples_leaf, args.random_seed)
pipe.fit(X_train, y_train)

print("[4/4] Evaluate & save ...")
acc, metrics_path, fi_path, fig_path, labels, cm = evaluate_and_save(
    pipe, X_test, y_test, out_model
)
from joblib import dump
dump(pipe, out_model)

print("\n=== Summary ===")
print("Model:", out_model)
print("Accuracy:", round(acc, 4))
print("Metrics JSON:", metrics_path)
print("Feature Importances CSV:", fi_path)
print("Feature Importances Plot:", fig_path)
print("Labels (order):", labels)
print("Confusion matrix (rows=true, cols=pred):")
print(cm)


