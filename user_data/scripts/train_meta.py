"""Train calibrated meta-label model for EnsembleMetaStrategy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

FEATURE_COLUMNS = [
    "rsi",
    "zret",
    "atr_z",
    "adx_15m",
    "slope_15m",
    "slope_1h",
    "hour",
    "weekday",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train meta-label model.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("user_data/features/trade_features.parquet"),
        help="Input feature parquet file",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("user_data/models/meta_label_xgb.pkl"),
        help="Output path for calibrated model",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=Path("user_data/models/scaler.pkl"),
        help="Output path for feature scaler",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of time-series splits for calibration",
    )
    parser.add_argument(
        "--min-profit",
        type=float,
        default=0.0,
        help="Minimum profit ratio considered a positive label",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Feature file not found: {path}")
    df = pd.read_parquet(path)
    if "label" not in df.columns:
        raise SystemExit("Feature dataset missing 'label' column")
    return df


def prepare_features(
    df: pd.DataFrame, min_profit: float
) -> tuple[np.ndarray, np.ndarray, RobustScaler]:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required features: {', '.join(missing)}")
    X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    profit = df["profit_ratio"].astype(float)
    y = (profit >= min_profit).astype(np.int8)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_model(X: np.ndarray, y: np.ndarray, splits: int) -> CalibratedClassifierCV:
    base_model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        base_score=0.5,
    )

    splitter = TimeSeriesSplit(n_splits=splits)
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=splitter,
    )
    calibrated.fit(X, y)
    return calibrated


def main() -> None:
    args = parse_arguments()
    dataset = load_dataset(args.features)
    X, y, scaler = prepare_features(dataset, args.min_profit)

    model = train_model(X, y, args.cv_splits)
    preds = model.predict(X)
    print(classification_report(y, preds, digits=3))

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.scaler_output.parent.mkdir(parents=True, exist_ok=True)

    import pickle

    with open(args.model_output, "wb") as fh:
        pickle.dump(model, fh)
    with open(args.scaler_output, "wb") as fh:
        pickle.dump(scaler, fh)

    print(f"Saved model to {args.model_output}")
    print(f"Saved scaler to {args.scaler_output}")


if __name__ == "__main__":
    main()
