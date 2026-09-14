"""Train and package a complete preprocessing and regression pipeline."""

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
TARGET = "median_house_value"
NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms",
           "total_bedrooms", "population", "households", "median_income"]
FEATURES = NUMERIC + ["ocean_proximity"]


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def train(data_dir, output_dir, seed=42):
    started = time.monotonic()
    frames = {}
    hashes = {}
    for name in ("train", "test"):
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run code/datasets/prepare.py first")
        frame = pd.read_csv(path)
        if frame.empty or not set(FEATURES + [TARGET]).issubset(frame.columns):
            raise ValueError(f"{path} is empty or missing required columns")
        if frame[FEATURES + [TARGET]].isna().any().any():
            raise ValueError(f"{path} contains missing values; run data preparation first")
        if not np.isfinite(frame[NUMERIC + [TARGET]].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains non-finite numeric values")
        frames[name] = frame
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    training, testing = frames["train"], frames["test"]
    preprocessing = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ocean_proximity"]),
    ])
    parameters = {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 2,
                  "random_state": seed, "n_jobs": 1}
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", RandomForestRegressor(**parameters)),
    ])
    logging.info("Training RandomForestRegressor on %s rows", len(training))
    model.fit(training[FEATURES], training[TARGET])
    predictions = model.predict(testing[FEATURES])
    testing_metrics = metrics(testing[TARGET], predictions)
    baseline = DummyRegressor(strategy="mean").fit(training[FEATURES], training[TARGET])
    baseline_metrics = metrics(testing[TARGET], baseline.predict(testing[FEATURES]))

    output_dir.mkdir(parents=True, exist_ok=True)
    # Save preprocessing with the estimator; API can pass the original input columns.
    temporary_model = output_dir / "model.tmp.joblib"
    joblib.dump(model, temporary_model, compress=3)
    restored = joblib.load(temporary_model)
    np.testing.assert_allclose(restored.predict(testing[FEATURES].iloc[:10]), predictions[:10])
    os.replace(temporary_model, output_dir / "model.joblib")
    report = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "RandomForestRegressor", "parameters": parameters,
        "features": FEATURES, "target": TARGET,
        "train_rows": len(training), "test_rows": len(testing),
        "test_metrics": testing_metrics, "baseline_mean_metrics": baseline_metrics,
        "data_sha256": hashes,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "duration_seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Append each run for a local experiment history, without requiring a server.
    with (output_dir / "training_history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(report) + "\n")
    logging.info("Test metrics: %s", testing_metrics)
    logging.info("Mean baseline metrics: %s", baseline_metrics)
    logging.info("Model saved to %s", output_dir / "model.joblib")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.seed)
