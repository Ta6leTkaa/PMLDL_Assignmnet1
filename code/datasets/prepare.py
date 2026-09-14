"""Prepare California Housing data using only the Python standard library."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import median, quantiles


ROOT = Path(__file__).resolve().parents[2]
TARGET = "median_house_value"
NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income", TARGET,
]
# Coordinates and capped age/target values are not treated as statistical outliers.
OUTLIER_COLUMNS = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]


def prepare(raw_path, output_dir, seed=42, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test-size must be between 0 and 1")
    with raw_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        required = set(NUMERIC + ["ocean_proximity"])
        if columns is None or not required.issubset(columns):
            raise ValueError(f"Missing required columns: {required - set(columns or [])}")
        raw_rows = list(reader)

    rows, seen = [], set()
    duplicates = missing_targets = 0
    for raw in raw_rows:
        key = tuple(raw[column] for column in columns)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        row = dict(raw)
        for column in NUMERIC:
            row[column] = float(raw[column]) if raw[column].strip() else None
        if row[TARGET] is None:
            missing_targets += 1
            continue
        rows.append(row)

    # Split first: all cleaning parameters must be learned from training data.
    random.Random(seed).shuffle(rows)
    test_count = round(len(rows) * test_size)
    test, train = rows[:test_count], rows[test_count:]
    if len(train) < 4 or not test:
        raise ValueError("Dataset is too small for the requested split")
    medians = {}
    imputed = {"train": {}, "test": {}}
    for column in NUMERIC:
        if column == TARGET:
            continue
        observed = [row[column] for row in train if row[column] is not None]
        if not observed:
            raise ValueError(f"Training column {column} has no observed values")
        medians[column] = median(observed)
        for name, partition in (("train", train), ("test", test)):
            imputed[name][column] = sum(row[column] is None for row in partition)
            for row in partition:
                if row[column] is None:
                    row[column] = medians[column]

    categories = [row["ocean_proximity"] for row in train if row["ocean_proximity"].strip()]
    if not categories:
        raise ValueError("Training data has no ocean_proximity categories")
    category_mode = max(sorted(set(categories)), key=categories.count)
    for name, partition in (("train", train), ("test", test)):
        imputed[name]["ocean_proximity"] = sum(not row["ocean_proximity"].strip() for row in partition)
        for row in partition:
            if not row["ocean_proximity"].strip():
                row["ocean_proximity"] = category_mode

    bounds = {}
    for column in OUTLIER_COLUMNS:
        q1, _, q3 = quantiles([row[column] for row in train], n=4, method="inclusive")
        spread = q3 - q1
        bounds[column] = [q1 - 3 * spread, q3 + 3 * spread]
    train_before = len(train)
    train = [row for row in train if all(low <= row[column] <= high for column, (low, high) in bounds.items())]
    if not train:
        raise ValueError("Outlier filtering removed every training row")
    # Keep test outliers to measure quality on the full held-out distribution.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in (("train", train), ("test", test)):
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=columns)
            writer.writeheader()
            writer.writerows(partition)
    report = {
        "seed": seed, "test_size": test_size, "raw_rows": len(raw_rows),
        "duplicates_removed": duplicates, "missing_targets_removed": missing_targets,
        "train_rows_before_outlier_removal": train_before,
        "train_outliers_removed": train_before - len(train),
        "train_rows": len(train), "test_rows": len(test),
        "imputed_values": imputed, "training_medians": medians,
        "ocean_proximity_mode": category_mode,
        "outlier_rule": "3 IQR, fitted on train; remove from train only",
        "outlier_bounds": bounds,
    }
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/housing.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.seed, args.test_size)
