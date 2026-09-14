"""Integration checks for model loading and input validation."""
import importlib.util
import unittest
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("housing_api", ROOT / "code/deployment/api/main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ApiTests(unittest.TestCase):
    def test_prediction_and_validation(self):
        row = pd.read_csv(ROOT / "data/processed/test.csv").iloc[0].drop("median_house_value").to_dict()
        model = joblib.load(ROOT / "models/model.joblib")
        expected = round(float(model.predict(pd.DataFrame([row]))[0]), 2)
        with TestClient(module.app) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            response = client.post("/predict", json=row)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"median_house_value": expected, "currency": "USD"})
            for invalid in (dict(row, households=0), dict(row, median_income=-1),
                            dict(row, ocean_proximity="UNKNOWN"),
                            dict(row, median_house_value=123),
                            dict(row, total_bedrooms=row["total_rooms"] + 1)):
                self.assertEqual(client.post("/predict", json=invalid).status_code, 422)
            missing = dict(row)
            del missing["latitude"]
            self.assertEqual(client.post("/predict", json=missing).status_code, 422)


if __name__ == "__main__":
    unittest.main()
