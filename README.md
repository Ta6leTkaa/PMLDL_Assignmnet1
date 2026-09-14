# California Housing Pipeline

Predicts median housing values for California districts using the
[California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices).

The pipeline cleans and splits the data, trains a Random Forest with one-hot
encoding for `ocean_proximity`, and deploys a FastAPI API and Streamlit app in
separate Docker containers. Test metrics include MAE, RMSE, and R².

## Run

Requires Python 3.14 and Docker with Compose. Start Docker before running these
commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python services/pipeline/run.py --schedule
```

The first run starts immediately. The full pipeline repeats every 5 minutes
while the process is running. Use Ctrl+C to stop the scheduler.

- App: http://localhost:8501
- API docs: http://localhost:8001/docs
- Health check: http://localhost:8001/health

Enter district characteristics in the app and click **Get prediction**.
Room counts and population describe the whole district; predictions use historical dataset values.

For a single run:

```bash
.venv/bin/python services/pipeline/run.py
```

To stop the containers:

```bash
docker compose -f code/deployment/docker-compose.yml down
```

## Files

- `data/raw/housing.csv` — source dataset.
- `data/processed/` — train/test splits and preparation report.
- `models/model.joblib` — trained model and feature transformer.
- `models/metrics.json` — test metrics and training parameters.
- `models/training_history.jsonl` — training history.
- `services/pipeline/logs/` — stage logs, `latest.json`, and `runs.jsonl`.

Data is split 80/20 with seed 42. Missing values are filled using training
statistics, and training outliers are removed using 3 IQR bounds.
Each run checks the deployed model hash and a prediction through the web form.

## Checks

After running the pipeline:

```bash
.venv/bin/python -m unittest discover -s tests
```
