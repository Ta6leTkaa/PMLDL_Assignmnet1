"""Run all three stages once or periodically (default: every five minutes)."""
import argparse
import fcntl
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "services/pipeline/logs"
COMPOSE = ["docker", "compose", "-f", str(ROOT / "code/deployment/docker-compose.yml")]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_once():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logging.warning("Another pipeline run is active; skipping this start")
            return None
        started = time.monotonic()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = {"run_id": run_id, "started_at_utc": utc_now(), "status": "running", "stages": []}
        log_path = STATE / f"{run_id}.log"
        steps = [
            ("data_engineering", [sys.executable, str(ROOT / "code/datasets/prepare.py")]),
            ("model_engineering", [sys.executable, str(ROOT / "code/models/train.py")]),
            ("deployment", COMPOSE + ["up", "--build", "-d", "--wait", "--wait-timeout", "180"]),
        ]
        logging.info("Run %s started; detailed log: %s", run_id, log_path)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                for name, command in steps:
                    stage_started = time.monotonic()
                    logging.info("Starting %s", name)
                    stage = {"name": name, "status": "running"}
                    record["stages"].append(stage)
                    log.write(f"\n=== {name} ===\n")
                    log.flush()
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
                    stage.update(status="success" if result.returncode == 0 else "failed",
                                 duration_seconds=round(time.monotonic() - stage_started, 3))
                    result.check_returncode()
                # Verify the deployed model identity and a real form submission over HTTP.
                expected_hash = hashlib.sha256((ROOT / "models/model.joblib").read_bytes()).hexdigest()
                smoke = '''import sys, requests
from streamlit.testing.v1 import AppTest
health = requests.get("http://api:8000/health", timeout=10)
health.raise_for_status()
assert health.json()["model_sha256"] == sys.argv[1], "Deployed model differs from trained model"
app = AppTest.from_file("/workspace/main.py").run(timeout=30)
assert not app.exception
app.button[0].click().run(timeout=30)
assert not app.exception and not app.error
assert len(app.metric) == 1
print("Prediction displayed:", app.metric[0].value)
'''
                subprocess.run(COMPOSE + ["exec", "-T", "app", "python", "-c", smoke, expected_hash],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
                record["status"] = "success"
                record["model_sha256"] = expected_hash
                record["test_metrics"] = json.loads((ROOT / "models/metrics.json").read_text())["test_metrics"]
                record["deployment_smoke_test"] = "passed"
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                record["status"] = "failed"
                record["error"] = str(error)
                for stage in record["stages"]:
                    if stage["status"] == "running":
                        stage["status"] = "failed"
                logging.exception("Run failed; see %s", log_path)
            finally:
                record["finished_at_utc"] = utc_now()
                record["duration_seconds"] = round(time.monotonic() - started, 3)
                (STATE / "latest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                with (STATE / "runs.jsonl").open("a", encoding="utf-8") as history:
                    history.write(json.dumps(record) + "\n")
        logging.info("Run %s: %s in %.1fs", run_id, record["status"], record["duration_seconds"])
        return record["status"] == "success"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", action="store_true", help="Stay running and repeat the full pipeline")
    parser.add_argument("--interval", type=float, default=300, help="Seconds between starts (default 300)")
    parser.add_argument("--runs", type=int, help="Stop after this many scheduled starts (for verification)")
    args = parser.parse_args()
    if args.interval <= 0 or (args.runs is not None and args.runs <= 0):
        parser.error("interval and runs must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not args.schedule:
        return 1 if run_once() is False else 0
    logging.info("Scheduler active: interval=%ss; stop with Ctrl+C", args.interval)
    starts = 0
    failed = False
    while True:
        started = time.monotonic()
        failed = run_once() is False or failed
        starts += 1
        if args.runs is not None and starts >= args.runs:
            return int(failed)
        elapsed = time.monotonic() - started
        if elapsed >= args.interval:
            logging.warning("Run exceeded interval; waiting one full interval before next start")
            time.sleep(args.interval)
        else:
            logging.info("Next start in %.1fs", args.interval - elapsed)
            time.sleep(args.interval - elapsed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nScheduler stopped; API and app containers remain running.")
        sys.exit(0)
