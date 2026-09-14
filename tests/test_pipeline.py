"""Verify failed stages stop the pipeline, locking and scheduler cadence."""
import fcntl
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('pipeline_runner', ROOT / 'services/pipeline/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class PipelineTests(unittest.TestCase):
    def test_failure_stops_later_stages_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with patch.object(runner, 'STATE', state), patch.object(runner.subprocess, 'run') as run:
                run.return_value = subprocess.CompletedProcess(['prepare'], returncode=1)
                with self.assertLogs(level='ERROR'):
                    self.assertFalse(runner.run_once())
                self.assertEqual(run.call_count, 1)
                report = json.loads((state / 'latest.json').read_text())
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['stages'][0]['status'], 'failed')
                self.assertEqual(len((state / 'runs.jsonl').read_text().splitlines()), 1)

    def test_active_run_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with (state / 'pipeline.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(runner, 'STATE', state), patch.object(runner.subprocess, 'run') as run:
                    with self.assertLogs(level='WARNING'):
                        self.assertIsNone(runner.run_once())
                    run.assert_not_called()

    def test_schedule_waits_between_start_times(self):
        with patch.object(runner.sys, 'argv', ['run.py', '--schedule', '--runs', '2']), \
             patch.object(runner, 'run_once', return_value=True) as run, \
             patch.object(runner.time, 'monotonic', side_effect=[0, 5, 300]), \
             patch.object(runner.time, 'sleep') as sleep:
            self.assertEqual(runner.main(), 0)
            self.assertEqual(run.call_count, 2)
            sleep.assert_called_once_with(295)


if __name__ == '__main__':
    unittest.main()
