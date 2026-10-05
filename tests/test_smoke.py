import json
import os
import subprocess
import sys

import pytest

from tests.helpers import ROOT


@pytest.mark.smoke
def test_quick_entrypoint(tmp_path):
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    model_dir = ROOT / "data" / "models" / "SmolLM2-360M-Instruct"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "--quick",
            "--device",
            "cpu",
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(results_dir),
            "--model-dir",
            str(model_dir),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:] + completed.stdout[-2000:]
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in metrics["success_criteria"]}
    for name, item in by_name.items():
        if name == "Test coverage of src/":
            continue
        assert item["passed"], name
    assert metrics["metrics"]["summary"]["model_loaded"] is True
    assert metrics["hardware"]["peak_rss_gb"] < 3.0
    assert (results_dir / "exec_summary.md").exists()
    assert (model_dir / "model.safetensors").exists()
