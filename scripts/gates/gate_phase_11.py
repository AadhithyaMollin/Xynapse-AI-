#!/usr/bin/env python3
"""Phase 11 dashboard gate: AppTest, full-row CSV, and short headless-server smoke test."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
DISCLAIMER = "Predictions are computational hypotheses, not experimental findings."


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run_gate() -> int:
    from app.app import csv_bytes
    from streamlit.testing.v1 import AppTest

    checks: list[tuple[str, bool, str]] = []
    data_dir = ROOT / "data/processed"
    predictions_path = data_dir / "predictions.parquet"
    features_path = data_dir / "features_tier1.parquet"
    if not predictions_path.is_file() or not features_path.is_file():
        print("FAIL: Phase 11 data gate requires local predictions and Tier-1 features")
        return 1
    os.environ["XYNAPSE_DATA_DIR"] = str(data_dir)

    app_test = AppTest.from_file(str(ROOT / "app/app.py"), default_timeout=30).run()
    no_exception = len(app_test.exception) == 0
    checks.append(("Streamlit AppTest runs without exception and shows the disclaimer",
                   no_exception and any(DISCLAIMER in item.value for item in app_test.warning),
                   f"exceptions={len(app_test.exception)}"))
    checks.append(("heatmap, mutation table, and CSV download controls render",
                   len(app_test.get("plotly_chart")) >= 1 and len(app_test.get("download_button")) == 1
                   and len(app_test.get("dataframe")) >= 2,
                   f"charts={len(app_test.get('plotly_chart'))}; tables={len(app_test.get('dataframe'))}"))
    protein_selector = next((item for item in app_test.selectbox if item.label == "Protein"), None)
    if protein_selector is not None:
        app_test = protein_selector.set_value("P60484").run()
    pten_visible = (len(app_test.exception) == 0
                    and any("Protein summary: P60484" in item.value for item in app_test.subheader)
                    and len(app_test.get("dataframe")) >= 2
                    and app_test.get("dataframe")[1].value.shape[0] == 19 * 403)
    checks.append(("P60484 heatmap and complete 19 x L mutation table render",
                   pten_visible, f"table_rows={app_test.get('dataframe')[1].value.shape[0] if len(app_test.get('dataframe')) > 1 else 0}"))

    predictions = pd.read_parquet(predictions_path)
    pten = predictions.loc[predictions.protein_id.astype(str).eq("P60484")]
    csv_frame = pd.read_csv(BytesIO(csv_bytes(pten)))
    checks.append(("CSV download payload contains every P60484 mutation row",
                   len(csv_frame) == 19 * 403 and len(csv_frame) == len(pten),
                   f"csv_rows={len(csv_frame)}; expected={19 * 403}"))

    port = _available_port()
    env = os.environ.copy()
    env["XYNAPSE_DATA_DIR"] = str(data_dir)
    log_path = ROOT / "reports/gates/phase_11_server.txt"
    started = time.monotonic()
    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as server_log:
        command = [sys.executable, "-m", "streamlit", "run", "app/app.py",
                   "--server.headless", "true", "--server.port", str(port)]
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=server_log,
                                   stderr=subprocess.STDOUT, text=True)
        time.sleep(4)
        server_started = process.poll() is None
        if server_started:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        server_runtime = time.monotonic() - started
        server_log.seek(0)
        server_text = server_log.read()
    server_text = "\n".join(
        line for line in server_text.splitlines()
        if not line.strip().startswith(("Network URL:", "External URL:"))
    ) + "\n"
    log_path.write_text(server_text, encoding="utf-8")
    checks.append(("headless Streamlit server starts and stops cleanly", server_started,
                   f"port={port}; smoke_seconds={server_runtime:.3f}; exit={process.returncode}"))

    summary = {
        "app_test_exceptions": int(len(app_test.exception)),
        "p60484_rows": int(len(pten)),
        "csv_rows": int(len(csv_frame)),
        "headless_server_started": bool(server_started),
        "headless_server_smoke_seconds": server_runtime,
        "data_dir": "data/processed",
        "disclaimer": DISCLAIMER,
    }
    summary_path = ROOT / "reports/tables/phase_11_dashboard_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    test_run = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True)
    (ROOT / "reports/gates/phase_11_pytest.txt").write_text(test_run.stdout + test_run.stderr)
    last = test_run.stdout.strip().splitlines()[-1] if test_run.stdout.strip() else test_run.stderr.strip()
    checks.append(("pytest -q passes", test_run.returncode == 0, last))
    lines = [DISCLAIMER] + [f"{'PASS' if passed else 'FAIL'}: {name} | {detail}" for name, passed, detail in checks]
    (ROOT / "reports/gates/phase_11_gate.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(run_gate())
