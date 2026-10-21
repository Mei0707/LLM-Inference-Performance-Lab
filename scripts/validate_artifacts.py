"""Exercise report regeneration and the regression CLI against measured mock artifacts."""

import json
import subprocess
import sys
from pathlib import Path

from inference_lab.analysis import load_policy, regression
from inference_lab.reporting import regenerate


def main(root):
    experiments = sorted(Path(root).glob("mock-concurrency-*/config.json"))
    if not experiments:
        raise RuntimeError("No mock benchmark found")
    directory = max(experiments, key=lambda path: path.stat().st_mtime).parent
    regenerate(directory)
    summaries = sorted(directory.glob("*/summary.json"))
    assert len(summaries) == 9
    for path in summaries:
        summary = json.loads(path.read_text())
        assert summary["complete"] and summary["successful_requests"] == 24
        assert summary["telemetry"]["server_samples"] > 0
        policy = load_policy("configs/regression.yaml")
        assert regression(summary, summary, policy)["status"] == "PASS"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "inference_lab.cli",
            "check",
            str(summaries[0]),
            str(summaries[0]),
            "--policy",
            "configs/regression.yaml",
        ],
        check=True,
        capture_output=True,
    )
    print(
        f"Validated {len(summaries)} runs, report regeneration and regression checks: {directory}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
