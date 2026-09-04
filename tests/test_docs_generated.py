"""docs/benchmark.md must be exactly what the generator produces from the archived benchmark.json.

This guards against hand-edited or stale numbers in the throughput document: every figure quoted
from it can be traced to reports/data/exp00_benchmark/benchmark.json.
"""

import json
from pathlib import Path

from snake4d.benchmark import write_markdown
from snake4d.config import Config

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "data" / "exp00_benchmark"


def test_docs_benchmark_md_equals_generator_output_of_archived_json(tmp_path):
    data = json.loads((DATA / "benchmark.json").read_text(encoding="utf-8"))
    cfg = Config(**json.loads((DATA / "config.json").read_text(encoding="utf-8")))
    generated = tmp_path / "benchmark.md"
    write_markdown(cfg, data["results"], data["env_only"], data["recommended"], generated,
                   data["machine"], data.get("minibatch"))
    assert generated.read_text(encoding="utf-8") == (ROOT / "docs" / "benchmark.md").read_text(
        encoding="utf-8")
