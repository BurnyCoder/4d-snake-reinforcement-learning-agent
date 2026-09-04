"""Tests for snake4d.main: phase resolution, CLI dispatch and the pipeline hand-off."""

from pathlib import Path

from snake4d import main as cli
from snake4d.config import Config


def test_every_phase_spec_resolves_to_a_callable():
    assert all(callable(cli.resolve(spec)) for spec in cli.PHASES.values())


def test_dispatch_passes_config_overrides_and_the_pdf_flag(monkeypatch):
    calls = []
    monkeypatch.setattr("snake4d.report.run",
                        lambda cfg, pdf=False: calls.append(("report", cfg, pdf)))
    monkeypatch.setattr("snake4d.evaluation.run", lambda cfg: calls.append(("evaluate", cfg, None)))
    cli.main(["report", "--pdf", "--set", "size=3"])
    cli.main(["evaluate", "--set", "policy=random"])
    assert calls[0][0] == "report" and calls[0][1].size == 3 and calls[0][2] is True
    assert calls[1][0] == "evaluate" and calls[1][1].policy == "random"


def test_pipeline_evaluates_the_model_it_trained(monkeypatch, tmp_path):
    (tmp_path / "best_model.zip").write_bytes(b"")
    seen = {}
    monkeypatch.setattr("snake4d.train.run", lambda cfg: Path(tmp_path))
    monkeypatch.setattr("snake4d.evaluation.run", lambda cfg: seen.setdefault("evaluate", cfg))
    monkeypatch.setattr("snake4d.report.run", lambda cfg: seen.setdefault("report", cfg))
    cli.pipeline(Config(size=2, ndim=2))
    assert seen["evaluate"].model_path == str(tmp_path / "best_model.zip")
    assert seen["report"].model_path == seen["evaluate"].model_path
