"""Tests for snake4d.report on synthetic run artifacts (progress.csv, evaluations.npz, monitor)."""

import json

import numpy as np
import pandas as pd
import pytest

from snake4d.config import Config
from snake4d.report import plot_run, run, summarize_run, to_pdf


def _train_run(root, name="exp02_ppo_2x4"):
    run_dir = root / f"20260904-120000_train_{name}"
    (run_dir / "eval").mkdir(parents=True)
    Config(size=2, ndim=4).to_json(run_dir / "config.json")
    (run_dir / "versions.json").write_text("{}", encoding="utf-8")
    steps = np.arange(1, 6) * 1000
    pd.DataFrame({
        "time/total_timesteps": steps, "time/time_elapsed": steps / 100, "time/fps": 100,
        "rollout/fill_mean": [0.2, 0.4, 0.6, 0.8, 0.9],
        "rollout/fill_mean_true_start": [0.1, 0.3, 0.5, 0.7, 0.85],
        "eval/success_rate": [np.nan, 0.0, np.nan, 0.5, np.nan],
        "curriculum/frontier": [15, 12, 9, 5, 1], "rollout/ep_len_mean": [5, 9, 14, 20, 30],
    }).to_csv(run_dir / "progress.csv", index=False)
    np.savez(run_dir / "eval" / "evaluations.npz", timesteps=np.array([2000, 4000]),
             results=np.ones((2, 4)), ep_lengths=np.ones((2, 4)),
             successes=np.array([[0, 0, 0, 0], [1, 1, 0, 1]]))
    monitor = "#{\"t_start\": 0.0}\nr,l,t,is_success,fill,start_len\n"
    monitor += "".join(f"{r},{k + 3},{k},{k % 2 == 0},{0.25 * (k % 4 + 1)},1\n"
                       for k, r in enumerate([1.0, 2.0, 3.0, 4.0]))
    (run_dir / "monitor.monitor.csv").write_text(monitor, encoding="utf-8")
    return run_dir


def _evaluate_run(root, name="exp01_route_2x4"):
    run_dir = root / f"20260904-110000_evaluate_{name}"
    run_dir.mkdir(parents=True)
    Config(size=2, ndim=4).to_json(run_dir / "config.json")
    summary = {"board": "2^4", "modes": {"deterministic=True": {
        "success_rate_mean": 1.0, "success_rate_std": 0.0, "fill_mean": 1.0,
        "steps_to_complete_mean": 68.0}}}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_plot_run_writes_curves_and_histogram(tmp_path):
    figures = plot_run(_train_run(tmp_path / "runs"), tmp_path / "figs")
    assert [f.name for f in figures] == ["exp02_ppo_2x4_curves.png", "exp02_ppo_2x4_fill_hist.png"]
    assert all(f.stat().st_size > 1000 for f in figures)


def test_summarize_train_and_evaluate_runs(tmp_path):
    train_row = summarize_run(_train_run(tmp_path / "runs"))
    assert train_row["timesteps"] == 5000 and train_row["fill_final"] == 0.9
    assert train_row["eval_success_best"] == 0.75 and train_row["eval_success_best_at"] == 4000
    eval_row = summarize_run(_evaluate_run(tmp_path / "runs"))
    assert eval_row["success_det"] == "1.000 +- 0.000" and eval_row["steps_to_complete_det"] == 68.0


def test_report_phase_writes_table_figures_and_data(tmp_path):
    runs = tmp_path / "runs"
    _train_run(runs)
    _evaluate_run(runs)
    for phase in ("publish", "watch"):  # not experiments: they stay out of the table and data
        other = runs / f"20260904-140000_{phase}_run"
        other.mkdir()
        Config(size=2, ndim=4).to_json(other / "config.json")
    reports = tmp_path / "reports"
    (reports / "experiments").mkdir(parents=True)
    (reports / "experiments" / "exp02_ppo_2x4.md").write_text("# exp02\n", encoding="utf-8")
    cfg = Config(runs_dir=str(runs))
    table = run(cfg, reports_dir=reports)
    text = table.read_text(encoding="utf-8")
    assert "exp02_ppo_2x4" in text and "exp01_route_2x4" in text and "| run " in text
    assert "publish" not in text and "watch" not in text
    assert not (reports / "data" / "run").exists()
    assert "[exp02_ppo_2x4](experiments/exp02_ppo_2x4.md)" in text
    assert (reports / "figures" / "exp02_ppo_2x4_curves.png").exists()
    data = reports / "data" / "exp02_ppo_2x4"
    assert (data / "evaluations.npz").exists() and (data / "progress.csv").exists()
    stats = json.loads((data / "episodes.json").read_text(encoding="utf-8"))
    assert stats["true_start_episodes"] == 4 and stats["curriculum_episodes"] == 0
    assert stats["true_start_fill_max"] == 1.0 and stats["true_start_success_rate"] == 0.5
    assert (reports / "data" / "exp01_route_2x4" / "summary.json").exists()
    assert not (reports / "data" / "exp01_route_2x4" / "episodes.json").exists()


@pytest.mark.slow
def test_to_pdf_builds_a_document_in_order(tmp_path):
    pytest.importorskip("markdown_pdf")
    pymupdf = pytest.importorskip("pymupdf")  # markdown_pdf's renderer, used here to read back
    reports = tmp_path / "reports"
    (reports / "experiments").mkdir(parents=True)
    (reports / "paper.md").write_text("# Paper\n\nPAPERMARK\n", encoding="utf-8")
    (reports / "all_experiments.md").write_text("# All\n\n| a |\n|---|\n| TABLEMARK |\n",
                                                encoding="utf-8")
    (reports / "networks.md").write_text("# Networks\n\nNETWORKSMARK\n", encoding="utf-8")
    (reports / "experiments" / "exp02_ppo_2x4.md").write_text("# Exp02\n\nEXPMARK\n",
                                                               encoding="utf-8")
    pdf = to_pdf(reports, tmp_path / "paper.pdf")
    assert pdf.stat().st_size > 1000
    text = "".join(page.get_text() for page in pymupdf.open(str(pdf)))
    marks = [text.index(m) for m in ("PAPERMARK", "TABLEMARK", "NETWORKSMARK", "EXPMARK")]
    assert marks == sorted(marks)  # paper, cross-run table, networks comparison, then write-ups
