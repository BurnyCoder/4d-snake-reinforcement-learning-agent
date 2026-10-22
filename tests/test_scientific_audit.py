"""Recompute the scientific audit from saved records without importing any model code.

Global context: these checks protect the corrected claims, not training performance. Standard
library statistics independently check the evaluation summaries; archived monitor excerpts expose
the stale-frontier problem. Complete local monitors are optional and never regenerated here.
"""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "data"
AUDIT = DATA / "scientific_audit"
EVALUATIONS = sorted(DATA.glob("*/eval_episodes.csv"))


def read_json(path):
    """Read an existing artifact; no project phase or model is invoked."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    """Read complete episode rows, preserving their recorded decimal strings."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recompute(rows, cells):
    """Independent arithmetic for evaluation.py's metrics, including failed episodes."""
    wins = [int(row["length"]) for row in rows if row["success"] == "1"]
    budgets = {"C": cells, "2C": 2 * cells, "4C": 4 * cells, "C2/2": cells**2 // 2}
    return {
        "episodes": len(rows),
        "success_rate": len(wins) / len(rows),
        "fill_mean": statistics.fmean(float(row["fill"]) for row in rows),
        "return_mean": statistics.fmean(float(row["return"]) for row in rows),
        "length_mean": statistics.fmean(int(row["length"]) for row in rows),
        "steps_to_complete_mean": statistics.fmean(wins) if wins else None,
        "win_within": {key: sum(t <= cap for t in wins) / len(rows)
                       for key, cap in budgets.items()},
    }


@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda path: path.parent.name)
def test_all_archived_evaluation_fields_match_episode_records(path):
    """Check each seed/mode and pooled mode, rather than only hand-selected headline scores."""
    rows = read_csv(path)
    summary = read_json(path.with_name("summary.json"))
    cells = summary["n_cells"]
    for row in rows:
        fill = float(row["fill"])
        assert (row["success"] == "1") == (fill == 1)
        assert 0 < fill <= 1 and math.isclose(fill * cells, round(fill * cells))
        assert 0 < int(row["length"]) <= cells**2
    for section in ("per_run", "modes"):
        for key, expected in summary[section].items():
            mode = str(int(key.startswith("deterministic=True")))
            subset = [row for row in rows if row["deterministic"] == mode
                      and ("/seed=" not in key or row["seed"] == key.split("/seed=")[1])]
            actual = recompute(subset, cells)
            if section == "per_run":
                assert len(subset) == 100
            else:
                rates = [statistics.fmean(int(row["success"]) for row in subset
                                          if row["seed"] == str(seed))
                         for seed in summary["seeds"]]
                actual.update(success_rate_mean=statistics.fmean(rates),
                              success_rate_std=statistics.pstdev(rates))
            for metric, value in actual.items():
                assert expected[metric] == pytest.approx(value) if value is not None else (
                    expected[metric] is None)


def test_audit_scope_and_training_replication():
    """The published counts and training-seed limitation follow from the complete archive."""
    assert len(EVALUATIONS) == 16
    assert sum(len(read_csv(path)) for path in EVALUATIONS) == 7800
    summaries = [read_json(path.with_name("summary.json")) for path in EVALUATIONS]
    assert sum(len(item["per_run"]) for item in summaries) == 78
    assert sum(len(item["modes"]) for item in summaries) == 26
    assert sum(7 * len(item["per_run"]) + 9 * len(item["modes"]) for item in summaries) == 780
    training = [path.parent for path in DATA.glob("*/config.json")
                if (path.parent / "progress.csv").exists() or path.parent.name == "exp05_bc_4x4"]
    assert len(training) == 10
    assert {read_json(path / "config.json")["seed"] for path in training} == {0}
    report = (ROOT / "reports/scientific_audit.md").read_text(encoding="utf-8")
    assert "7,800" in report and "780" in report and "one training seed" in report


def test_gate_excerpts_expose_stale_episode_contamination():
    """Recompute the reported gate and the conservative exclusion of definitely stale starts."""
    evidence = read_json(AUDIT / "evidence.json")
    assert hashlib.sha256((AUDIT / "gate_episodes.csv").read_bytes()).hexdigest() == (
        evidence["gate_excerpt_sha256"])
    rows = read_csv(AUDIT / "gate_episodes.csv")
    progress = read_csv(DATA / "exp04_ppo_4x4/progress.csv")
    report = (ROOT / "reports/scientific_audit.md").read_text(encoding="utf-8")
    for gate in evidence["gates"]:
        selected = [row for row in rows if int(row["frontier"]) == gate["frontier"]]
        assert len(selected) == gate["episodes"]
        assert all(gate["time_lower_exclusive"] < float(row["t"])
                   <= gate["time_upper_inclusive"] for row in selected)
        wins = sum(row["is_success"] == "True" for row in selected)
        assert wins == gate["wins"]
        logged = next(row for row in progress
                      if row["time/total_timesteps"] == str(gate["timesteps"]))
        assert float(logged["curriculum/success_rate"]) == pytest.approx(wins / len(selected))
        remaining = [row for row in selected if int(row["start_len"]) <= gate["frontier"]]
        remaining_wins = sum(row["is_success"] == "True" for row in remaining)
        assert remaining_wins / len(remaining) < 0.8 < wins / len(selected)
        assert f"{remaining_wins}/{len(remaining)}" in report
        assert f"{100 * remaining_wins / len(remaining):.2f}%" in report


@pytest.mark.parametrize("run", ["exp03b_ppo_3x4_backplay", "exp03c_ppo_3x4_backplay_relaxed",
                                "exp04_ppo_4x4"])
def test_transition_shares_match_complete_local_monitor_when_available(run):
    """Local-only corroboration: distinguish exact raw sums from rounded archived means."""
    evidence = read_json(AUDIT / "evidence.json")["monitors"][run]
    source = ROOT / evidence["path"]
    if not source.exists():
        pytest.skip("full ignored monitor unavailable; counts and gate excerpts are archived")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == evidence["sha256"]
    with source.open(newline="", encoding="utf-8") as handle:
        next(handle)
        rows = list(csv.DictReader(handle))
    assert len(rows) == evidence["completed_episodes"]
    assert sum(int(row["l"]) for row in rows) == evidence["completed_transitions"]
    assert sum(int(row["l"]) for row in rows if row["start_len"] == "1") == (
        evidence["true_start_transitions"])
    if run == "exp04_ppo_4x4":
        for row in read_csv(AUDIT / "gate_episodes.csv"):
            original = rows[int(row["source_line"]) - 3]
            assert all(row[key] == value for key, value in original.items())


def test_fine_tune_coverage_and_nonzero_exploration():
    """The saved record contradicts no-change/no-exploration claims without new inference."""
    run = DATA / "exp05b_ppo_4x4_from_bc"
    episodes = read_json(run / "episodes.json")
    assert episodes["episodes"] == 1 and episodes["true_start_success_rate"] == 0
    progress = read_csv(run / "progress.csv")
    steps = max(int(row["time/total_timesteps"]) for row in progress)
    assert steps // read_json(run / "config.json")["n_envs"] == 4928
    assert all(float(row["train/approx_kl"]) > 0 for row in progress if row["train/approx_kl"])
    for name, failures in [("exp05_bc_4x4_eval", 64), ("exp05b_ppo_4x4_from_bc_best", 71)]:
        rows = read_csv(DATA / name / "eval_episodes.csv")
        assert sum(row["deterministic"] == "0" and row["success"] == "0"
                   for row in rows) == failures


@pytest.mark.parametrize("wins", [0, 289, 300])
def test_quoted_wilson_intervals_are_fixed_checkpoint_intervals(wins):
    """Use NIST's Wilson formula, not a zero-width interval at observed zero/all successes.

    https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm
    """
    n = 300
    z = statistics.NormalDist().inv_cdf(0.975)
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    interval = f"{100 * max(0, center - half):.2f}-{100 * min(1, center + half):.2f}%"
    report = (ROOT / "reports/scientific_audit.md").read_text(encoding="utf-8")
    assert interval in report and "fixed checkpoint" in report
