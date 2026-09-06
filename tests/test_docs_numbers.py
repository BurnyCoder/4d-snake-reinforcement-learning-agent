"""Headline numbers in README/paper/reports must be recomputable from reports/data artifacts.

Each case names a document and a formatter that derives the quoted string from an archived
summary.json / benchmark.json; the string must appear verbatim in the document.  Rounding follows
AGENTS.md: tables at artifact precision, prose as half-up integers or the same decimals.
reports/networks.md quotes whole table cells (network score beside both scripted baselines with a
better/worse/tie verdict); ``cell`` below generates them, so the document and the check cannot drift.
"""

import csv
import itertools
import json
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

from snake4d.config import Config
from snake4d.train import build_model
from snake4d.vec_env import make_env

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "data"


def summary(run: str) -> dict:
    return json.loads((DATA / run / "summary.json").read_text(encoding="utf-8"))


def det(run: str, key: str = "success_rate_mean") -> float:
    return summary(run)["modes"]["deterministic=True"][key]


def stoch(run: str, key: str = "success_rate_mean") -> float:
    return summary(run)["modes"]["deterministic=False"][key]


def half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def benchmark() -> dict:
    return json.loads((DATA / "exp00_benchmark" / "benchmark.json").read_text(encoding="utf-8"))


def best_ppo_fps() -> int:
    data = benchmark()
    rows = [r for r in data["results"] + data.get("minibatch", []) if "fps" in r]
    return max(r["fps"] for r in rows)


def grid_fps(env: str, n_envs: int, device: str, threads: int) -> int:
    return next(r["fps"] for r in benchmark()["results"]
                if (r["env"], r["n_envs"], r["device"], r["threads"]) == (env, n_envs, device, threads))


def episodes(run: str) -> dict:
    return json.loads((DATA / run / "episodes.json").read_text(encoding="utf-8"))


# --- derived efficiency metrics, formulas as written in docs/evaluation.md ---------------------
def board_size(run: str) -> int:
    return int(summary(run)["board"].split("^")[0])


def foods(run: str) -> float:
    """Mean number of foods eaten: final length (fill * C) minus the starting length of 1."""
    return det(run, "fill_mean") * summary(run)["n_cells"] - 1


def steps_per_food(run: str) -> float:
    return det(run, "length_mean") / foods(run)


def geometric_floor(n: int, ndim: int = 4) -> float:
    """Mean Manhattan distance between two uniform random cells of an n^ndim board."""
    return ndim * (n * n - 1) / (3 * n)


def non_eating_share(run: str) -> float:
    return 1 - foods(run) / det(run, "length_mean")


def won_within(run: str, key: str = "4C") -> float:
    return det(run, "win_within")[key]


def mlp_params(n_cells: int, width: int = 512, actions: int = 8) -> int:
    """Weights of MlpPolicy with net_arch {pi: [w, w], vf: [w, w]} on 4C + 2 inputs (train.py)."""
    obs = 4 * n_cells + 2
    return 2 * ((obs + 1) * width + (width + 1) * width) + (width + 1) * actions + width + 1


def eval_episodes(run: str, deterministic: bool = True) -> list[dict]:
    """Rows of eval_episodes.csv for one mode (evaluation.write_results stores deterministic as 0/1)."""
    with open(DATA / run / "eval_episodes.csv", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["deterministic"] == str(int(deterministic))]


def boxed_in(run: str) -> tuple[int, int]:
    """(lost argmax episodes that ended with exactly one free cell, all lost argmax episodes)."""
    n_cells = summary(run)["n_cells"]
    failed = [row for row in eval_episodes(run) if row["success"] == "0"]
    return sum(float(row["fill"]) == (n_cells - 1) / n_cells for row in failed), len(failed)


def fill_ratio(run: str) -> float:
    """Mean fill of a network over the random-play floor on the same board."""
    return det(run, "fill_mean") / det(f"exp01_random_{board_size(run)}x4", "fill_mean")


# --- run.log and progress.csv facts quoted in reports/networks.md ----------------------------
def log_lines(run: str, marker: str = "") -> list[str]:
    lines = (DATA / run / "run.log").read_text(encoding="utf-8").splitlines()
    return [line for line in lines if marker in line]


def log_seconds(run: str, marker: str = "") -> float:
    """Seconds between the first and the last run.log line containing ``marker`` (logging_utils stamps)."""
    def stamp(line: str) -> datetime:
        return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")

    lines = log_lines(run, marker)
    return (stamp(lines[-1]) - stamp(lines[0])).total_seconds()


def frontier_path(run: str) -> list[int]:
    """Curriculum frontier values logged by callbacks.Backplay: the start, then every advance."""
    lines = log_lines(run, "curriculum:")
    start = int(re.search(r"frontier (\d+)", lines[0]).group(1))
    return [start] + [int(re.search(r"-> (\d+)", line).group(1)) for line in lines[1:]]


def progress_column(run: str, column: str) -> list[tuple[int, float]]:
    """(timesteps, value) pairs of one SB3 progress.csv column, rows without the value skipped."""
    with open(DATA / run / "progress.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [(int(float(r["time/total_timesteps"])), float(r[column])) for r in rows if r.get(column)]


def collapse_points(run: str = "exp04_ppo_4x4") -> tuple[int, int]:
    """Timesteps from which the curriculum success stays below 1e-3, and from which it stays at 0."""
    series = progress_column(run, "curriculum/success_rate")
    last_above, last_positive = (max(t for t, v in series if v >= bound) for bound in (1e-3, 1e-12))
    return tuple(min(t for t, _ in series if t > last) for last in (last_above, last_positive))


NETWORKS_DOC = "reports/networks.md"
NETWORKS = sorted(p.name for p in DATA.iterdir() if p.name.endswith(("_best", "_eval")))
NEVER_FINISHED = [r for r in NETWORKS if det(r, "steps_to_complete_mean") is None]
ON_3X4 = [r for r in NETWORKS if board_size(r) == 3]

CASES = {
    "readme 2^4 success": ("README.md", lambda: f"**{100 * det('exp02d_ppo_2x4_backplay_strict_best'):.1f} %**"),
    "readme 2^4 steps": ("README.md", lambda: f"{det('exp02d_ppo_2x4_backplay_strict_best', 'steps_to_complete_mean'):.1f} steps"),
    "paper 2^4 steps": ("reports/paper.md", lambda: f"completing in {det('exp02d_ppo_2x4_backplay_strict_best', 'steps_to_complete_mean'):.1f} steps"),
    "readme route 2^4": ("README.md", lambda: f"100 %, {half_up(det('exp01_route_2x4', 'steps_to_complete_mean'))} steps"),
    "readme route 3^4": ("README.md", lambda: f"100 %, {half_up(det('exp01_route_3x4', 'steps_to_complete_mean')):,} steps"),
    "readme route 4^4": ("README.md", lambda: f"100 %, {half_up(det('exp01_route_4x4', 'steps_to_complete_mean')):,} steps"),
    "readme random 2^4": ("README.md", lambda: f"| {half_up(100 * det('exp01_random_2x4'))} % |"),
    "readme random 3^4 fill": ("README.md", lambda: f"0 % (fill {det('exp01_random_3x4', 'fill_mean'):.2f})"),
    "readme random 4^4 fill": ("README.md", lambda: f"0 % (fill {det('exp01_random_4x4', 'fill_mean'):.2f})"),
    "readme 4^4 ppo fill": ("README.md", lambda: f"0 %, fill {det('exp04_ppo_4x4_best', 'fill_mean'):.2f}"),
    "readme fps": ("README.md", lambda: f"{best_ppo_fps() / 1000:.0f}k PPO steps/s"),
    "readme watch exp02d": ("README.md", lambda: f"completing {100 * det('exp02d_ppo_2x4_backplay_strict_best'):.1f} % of its\ngames in {det('exp02d_ppo_2x4_backplay_strict_best', 'steps_to_complete_mean'):.1f} moves against the loop follower's {det('exp01_route_2x4', 'steps_to_complete_mean'):.1f}"),
    "readme watch exp04": ("README.md", lambda: f"never finishes (fill {det('exp04_ppo_4x4_best', 'fill_mean'):.2f})"),
    "paper exp02 plain": ("reports/paper.md", lambda: f"plain PPO {det('exp02_ppo_2x4_best'):.3f} (5M) and {det('exp02c_ppo_2x4_long_best'):.3f} (20M)"),
    "paper exp02 loose": ("reports/paper.md", lambda: f"Backplay (rho 0.2) {det('exp02b_ppo_2x4_backplay_best'):.3f}"),
    "paper exp02 strict": ("reports/paper.md", lambda: f"(rho 0.9, window 4) {det('exp02d_ppo_2x4_backplay_strict_best'):.3f} at 5M"),
    "paper exp05 det": ("reports/paper.md", lambda: f"{det('exp05_bc_4x4_eval'):.3f} +- {summary('exp05_bc_4x4_eval')['modes']['deterministic=True']['success_rate_std']:.3f}"),
    "paper exp05 stoch": ("reports/paper.md", lambda: f"({stoch('exp05_bc_4x4_eval'):.3f} +- {summary('exp05_bc_4x4_eval')['modes']['deterministic=False']['success_rate_std']:.3f} when sampling)"),
    "paper route steps": ("reports/paper.md", lambda: f"({det('exp01_route_2x4', 'steps_to_complete_mean'):.1f} / {det('exp01_route_3x4', 'steps_to_complete_mean'):,.1f} /"),
    "paper fps": ("reports/paper.md", lambda: f"{best_ppo_fps() / 1000:.0f}k PPO steps per second"),
    "exp01 route 3^4": ("reports/experiments/exp01_baselines.md", lambda: f"| {det('exp01_route_3x4', 'steps_to_complete_mean'):,.1f} |"),
    "exp01 route 4^4": ("reports/experiments/exp01_baselines.md", lambda: f"| {det('exp01_route_4x4', 'steps_to_complete_mean'):,.1f} |"),
    "exp01 random 2^4": ("reports/experiments/exp01_baselines.md", lambda: f"| {det('exp01_random_2x4'):.3f} +- {summary('exp01_random_2x4')['modes']['deterministic=True']['success_rate_std']:.3f} |"),
    "exp02 table": ("reports/experiments/exp02_ppo_2x4.md", lambda: f"**{det('exp02d_ppo_2x4_backplay_strict_best'):.3f} +- {summary('exp02d_ppo_2x4_backplay_strict_best')['modes']['deterministic=True']['success_rate_std']:.3f}**"),
    "exp03 fills": ("reports/experiments/exp03_ppo_3x4.md", lambda: f"| 0.000 | {det('exp03a_ppo_3x4_nocur_best', 'fill_mean'):.3f} |"),
    "exp04 fill": ("reports/experiments/exp04_ppo_4x4.md", lambda: f"fill {det('exp04_ppo_4x4_best', 'fill_mean'):.3f} / {stoch('exp04_ppo_4x4_best', 'fill_mean'):.3f}"),
    "exp05 fine-tune": ("reports/experiments/exp05_bc_4x4.md", lambda: f"| **{det('exp05b_ppo_4x4_from_bc_best'):.3f} +- 0.000** | {stoch('exp05b_ppo_4x4_from_bc_best'):.3f}"),
    "exp00 best fps": ("reports/experiments/exp00_benchmark.md", lambda: f"{best_ppo_fps():,}"),
    "exp00 single-thread cpu": ("reports/experiments/exp00_benchmark.md", lambda: f"| 1 thread: {grid_fps('batched', 4096, 'cpu', 1):,} |"),
    "exp05 copy of exp04 fills": ("reports/experiments/exp05_bc_4x4.md", lambda: f"| {det('exp04_ppo_4x4_best', 'fill_mean'):.3f} / {stoch('exp04_ppo_4x4_best', 'fill_mean'):.3f} |"),
    "exp03a true-start fill": ("reports/experiments/exp03_ppo_3x4.md", lambda: f"| 0.000 | {det('exp03a_ppo_3x4_nocur_best', 'fill_mean'):.3f} | {episodes('exp03a_ppo_3x4_nocur')['true_start_fill_mean']:.4f} / {episodes('exp03a_ppo_3x4_nocur')['true_start_fill_max']:.3f}"),
    "exp03b true-start fill": ("reports/experiments/exp03_ppo_3x4.md", lambda: f"| 0.000 | {det('exp03b_ppo_3x4_backplay_best', 'fill_mean'):.3f} | {episodes('exp03b_ppo_3x4_backplay')['true_start_fill_mean']:.4f} / {episodes('exp03b_ppo_3x4_backplay')['true_start_fill_max']:.3f}"),
    "exp03c true-start fill": ("reports/experiments/exp03_ppo_3x4.md", lambda: f"| 0.000 | {det('exp03c_ppo_3x4_backplay_relaxed_best', 'fill_mean'):.3f} | {episodes('exp03c_ppo_3x4_backplay_relaxed')['true_start_fill_mean']:.4f} / {episodes('exp03c_ppo_3x4_backplay_relaxed')['true_start_fill_max']:.3f}"),
    "evaluation floors": ("docs/evaluation.md", lambda: " / ".join(f"{geometric_floor(n):.2f}" for n in (2, 3, 4))),
    "networks floors": ("reports/networks.md", lambda: " / ".join(f"{geometric_floor(n):.2f}" for n in (2, 3, 4))),
    "networks weights": ("reports/networks.md", lambda: " / ".join(f"{mlp_params(c) / 1e6:.2f}M" for c in (16, 81, 256))),
    "networks exp02 boxed in": ("reports/networks.md", lambda: "{} of its {}\n  lost argmax games end with one cell left".format(*boxed_in("exp02_ppo_2x4_best"))),
    "networks exp02d sampling": ("reports/networks.md", lambda: f"{100 * stoch('exp02d_ppo_2x4_backplay_strict_best'):.1f} % when sampling"),
    "networks exp02d gain": ("reports/networks.md", lambda: f"{100 * (det('exp02d_ppo_2x4_backplay_strict_best') - det('exp02_ppo_2x4_best')):.1f} points more completions"),
    "networks exp02d moves": ("reports/networks.md", lambda: f"for about {half_up(det('exp02d_ppo_2x4_backplay_strict_best', 'steps_to_complete_mean') - det('exp02_ppo_2x4_best', 'steps_to_complete_mean'))} more"),
    "networks exp02c gain": ("reports/networks.md", lambda: f"from {100 * det('exp02_ppo_2x4_best'):.1f} % to {100 * det('exp02c_ppo_2x4_long_best'):.1f} %"),
    "networks exp02b seconds": ("reports/networks.md", lambda: f"from 15 to 1 in {half_up(log_seconds('exp02b_ppo_2x4_backplay', 'curriculum:'))} seconds of"),
    "networks exp03a foods": ("reports/networks.md", lambda: f"eats about {half_up(foods('exp03a_ppo_3x4_nocur_best'))} cells"),
    "networks exp03b frontier": ("reports/networks.md", lambda: "logged frontier {} -> {}".format(frontier_path('exp03b_ppo_3x4_backplay')[0], frontier_path('exp03b_ppo_3x4_backplay')[-1])),
    "networks exp03c frontier": ("reports/networks.md", lambda: "from {} to {} once".format(frontier_path('exp03c_ppo_3x4_backplay_relaxed')[0], frontier_path('exp03c_ppo_3x4_backplay_relaxed')[-1])),
    "networks exp04 frontier": ("reports/networks.md", lambda: "from {} to {} in {} advances".format(frontier_path('exp04_ppo_4x4')[0], frontier_path('exp04_ppo_4x4')[-1], len(frontier_path('exp04_ppo_4x4')) - 1)),
    "networks exp04 collapse": ("reports/networks.md", lambda: "from about {:.0f}M moves and to exactly zero from {:.0f}M".format(*(t / 1e6 for t in collapse_points()))),
    "networks exp04 length": ("reports/networks.md", lambda: f"about {half_up(det('exp04_ppo_4x4_best', 'fill_mean') * summary('exp04_ppo_4x4_best')['n_cells'])} cells (fill {det('exp04_ppo_4x4_best', 'fill_mean'):.3f})"),
    "networks exp05 seconds": ("reports/networks.md", lambda: f"about {half_up(log_seconds('exp05_bc_4x4'))} s"),
    "networks exp05 accuracy": ("reports/networks.md", lambda: re.search(r"expert-action accuracy [0-9.]+", log_lines('exp05_bc_4x4')[-1]).group(0)),
    "networks points short": ("reports/networks.md", lambda: f"{100 - 100 * det('exp02d_ppo_2x4_backplay_strict_best'):.1f} points short"),
    "networks fewer moves": ("reports/networks.md", lambda: f"{half_up(100 * (1 - det('exp02d_ppo_2x4_backplay_strict_best', 'steps_to_complete_mean') / det('exp01_route_2x4', 'steps_to_complete_mean')))} % fewer moves"),
    "networks fill ratios": ("reports/networks.md", lambda: "by {:.1f} to {:.1f} times and by {:.1f} times".format(min(map(fill_ratio, ON_3X4)), max(map(fill_ratio, ON_3X4)), fill_ratio('exp04_ppo_4x4_best'))),
    "networks fill range": ("reports/networks.md", lambda: "{} to {} % full".format(half_up(100 * min(det(r, 'fill_mean') for r in NEVER_FINISHED)), half_up(100 * max(det(r, 'fill_mean') for r in NEVER_FINISHED)))),
    "networks step change": ("reports/networks.md", lambda: f"changed its step count by {100 * (det('exp01_route_4x4', 'steps_to_complete_mean') - det('exp05b_ppo_4x4_from_bc_best', 'steps_to_complete_mean')) / det('exp01_route_4x4', 'steps_to_complete_mean'):.1f} %"),
    "networks floor moves": ("reports/networks.md", lambda: f"about {255 * geometric_floor(4):,.0f} moves"),
    "networks input sizes": ("reports/networks.md", lambda: f"{4 * 16 + 2} numbers on the smallest board and {4 * 256 + 2:,} on the largest"),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_quoted_number_matches_its_artifact(case):
    path, expected = CASES[case]
    text = (ROOT / path).read_text(encoding="utf-8")
    assert expected() in text, f"{path} does not contain {expected()!r}"


# --- reports/networks.md: every network's cells beside the loop and random baselines -----------
METRICS = {  # name: (value(run), printer(value), higher is better)
    "completion": (lambda r: 100 * det(r), lambda v: f"{v:.1f} %", True),
    "fill": (lambda r: det(r, "fill_mean"), lambda v: f"{v:.3f}", True),
    "steps": (lambda r: det(r, "steps_to_complete_mean"), lambda v: "never" if v is None else f"{v:,.1f}", False),
    "steps per food": (steps_per_food, lambda v: f"{v:.2f}", False),
    "times floor": (lambda r: steps_per_food(r) / geometric_floor(board_size(r)), lambda v: f"{v:.2f}x", False),
    "non-eating moves": (lambda r: 100 * non_eating_share(r), lambda v: f"{v:.1f} %", False),
    "won within 4C": (lambda r: 100 * won_within(r), lambda v: f"{v:.1f} %", True),
}


def verdict(printer, value, base, higher_is_better: bool) -> str:
    """The network against one baseline, judged at the precision the reader sees."""
    if printer(value) == printer(base):
        return "tie"
    if value is None or base is None:  # never finishing loses to finishing
        return "worse" if value is None else "better"
    return "better" if (value > base) == higher_is_better else "worse"


def cell(run: str, metric: str) -> str:
    """One score cell of reports/networks.md: bold network score, then loop and random with verdicts."""
    value, printer, higher = METRICS[metric]
    loop, rand = (value(f"exp01_{kind}_{board_size(run)}x4") for kind in ("route", "random"))
    v = value(run)
    if v is None:  # a network that never finished: no score of its own, only the baselines
        return f"never (loop {printer(loop)}; random {printer(rand)})"
    return (f"**{printer(v)}** (loop {printer(loop)}, {verdict(printer, v, loop, higher)}; "
            f"random {printer(rand)}, {verdict(printer, v, rand, higher)})")


@pytest.mark.parametrize("run, metric", [(r, m) for r in NETWORKS for m in METRICS])
def test_networks_row_quotes_its_cell(run, metric):
    lines = (ROOT / NETWORKS_DOC).read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("|") and run in line.split("|")[1]]
    assert any(cell(run, metric) in row for row in rows), f"{run}: no row contains {cell(run, metric)!r}"


@pytest.mark.parametrize("n", [2, 3, 4])
def test_geometric_floor_is_the_mean_manhattan_distance(n):
    cells = list(itertools.product(range(n), repeat=4))
    brute = sum(sum(abs(a - b) for a, b in zip(p, q, strict=True)) for p in cells for q in cells) / len(cells) ** 2
    assert geometric_floor(n) == pytest.approx(brute)


def test_mlp_params_matches_the_built_policy():
    cfg = Config(size=2, device="cpu")
    policy = build_model(cfg, make_env(cfg, 2, 0)).policy
    assert sum(p.numel() for p in policy.parameters()) == mlp_params(cfg.n_cells)


def test_networks_md_counts_quoted_in_words():
    """'twice', 'once', 'below 2e-5 at every update' and 'from 15 to 1' in reports/networks.md."""
    assert len(frontier_path("exp03b_ppo_3x4_backplay")) - 1 == 2
    assert len(frontier_path("exp03c_ppo_3x4_backplay_relaxed")) - 1 == 1
    assert frontier_path("exp02b_ppo_2x4_backplay")[::14] == [15, 1]
    assert max(v for _, v in progress_column("exp05b_ppo_4x4_from_bc", "train/approx_kl")) < 2e-5
