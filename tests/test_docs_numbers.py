"""Headline numbers in README/paper/reports must be recomputable from reports/data artifacts.

Each case names a document and a formatter that derives the quoted string from an archived
summary.json / benchmark.json; the string must appear verbatim in the document.  Rounding follows
AGENTS.md: tables at artifact precision, prose as half-up integers or the same decimals.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

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
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_quoted_number_matches_its_artifact(case):
    path, expected = CASES[case]
    text = (ROOT / path).read_text(encoding="utf-8")
    assert expected() in text, f"{path} does not contain {expected()!r}"
