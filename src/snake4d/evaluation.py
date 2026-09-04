"""Evaluation phase: measure how often (and how fast) a policy completes the board.

Global context: the headline numbers of every report come from here - ``success_rate`` (board
filled within the episode caps), mean ``fill`` and steps-to-complete - for scripted baselines
(``agents.py``) and saved MaskablePPO models alike.  ``train.py`` reuses ``make_eval_env`` for its
in-training evaluation callback.

Local notes:
* ``sb3_contrib.common.maskable.evaluation.evaluate_policy`` (not SB3's) so action masks are
  applied; it requires a Monitor-wrapped env and reads ``info["episode"]`` at episode end -
  https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/evaluation.py
* ``make_vec_env(..., n_envs=eval_episodes)`` runs one episode per env with one batched policy
  call per step; ``seed`` seeds every sub-env at its first reset -
  https://stable-baselines3.readthedocs.io/en/master/common/env_util.html
* Episode counts and seeds are explicit (100 episodes x 3 seeds), never SB3's defaults of 10/5.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from snake4d.agents import POLICIES
from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.logging_utils import make_run_dir, setup_logging

INFO_KEYS = ("is_success", "fill", "start_len")  # Monitor copies these into info["episode"]
log = logging.getLogger("snake4d.evaluation")


def make_eval_env(cfg: Config, n_envs: int, seed: int | None) -> VecEnv:
    """``n_envs`` Monitor-wrapped SnakeEnvs in a DummyVecEnv, seeded for their first reset."""
    return make_vec_env(
        SnakeEnv,
        n_envs=n_envs,
        seed=seed,
        env_kwargs={"cfg": cfg},
        monitor_kwargs={"info_keywords": INFO_KEYS},
    )


def run_episodes(policy, cfg: Config, seed: int, deterministic: bool) -> list[dict]:
    """One evaluation pass: ``cfg.eval_episodes`` episodes, one per env, as a list of records."""
    env = make_eval_env(cfg, cfg.eval_episodes, seed)
    policy.set_random_seed(seed)
    episodes: list[dict] = []

    def on_step(locals_: dict, _globals: dict) -> None:
        if locals_["done"]:  # Monitor adds info["episode"] = {r, l, t, *INFO_KEYS} at episode end
            ep = locals_["info"]["episode"]
            episodes.append(
                {
                    "seed": seed,
                    "deterministic": int(deterministic),
                    "success": int(ep["is_success"]),
                    "fill": float(ep["fill"]),
                    "length": int(ep["l"]),
                    "return": float(ep["r"]),
                }
            )

    evaluate_policy(
        policy,
        env,
        n_eval_episodes=cfg.eval_episodes,
        deterministic=deterministic,
        callback=on_step,
        warn=True,
    )
    env.close()
    return episodes


def summarize(episodes: list[dict], cfg: Config) -> dict:
    """Success rate, mean fill, steps-to-complete on wins and win-within-K step budgets."""
    success = np.array([e["success"] for e in episodes], dtype=float)
    lengths = np.array([e["length"] for e in episodes], dtype=float)
    wins = lengths[success == 1]
    budgets = {"C": cfg.n_cells, "2C": 2 * cfg.n_cells, "4C": 4 * cfg.n_cells,
               "C2/2": cfg.n_cells**2 // 2}
    return {
        "episodes": len(episodes),
        "success_rate": float(success.mean()),
        "fill_mean": float(np.mean([e["fill"] for e in episodes])),
        "return_mean": float(np.mean([e["return"] for e in episodes])),
        "length_mean": float(lengths.mean()),
        "steps_to_complete_mean": float(wins.mean()) if wins.size else None,
        "win_within": {k: float((wins <= v).sum() / len(episodes)) for k, v in budgets.items()},
    }


def evaluate(policy, cfg: Config, seeds: tuple[int, ...], modes: tuple[bool, ...] = (True,)):
    """Full protocol: every seed x every mode; returns ``(summary, episode_records)``."""
    records: list[dict] = []
    per_run: dict[str, dict] = {}
    for deterministic in modes:
        for seed in seeds:
            episodes = run_episodes(policy, cfg, seed, deterministic)
            key = f"deterministic={deterministic}/seed={seed}"
            per_run[key] = summarize(episodes, cfg)
            log.info("%s: success=%.3f fill=%.3f steps_to_complete=%s", key,
                     per_run[key]["success_rate"], per_run[key]["fill_mean"],
                     per_run[key]["steps_to_complete_mean"])
            records += episodes
    summary = {"board": f"{cfg.size}^{cfg.ndim}", "n_cells": cfg.n_cells, "seeds": list(seeds),
               "per_run": per_run, "modes": {}}
    for deterministic in modes:
        rates = [per_run[f"deterministic={deterministic}/seed={s}"]["success_rate"] for s in seeds]
        subset = [e for e in records if e["deterministic"] == int(deterministic)]
        summary["modes"][f"deterministic={deterministic}"] = {
            "success_rate_mean": float(np.mean(rates)),
            "success_rate_std": float(np.std(rates)),
            **summarize(subset, cfg),
        }
    return summary, records


def write_results(run_dir: Path, summary: dict, records: list[dict]) -> None:
    """Persist ``summary.json`` and one CSV row per episode next to the run's log."""
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(run_dir / "eval_episodes.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_policy(cfg: Config):
    """A saved MaskablePPO (evaluated deterministic + stochastic) or a scripted policy."""
    if cfg.model_path:
        from sb3_contrib import MaskablePPO  # lazy: scripted baselines never need torch

        return MaskablePPO.load(cfg.model_path, device=cfg.device), (True, False)
    return POLICIES[cfg.policy](cfg), (True,)


def run(cfg: Config) -> dict:
    """Phase entry point: evaluate ``cfg.model_path`` or ``cfg.policy`` and write the results."""
    run_dir = make_run_dir(cfg, "evaluate")
    logger = setup_logging(run_dir)
    policy, modes = load_policy(cfg)
    logger.info("evaluating %s on %d^%d (%d cells), %d episodes x seeds %s, modes %s",
                cfg.model_path or cfg.policy, cfg.size, cfg.ndim, cfg.n_cells, cfg.eval_episodes,
                cfg.seeds, modes)
    summary, records = evaluate(policy, cfg, cfg.seeds, modes)
    write_results(run_dir, summary, records)
    logger.info("summary:\n%s", json.dumps(summary["modes"], indent=1))
    logger.info("results written to %s", run_dir)
    return summary
