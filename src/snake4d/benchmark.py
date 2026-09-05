"""Benchmark phase: measure environment and PPO throughput on this machine before any training.

Global context: no published steps/second figure exists for a numpy-batched snake vs SB3's
DummyVecEnv/SubprocVecEnv on Windows, so the architecture decision (single-process batched env)
and every training budget are justified by numbers measured here and written to
``runs/<ts>_bench_*/benchmark.json`` and, generated from it, ``docs/benchmark.md``.

Local notes: rows = ({batched SnakeVecEnv at 256/1024/4096} + {DummyVecEnv(16),
SubprocVecEnv(16)}) x device x torch threads, followed by a minibatch-size sweep on the best
row (``bench_batch_sizes``).
PPO fps = ``model.num_timesteps / wall time`` of ``learn()`` (SB3's own ``time/fps`` is cumulative
and would double count when averaged).  SubprocVecEnv needs the ``if __name__ == "__main__"``
guard on Windows (spawn), which the console script provides.  ``docs/benchmark.md`` is written
only by ``write_markdown`` so that every number in it has the JSON artifact behind it.
"""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from snake4d.config import Config
from snake4d.env import SnakeEnv
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.train import build_model
from snake4d.vec_env import INFO_KEYS, SnakeVecEnv, make_env

ROWS = [
    {"env": "batched", "n_envs": 256},
    {"env": "batched", "n_envs": 1024},
    {"env": "batched", "n_envs": 4096},
    {"env": "dummy", "n_envs": 16},
    {"env": "subproc", "n_envs": 16},
]
log = logging.getLogger("snake4d.benchmark")
PPO_TABLE_HEADER = (
    "| env | n_envs | device | threads | n_steps | batch | timesteps | seconds | fps |",
    "|---|---|---|---|---|---|---|---|---|",
)


def make_row_env(cfg: Config, row: dict, seed: int = 0):
    """Batched env, or SB3's DummyVecEnv / SubprocVecEnv over single SnakeEnvs."""
    if row["env"] == "batched":
        return make_env(cfg, row["n_envs"], seed)
    vec_env_cls = DummyVecEnv if row["env"] == "dummy" else SubprocVecEnv
    return make_vec_env(SnakeEnv, n_envs=row["n_envs"], seed=seed, env_kwargs={"cfg": cfg},
                        monitor_kwargs={"info_keywords": INFO_KEYS}, vec_env_cls=vec_env_cls)


def env_steps_per_second(cfg: Config, n_envs: int, steps: int = 200) -> float:
    """Env-only throughput of the batched env with random legal actions (no policy, no PPO)."""
    env = SnakeVecEnv(cfg, n_envs, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(steps):
        masks = env.action_masks()
        scores = np.where(masks, rng.random(masks.shape), -1.0)  # random legal action per row
        env.step(scores.argmax(axis=1))
    return steps * n_envs / (time.perf_counter() - start)


def ppo_fps(cfg: Config, row: dict, device: str, threads: int, timesteps: int,
            batch_size: int | None = None) -> dict:
    """Train MaskablePPO briefly on one row; returns fps and the settings that produced it."""
    env = make_row_env(cfg, row)
    n_steps = cfg.n_steps if row["env"] == "batched" else 512  # keep rollouts ~ 8k+ samples
    rollout = n_steps * row["n_envs"]
    batch_size = min(batch_size or cfg.batch_size, rollout)
    cfg_threads = Config(**{**cfg.__dict__, "torch_threads": threads})
    model = build_model(cfg_threads, env, n_steps=n_steps, batch_size=batch_size, device=device)
    start = time.perf_counter()
    try:
        model.learn(total_timesteps=timesteps)
    finally:
        env.close()  # SubprocVecEnv workers must die even when a row fails
    elapsed = time.perf_counter() - start
    return {**row, "device": device, "threads": threads, "n_steps": n_steps,
            "batch_size": batch_size, "timesteps": int(model.num_timesteps),
            "seconds": round(elapsed, 2), "fps": round(model.num_timesteps / elapsed)}


def recommend(results: list[dict]) -> dict:
    """Best row -> N_ENVS/N_STEPS/DEVICE/TORCH_THREADS; EVAL_EVERY ~ one evaluation per 10 min,
    CKPT_EVERY = 4x that."""
    successful = [r for r in results if "fps" in r]
    if not successful:  # every row failed: keep the measurements, recommend nothing
        return {"best_row": None}
    best = max(successful, key=lambda r: r["fps"])
    rollout = best["n_steps"] * best["n_envs"]
    eval_every = max(rollout, round(best["fps"] * 600 / rollout) * rollout)
    return {"SNAKE_N_ENVS": best["n_envs"], "SNAKE_DEVICE": best["device"],
            "SNAKE_TORCH_THREADS": best["threads"], "SNAKE_N_STEPS": best["n_steps"],
            "SNAKE_EVAL_EVERY": eval_every, "SNAKE_CKPT_EVERY": 4 * eval_every,
            "best_row": best}


def minibatch_sweep(cfg: Config, best: dict | None, timesteps: int) -> list[dict]:
    """PPO fps of the best batched row for every minibatch size in ``cfg.bench_batches``."""
    results: list[dict] = []
    if not best or best["env"] != "batched":
        return results
    rollout = best["n_steps"] * best["n_envs"]
    for size in cfg.bench_batches:
        if rollout % size:
            log.warning("minibatch %d skipped: it does not divide the rollout of %d", size, rollout)
            continue
        try:
            result = ppo_fps(cfg, {"env": best["env"], "n_envs": best["n_envs"]}, best["device"],
                             best["threads"], timesteps, batch_size=size)
        except Exception as exc:  # noqa: BLE001 - one bad size must not stop the sweep
            result = {**best, "batch_size": size, "error": repr(exc)}
        results.append(result)
        log.info("%s", json.dumps(result))
    return results


def machine_info() -> dict:
    """What the numbers were measured on: torch threads, logical CPUs, GPU name/memory, torch."""
    cuda = torch.cuda.is_available()
    return {
        "torch_threads": torch.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if cuda else "none",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        if cuda else None,
        "torch": torch.__version__,
    }


def _row_line(r: dict) -> str:
    """One markdown table row for a PPO measurement (or its error)."""
    if "fps" in r:
        return (f"| {r['env']} | {r['n_envs']} | {r['device']} | {r['threads']} | "
                f"{r['n_steps']} | {r['batch_size']} | {r['timesteps']:,} | "
                f"{r['seconds']} | {r['fps']:,} |")
    return (f"| {r['env']} | {r['n_envs']} | {r['device']} | {r['threads']} | "
            f"- | {r.get('batch_size', '-')} | - | - | error: {r['error']} |")


def write_markdown(cfg: Config, results: list[dict], env_only: dict, rec: dict, path: Path,
                   machine: dict, minibatch: list[dict] | None = None) -> None:
    """docs/benchmark.md from the measurements: grid table, minibatch sweep, recommended values."""
    lines = [
        "# Throughput benchmark", "",
        f"Board {cfg.size}^{cfg.ndim} ({cfg.n_cells} cells), MLP [{cfg.net_width}, "
        f"{cfg.net_width}], n_epochs={cfg.n_epochs}. Generated by `uv run snake4d bench` from "
        "`benchmark.json` (do not edit by hand).", "",
        "Machine: " + ", ".join(f"{k} {v}" for k, v in machine.items()), "",
        "## Environment only (batched SnakeVecEnv, random legal actions)", "",
        "| n_envs | env steps / s |", "|---|---|",
        *[f"| {n} | {round(v):,} |" for n, v in env_only.items()], "",
        "## MaskablePPO end to end (rollout + update), fps = timesteps / wall time", "",
        *PPO_TABLE_HEADER,
        *[_row_line(r) for r in results],
    ]
    if minibatch:
        lines += ["", "## Minibatch size on the best row (same env, device, threads and timesteps)",
                  "", *PPO_TABLE_HEADER, *[_row_line(r) for r in minibatch]]
    lines += ["", "## Recommended .env values (best row of the grid)", "",
              *[f"- `{k}={v}`" for k, v in rec.items() if k != "best_row"], ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(cfg: Config, rows: list[dict] | None = None, devices: list[str] | None = None,
        threads: list[int] | None = None, timesteps: int | None = None,
        docs_path: Path = Path("docs/benchmark.md")) -> dict:
    """Phase entry point: run the grid and the sweep, write benchmark.json + docs/benchmark.md."""
    run_dir = make_run_dir(cfg, "bench")
    logger = setup_logging(run_dir)
    rows = rows or ROWS
    devices = devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    threads = threads or [1, torch.get_num_threads()]
    timesteps = timesteps or cfg.bench_steps
    machine = machine_info()
    env_only = {}
    for row in rows:
        if row["env"] == "batched":
            env_only[row["n_envs"]] = env_steps_per_second(cfg, row["n_envs"])
            logger.info("env-only batched n_envs=%d: %.0f steps/s", row["n_envs"],
                        env_only[row["n_envs"]])
    results = []
    for row in rows:
        for device in devices:
            for n_threads in threads:
                try:
                    result = ppo_fps(cfg, row, device, n_threads, timesteps)
                except Exception as exc:  # noqa: BLE001 - one bad row must not stop the grid
                    result = {**row, "device": device, "threads": n_threads, "error": repr(exc)}
                results.append(result)
                logger.info("%s", json.dumps(result))
    rec = recommend(results)
    logger.info("recommended: %s", json.dumps(rec))
    minibatch = minibatch_sweep(cfg, rec.get("best_row"), timesteps)
    (run_dir / "benchmark.json").write_text(
        json.dumps({"machine": machine, "env_only": env_only, "results": results,
                    "recommended": rec, "minibatch": minibatch}, indent=2), encoding="utf-8")
    write_markdown(cfg, results, env_only, rec, docs_path, machine, minibatch)
    logger.info("wrote %s", docs_path)
    return rec
