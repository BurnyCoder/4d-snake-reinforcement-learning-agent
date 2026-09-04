"""Configuration for snake4d.

Global context
--------------
Every tunable (board size, rewards, PPO hyper-parameters, curriculum, evaluation, paths) lives in
the single flat :class:`Config` dataclass below.  ``main.py`` builds one ``Config`` per phase from
defaults + ``.env`` + an optional per-experiment env file + ``--set`` overrides and hands it to the
phase function, so no other module reads environment variables on its own.

Local notes
-----------
* Fields are ``int`` / ``float`` / ``str`` only, so casting an environment string with
  ``f.type(...)`` is exact (``bool("False")`` would be ``True``).  The dataclass field type is the
  real class only because this file does NOT use ``from __future__ import annotations``.
* Environment keys are ``SNAKE_<FIELD>`` (upper-cased) so they cannot clash with other variables.
* python-dotenv layering: https://pypi.org/project/python-dotenv/ - ``load_dotenv()`` reads ``.env``
  without overriding existing shell variables; ``load_dotenv(path, override=True)`` layers an
  experiment file on top.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

ENV_PREFIX = "SNAKE_"  # every Config field maps to the environment key SNAKE_<FIELD_UPPER>


def _check(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a configuration guard fails (``assert`` can be stripped by -O)."""
    if not condition:
        raise ValueError(message)


@dataclass
class Config:
    """All tunables with their defaults; see .env.example for the one-line meaning of each key."""

    # --- game -------------------------------------------------------------------------------
    size: int = 4          # cells per axis
    ndim: int = 4          # number of dimensions
    idle_mult: int = 4     # starvation cap = idle_mult * n_cells (linyiLYi uses 4 * n_cells)
    # --- rewards ----------------------------------------------------------------------------
    r_food: float = 1.0
    r_death: float = -1.0
    r_win: float = 10.0
    r_step: float = -0.001
    shaping_coef: float = 0.0   # potential-based shaping weight (Ng et al. 1999); 0 = off
    # --- PPO --------------------------------------------------------------------------------
    n_envs: int = 4096          # docs/benchmark.md: batched env on CUDA peaks at 4096
    total_timesteps: int = 20_000_000
    n_steps: int = 64           # rollout = 64 * 4096 = 262,144 samples
    batch_size: int = 8192      # docs/benchmark.md: 2048 -> 8192 raises fps 21k -> 34k
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr_start: float = 3e-4
    lr_end: float = 1e-5
    clip_start: float = 0.2
    clip_end: float = 0.05
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    net_width: int = 512
    device: str = "auto"        # cuda when available (measured 6x faster than cpu)
    torch_threads: int = 8      # docs/benchmark.md: 8 threads beat 1 on every CPU row
    seed: int = 0
    # --- curriculum -------------------------------------------------------------------------
    curriculum: int = 1
    curriculum_window: int = 4     # exp02d: narrow window + strict gate beat the loose defaults
    curriculum_delta: int = 0   # 0 -> max(1, n_cells // 64), see curriculum_step
    curriculum_rho: float = 0.9    # advance only once the frontier is mastered (exp02d)
    curriculum_min_eps: int = 500
    p_true_start: float = 0.2
    # --- evaluation -------------------------------------------------------------------------
    eval_episodes: int = 100
    eval_every: int = 2_097_152  # 8 rollouts of 262,144 (about a minute at 34k fps)
    ckpt_every: int = 8_388_608  # 32 rollouts
    eval_seeds: str = "0,1,2"
    bench_steps: int = 200_000   # PPO timesteps per benchmark row
    # --- imitation warm start (phase `imitate`) ---------------------------------------------
    bc_epochs: int = 20          # passes over the n_envs * n_steps expert samples
    bc_lr: float = 1e-3          # Adam learning rate for behaviour cloning
    # --- paths / phase arguments ------------------------------------------------------------
    runs_dir: str = "runs"
    run_name: str = "run"
    model_path: str = ""
    policy: str = "route"

    def __post_init__(self) -> None:
        """Guards that catch silently-wrong experiments before any compute is spent."""
        _check(self.size >= 2 and self.ndim >= 1, "size must be >= 2 and ndim >= 1")
        _check(0.0 < self.gamma < 1.0, "gamma must be in (0, 1)")
        # Discounted cost of wandering forever must stay below the death penalty, otherwise dying
        # becomes preferable to walking (the gamma=0.999 sweep must lower r_step accordingly).
        wander_cost = round(abs(self.r_step) / (1.0 - self.gamma), 9)  # rounded: 0.001/0.001 == 1
        _check(
            wander_cost < abs(self.r_death),
            f"|r_step|/(1-gamma) = {wander_cost:.3f} must be < |r_death| = {abs(self.r_death)}",
        )
        # SB3 warns when the rollout size is not a multiple of batch_size; make it an error.
        _check((self.n_steps * self.n_envs) % self.batch_size == 0,
               "n_steps * n_envs must be a multiple of batch_size")
        _check(self.eval_every >= self.n_steps * self.n_envs,
               "eval_every must be >= one rollout (n_steps * n_envs)")
        _check(0.0 <= self.p_true_start <= 1.0, "p_true_start must be in [0, 1]")

    # --- derived quantities (single definitions used everywhere) -----------------------------
    @property
    def n_cells(self) -> int:
        """Number of board cells C = size ** ndim (256 for the default 4^4 board)."""
        return self.size**self.ndim

    @property
    def n_actions(self) -> int:
        """One +/- move per axis: 2 * ndim actions (8 in 4D)."""
        return 2 * self.ndim

    @property
    def idle_cap(self) -> int:
        """Steps allowed without eating before the episode is truncated."""
        return self.idle_mult * self.n_cells

    @property
    def max_steps(self) -> int:
        """Absolute episode cap C*C; a cycle follower needs ~C*C/2 so it still completes."""
        return self.n_cells * self.n_cells

    @property
    def curriculum_step(self) -> int:
        """Frontier decrement per curriculum advance (config value or max(1, C // 64))."""
        return self.curriculum_delta or max(1, self.n_cells // 64)

    @property
    def seeds(self) -> tuple[int, ...]:
        """Evaluation seeds parsed from the comma-separated ``eval_seeds`` string."""
        return tuple(int(s) for s in self.eval_seeds.split(",") if s.strip())

    # --- construction / persistence ----------------------------------------------------------
    @classmethod
    def from_env(cls, env_file: str | None = None, overrides: tuple[str, ...] = ()) -> "Config":
        """Build a Config from defaults, ``.env``, an optional experiment env file and overrides.

        ``overrides`` are ``field=value`` strings (from ``--set``); they are written into
        ``os.environ`` so a single casting path (below) handles every source.
        """
        # Only the project's own .env: the zero-argument load_dotenv() walks up parent directories
        # and would load an unrelated .env from outside the repository.
        load_dotenv(Path.cwd() / ".env")  # a missing file is a no-op; shell variables keep priority
        if env_file:
            _check(Path(env_file).is_file(), f"env file not found: {env_file}")
            load_dotenv(env_file, override=True)  # experiment file layers on top of .env
        names = {f.name for f in fields(cls)}
        for item in overrides:
            key, _, value = item.partition("=")
            name = key.strip().lower().removeprefix(ENV_PREFIX.lower())
            _check(name in names, f"unknown config field: {key} (known: {sorted(names)})")
            os.environ[ENV_PREFIX + name.upper()] = value.strip()
        # Exact-type casting: f.type is int/float/str, see module docstring.
        values = {
            f.name: f.type(os.environ[ENV_PREFIX + f.name.upper()])
            for f in fields(cls)
            if ENV_PREFIX + f.name.upper() in os.environ
        }
        return cls(**values)

    def to_json(self, path: Path) -> None:
        """Write the resolved configuration next to a run's logs for reproducibility."""
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")
