"""Training phase: MaskablePPO on the batched environment.

Global context: ``build_model`` turns a ``Config`` into an sb3-contrib ``MaskablePPO`` (shared by
the benchmark and by ``run``); ``run`` (the ``train`` phase) adds the SB3 logger, the curriculum and
evaluation callbacks, trains for ``total_timesteps`` and saves the model.  Design choices are
documented in docs/rl_design.md.

Local notes:
* ``MaskablePPO`` API: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
* Linear decay of both the learning rate and the clip range (the reference snake agent
  linyiLYi/snake-ai decays both) via SB3's own ``LinearSchedule(start, end, end_fraction=1.0)``:
  https://stable-baselines3.readthedocs.io/en/master/common/utils.html
* ``torch.set_num_threads``: SB3 maintainers report large CPU speed-ups with one thread for small
  MLPs (https://github.com/DLR-RM/stable-baselines3/issues/121).
* Logger ``configure(run_dir, ["stdout", "log", "csv", "tensorboard"])`` writes the console
  tables, ``log.txt``, ``progress.csv`` and TensorBoard events into the run directory:
  https://stable-baselines3.readthedocs.io/en/master/common/logger.html
* ``MaskableEvalCallback`` evaluates with masks on the true-start eval env and saves
  ``best_model.zip`` + ``evaluations.npz``; ``eval_freq``/``save_freq`` are per vec-step, hence the
  division by ``n_envs`` (https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html).
"""

import logging
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecEnv

from snake4d.callbacks import Backplay, FillLogger
from snake4d.config import Config
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.vec_env import make_env

log = logging.getLogger("snake4d.train")


def build_model(cfg: Config, env: VecEnv, n_steps: int | None = None,
                batch_size: int | None = None, device: str | None = None,
                tensorboard_log: str | None = None, verbose: int = 0) -> MaskablePPO:
    """MaskablePPO with the project's hyper-parameters (overridable rollout shape and device)."""
    torch.set_num_threads(cfg.torch_threads)
    width = cfg.net_width
    return MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=n_steps or cfg.n_steps,
        batch_size=batch_size or cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=LinearSchedule(cfg.lr_start, cfg.lr_end, 1.0),
        clip_range=LinearSchedule(cfg.clip_start, cfg.clip_end, 1.0),
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
        policy_kwargs={"net_arch": {"pi": [width, width], "vf": [width, width]}},
        device=device or cfg.device,
        seed=cfg.seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
    )


def build_callbacks(cfg: Config, run_dir: Path) -> list:
    """Fill dashboard, masked evaluation from the true start, checkpoints, optional curriculum."""
    eval_env = make_env(cfg, cfg.eval_episodes, seed=cfg.seed + 1000)  # never gets the curriculum
    callbacks = [
        FillLogger(),
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=cfg.eval_episodes,
            eval_freq=max(cfg.eval_every // cfg.n_envs, 1),
            deterministic=True,
            log_path=str(run_dir / "eval"),
            best_model_save_path=str(run_dir),
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(cfg.ckpt_every // cfg.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="model",
        ),
    ]
    if cfg.curriculum:
        callbacks.append(Backplay(cfg))
    return callbacks


def run(cfg: Config) -> Path:
    """Phase entry point: train (or resume from ``model_path``), save ``final_model.zip``."""
    run_dir = make_run_dir(cfg, "train")
    logger = setup_logging(run_dir)
    env = make_env(cfg, cfg.n_envs, seed=cfg.seed, monitor_path=str(run_dir / "monitor"))
    if cfg.model_path:
        # custom_objects replaces the hyper-parameters stored in the checkpoint with this run's
        # Config (otherwise the saved schedules/entropy would silently apply):
        # https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html
        torch.set_num_threads(cfg.torch_threads)
        model = MaskablePPO.load(cfg.model_path, env=env, device=cfg.device, custom_objects={
            "learning_rate": LinearSchedule(cfg.lr_start, cfg.lr_end, 1.0),
            "clip_range": LinearSchedule(cfg.clip_start, cfg.clip_end, 1.0),
            "ent_coef": cfg.ent_coef, "gamma": cfg.gamma, "target_kl": cfg.target_kl,
            "n_steps": cfg.n_steps, "batch_size": cfg.batch_size, "n_epochs": cfg.n_epochs,
        })
        logger.info("resumed %s with this run's schedules and hyper-parameters", cfg.model_path)
    else:
        model = build_model(cfg, env)
    model.set_logger(configure(str(run_dir), ["stdout", "log", "csv", "tensorboard"]))
    logger.info("training %d^%d (%d cells) for %s steps on %s: n_envs=%d n_steps=%d batch=%d "
                "curriculum=%d", cfg.size, cfg.ndim, cfg.n_cells, f"{cfg.total_timesteps:,}",
                model.device, cfg.n_envs, cfg.n_steps, cfg.batch_size, cfg.curriculum)
    model.learn(total_timesteps=cfg.total_timesteps, callback=build_callbacks(cfg, run_dir),
                reset_num_timesteps=not cfg.model_path)
    model.save(run_dir / "final_model")
    logger.info("saved %s after %s timesteps", run_dir / "final_model.zip",
                f"{model.num_timesteps:,}")
    return run_dir
