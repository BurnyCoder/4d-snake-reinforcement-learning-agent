"""Training phase: MaskablePPO on the batched environment.

Global context: ``build_model`` turns a ``Config`` into an sb3-contrib ``MaskablePPO`` (shared by
the benchmark and by ``run``); ``run`` (the ``train`` phase) adds the SB3 logger, the curriculum and
evaluation callbacks and saves the model.  Design choices are documented in docs/rl_design.md.

Local notes:
* ``MaskablePPO`` API: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
* Linear decay of both the learning rate and the clip range (the reference snake agent
  linyiLYi/snake-ai decays both) via SB3's own ``LinearSchedule(start, end, end_fraction=1.0)``:
  https://stable-baselines3.readthedocs.io/en/master/common/utils.html
* ``torch.set_num_threads``: SB3 maintainers report large CPU speed-ups with one thread for small
  MLPs (https://github.com/DLR-RM/stable-baselines3/issues/121).
"""

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import VecEnv

from snake4d.config import Config


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
