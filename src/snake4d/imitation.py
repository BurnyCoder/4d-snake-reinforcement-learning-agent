"""Imitation warm start: behaviour-clone the MaskablePPO policy on the route follower's decisions.

Global context: exp04 showed model-free PPO plus a reverse curriculum does not complete the 256-cell
board within 100M steps, while the scripted Hamiltonian route follower (``agents.RoutePolicy``)
completes it every time.  Behaviour cloning (supervised learning of the expert's action from the
observation) gives the network a policy that already fills the board; ``train`` can then fine-tune
it with PPO from ``SNAKE_MODEL_PATH``.

Local notes:
* Data: the batched env with curriculum-style starts spread uniformly over every snake length
  (``set_curriculum(hi=C-1, window=C-2, p_true_start=0.2)``), stepped by the route follower, so
  the dataset covers early, middle and endgame states in one pass of ``n_steps`` x ``n_envs``.
* Loss: negative log-likelihood of the expert action under the masked policy distribution -
  ``MaskableActorCriticPolicy.evaluate_actions(obs, actions, action_masks)`` returns exactly that
  log-probability (https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html); the value
  head is left untouched.  Optimiser: Adam (https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html).
"""

import logging
from pathlib import Path

import numpy as np
import torch

from snake4d.agents import RoutePolicy
from snake4d.config import Config
from snake4d.logging_utils import make_run_dir, setup_logging
from snake4d.train import build_model
from snake4d.vec_env import SnakeVecEnv, make_env

log = logging.getLogger("snake4d.imitation")


def collect(cfg: Config, n_envs: int, steps: int, seed: int):
    """Expert dataset ``(obs, actions, masks)`` from route-follower play on all-length starts."""
    env = SnakeVecEnv(cfg, n_envs, seed)
    env.set_curriculum(cfg.n_cells - 1, cfg.n_cells - 2, cfg.p_true_start)  # uniform lengths
    expert = RoutePolicy(cfg)
    obs = env.reset()
    observations, actions, masks = [], [], []
    for _ in range(steps):
        mask = env.action_masks()
        action, _ = expert.predict(obs, action_masks=mask)
        observations.append(obs.copy())
        actions.append(action.astype(np.int64))
        masks.append(mask.copy())
        obs, _, _, _ = env.step(action)
    return np.concatenate(observations), np.concatenate(actions), np.concatenate(masks)


def behaviour_clone(model, obs: np.ndarray, actions: np.ndarray, masks: np.ndarray,
                    epochs: int, batch_size: int, lr: float) -> float:
    """Supervised training of the policy head on the expert data; returns the final accuracy."""
    policy, device = model.policy, model.device
    obs_t = torch.as_tensor(obs, device=device)
    act_t = torch.as_tensor(actions, device=device)
    mask_t = torch.as_tensor(masks, device=device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(obs_t)
    accuracy = 0.0
    for epoch in range(epochs):
        order = torch.randperm(n, device=device)
        losses, correct = [], 0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            _, log_prob, _ = policy.evaluate_actions(obs_t[idx], act_t[idx],
                                                     action_masks=mask_t[idx])
            loss = -log_prob.mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
            with torch.no_grad():
                dist = policy.get_distribution(obs_t[idx], action_masks=mask_t[idx])
                correct += int((dist.get_actions(deterministic=True) == act_t[idx]).sum())
        accuracy = correct / n
        log.info("bc epoch %d/%d: loss %.4f accuracy %.4f", epoch + 1, epochs,
                 float(np.mean(losses)), accuracy)
    return accuracy


def run(cfg: Config) -> Path:
    """Phase entry point: collect expert data, clone it into a fresh MaskablePPO, save the model."""
    run_dir = make_run_dir(cfg, "imitate")
    logger = setup_logging(run_dir)
    obs, actions, masks = collect(cfg, cfg.n_envs, cfg.n_steps, cfg.seed)
    logger.info("collected %s expert samples on %d^%d", f"{len(obs):,}", cfg.size, cfg.ndim)
    model = build_model(cfg, make_env(cfg, cfg.n_envs, cfg.seed))
    accuracy = behaviour_clone(model, obs, actions, masks, cfg.bc_epochs, cfg.batch_size, cfg.bc_lr)
    model.save(run_dir / "bc_model.zip")
    logger.info("saved %s (expert-action accuracy %.3f); fine-tune with "
                "`snake4d train --set model_path=%s`", run_dir / "bc_model.zip", accuracy,
                run_dir / "bc_model.zip")
    return run_dir
