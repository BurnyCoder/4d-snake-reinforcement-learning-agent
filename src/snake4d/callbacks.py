"""Training callbacks: fill-fraction dashboards and the Backplay reverse curriculum.

Global context: SB3 only logs episode return/length (and ``rollout/success_rate`` from
``is_success``).  The objective here is *board fill*, and the curriculum changes where episodes
start, so ``FillLogger`` records fill split by start type and ``Backplay`` moves the curriculum
frontier.  Both read finished episodes from ``self.locals["infos"]`` (the ``info["episode"]`` dicts
VecMonitor writes) and never touch SB3's own buffers.

Local notes:
* Callback API: https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html -
  ``_on_step`` runs after every vec-step with ``self.locals`` = the rollout loop's locals,
  ``_on_rollout_end`` runs before SB3 dumps the iteration's logs, ``self.training_env`` is the
  VecEnv being trained on.
* Backplay (Resnick et al., https://arxiv.org/abs/1807.06919): start episodes near the end of a
  demonstration (here: as a prefix of the Hamiltonian route) and move the start backwards.
  Success-gated advance as in Salimans & Chen (https://arxiv.org/abs/1812.03381): move the reset
  point back when the success rate of curriculum episodes exceeds ``rho``.
"""

import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from snake4d.config import Config

log = logging.getLogger("snake4d.callbacks")


def finished_episodes(infos: list[dict]) -> list[dict]:
    """The ``info["episode"]`` dicts of envs that ended this step (VecMonitor adds them)."""
    return [info["episode"] for info in infos if "episode" in info]


class FillLogger(BaseCallback):
    """rollout/fill_mean, plus fill and win rate of episodes that began at the true start."""

    def __init__(self) -> None:
        super().__init__()
        self.fills: list[float] = []
        self.true_fills: list[float] = []
        self.true_wins: list[float] = []

    def _on_step(self) -> bool:
        for ep in finished_episodes(self.locals["infos"]):
            self.fills.append(ep["fill"])
            if ep["start_len"] == 1:
                self.true_fills.append(ep["fill"])
                self.true_wins.append(float(ep["is_success"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.fills:
            self.logger.record("rollout/fill_mean", float(np.mean(self.fills)))
        if self.true_fills:
            self.logger.record("rollout/fill_mean_true_start", float(np.mean(self.true_fills)))
            self.logger.record("rollout/win_rate_true_start", float(np.mean(self.true_wins)))
        self.logger.record("rollout/episodes", len(self.fills))
        self.fills.clear()
        self.true_fills.clear()
        self.true_wins.clear()


class Backplay(BaseCallback):
    """Success-gated reverse curriculum along the Hamiltonian route (frontier ``hi`` -> 1)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.hi = cfg.n_cells - 1  # start one cell short of a full board
        self.results: list[float] = []  # successes of curriculum episodes (start_len > 1)

    def _push(self) -> None:
        self.training_env.env_method("set_curriculum", self.hi, self.cfg.curriculum_window,
                                     self.cfg.p_true_start)
        # read back the effective frontier: the simulator clamps it to route_len - 1 (odd boards)
        self.hi = int(self.training_env.get_attr("sim")[0].cur_hi)

    def _on_training_start(self) -> None:
        self._push()
        log.info("curriculum: frontier %d, window %d, p_true_start %.2f", self.hi,
                 self.cfg.curriculum_window, self.cfg.p_true_start)

    def _on_step(self) -> bool:
        for ep in finished_episodes(self.locals["infos"]):
            if ep["start_len"] > 1:
                self.results.append(float(ep["is_success"]))
        return True

    def _on_rollout_end(self) -> None:
        rate = float("nan")
        if len(self.results) >= self.cfg.curriculum_min_eps:
            rate = float(np.mean(self.results))
            if rate > self.cfg.curriculum_rho and self.hi > 1:
                self.hi = max(1, self.hi - self.cfg.curriculum_step)
                self.results.clear()
                self._push()
                log.info("curriculum: success %.2f > rho, frontier -> %d", rate, self.hi)
            else:
                self.results = self.results[-self.cfg.curriculum_min_eps :]  # rolling window
        self.logger.record("curriculum/frontier", self.hi)
        self.logger.record("curriculum/success_rate", rate)
