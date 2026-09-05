# Reinforcement-learning design and why

## Algorithm: MaskablePPO (sb3-contrib)

Snake in 4D has 8 actions of which several are fatal at almost every step. Invalid-action masking
(Huang and Ontanon 2020, https://arxiv.org/abs/2006.14171) removes them from the policy's
distribution instead of hoping the agent learns to avoid them; its benefit grows with the number
of invalid actions. The mask comes from the environment (`action_masks()`), so the agent only
learns *which legal move leads to a full board*.

PPO is on-policy and pairs naturally with thousands of batched environments; the reference agent
that fills a 12x12 board (linyiLYi/snake-ai) is also MaskablePPO.

## Network: MLP on the flat observation

The board is tiny (256 cells - fewer than the 784 pixels of an MNIST image - expanded to a
1026-float observation, about a 32x32 grayscale image) and PyTorch has no `Conv4d`
(https://docs.pytorch.org/docs/stable/nn.html#convolution-layers). A two-layer MLP of width `net_width = 512` for both the policy and the value head
(`net_arch = {"pi": [512, 512], "vf": [512, 512]}`) is the simplest sufficient choice; SB3's
default 64x64 (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) is
far too small for a 1026-dimensional input.

## Hyper-parameters (defaults in `config.py`)

| parameter | value | reason |
|---|---|---|
| `n_envs`, `n_steps` | 4096, 64 | batched env on CUDA peaks at 4096 envs (docs/benchmark.md); 262,144 samples per rollout |
| `batch_size` | 8192 | 32 minibatches x 4 epochs per rollout; the PPO update is the bottleneck, so larger minibatches raise throughput (minibatch sweep in docs/benchmark.md, exp00) |
| `n_epochs` | 4 | rollouts, not gradient steps, are cheap; more epochs mostly cost time |
| `gamma` | 0.99 | the recommended starting point (Andrychowicz et al. 2020, https://arxiv.org/abs/2006.05990); sweep 0.995/0.999 lowers `r_step` |
| `gae_lambda` | 0.95 | SB3 default (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters) |
| learning rate | linear 3e-4 -> 1e-5 | decaying LR + decaying clip range (below) is what lets the reference agent sharpen into near-deterministic endgame play |
| `clip_range` | linear 0.2 -> 0.05 | same |
| `ent_coef` | 0.01 | with heavy masking the entropy collapses fast; keep exploring |
| `target_kl` | 0.03 | a KL target as in Schulman et al. 2017 (https://arxiv.org/abs/1707.06347, adaptive-KL penalty); SB3's `target_kl` implements it as early stopping of the update (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters), a guard against late blow-ups under the decaying clip schedule |
| `vf_coef`, `max_grad_norm` | 0.5, 0.5 | SB3 defaults (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters) |

Schedules use SB3's `LinearSchedule(start, end, 1.0)` (`get_linear_fn` is deprecated since SB3
2.7.0: https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html).

## Reward

`+1` food, `-1` death, `+10` win, `-0.001` per step including the truncating step, never a death
penalty on truncation (see docs/game_rules.md).
The win bonus dominates the sum of food rewards on small boards so the objective is "fill the
board", not "eat a lot"; the step penalty removes the incentive to idle; deaths are the only
penalty. Distance-based progress bonuses are deliberately absent: the reference repository uses
one and it is exactly the term a circling policy can farm; if shaping is ever needed it is the
potential-based form, which is provably policy-invariant (Ng, Harada and Russell 1999,
https://dl.acm.org/doi/10.5555/645528.657613; `Phi(terminal) = 0` for episodic tasks: Grzes 2017,
https://dl.acm.org/doi/10.5555/3091125.3091208).

## Curriculum: Backplay along the Hamiltonian route

Filling 256 cells from length 1 is a sparse, very long-horizon task. Backplay (Resnick et al.
2018, https://arxiv.org/abs/1807.06919) starts episodes near the end of a demonstration and moves
the start backwards; Salimans and Chen (2018, https://arxiv.org/abs/1812.03381) gate the move on
the success rate. Here the demonstration is the Hamiltonian route:

- a fraction `1 - p_true_start` (default 0.8) of resets place the snake as a route segment of
  length `L ~ U[hi - window, hi]` (`window = 4`) at a random rotation; the rest start from length 1;
- `hi` starts at `C - 1` (clamped to `route_len - 1 = C - 2` on odd boards, whose route omits the
  corner); whenever the success rate of curriculum episodes (start length > 1,
  at least `curriculum_min_eps = 500` samples) exceeds `rho = 0.9`, `hi` decreases by
  `max(1, C // 64)`; at `hi = 1` the curriculum is over. The strict gate matters: with
  `rho = 0.2` the frontier races to 1 before the endgame is mastered and the agent gains nothing
  over plain PPO (exp02b vs exp02d);
- evaluation always starts from length 1 (`MaskableEvalCallback` uses a separate env that never
  receives a frontier), and `rollout/fill_mean_true_start` / `rollout/win_rate_true_start`
  track the true-start episodes during training, so distribution shift between curriculum
  starts and the real task is visible.

Board-size curriculum (2^4 -> 3^4 -> 4^4) is done by separate runs: the observation size differs,
so weights are not transferred.

## Imitation warm start (`snake4d imitate`)

exp04 showed that the reverse curriculum alone does not reach the true start on 4^4. The
`imitate` phase behaviour-clones (Pomerleau 1988,
https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html;
covariate-shift caveat and the DAgger remedy: Ross, Gordon and Bagnell 2011,
https://arxiv.org/abs/1011.0686) the same MaskablePPO policy on route-follower decisions collected
from the batched env with starts spread over every snake length (negative log-likelihood of the
expert action under the masked policy, Adam, `bc_epochs` passes over `n_envs * n_steps` samples),
saves `bc_model.zip`, and `train --set model_path=...` fine-tunes it with PPO. The clone starts
from a policy that already fills the board; RL then only has to make it faster.

## Evaluation

See docs/evaluation.md.
