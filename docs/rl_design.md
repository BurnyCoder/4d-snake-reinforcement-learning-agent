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
(`net_arch = {"pi": [512, 512], "vf": [512, 512]}`) is the chosen baseline; SB3's
default 64x64 (https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) is
an untested alternative here. Input dimension alone does not establish that it is insufficient.

## Hyper-parameters (defaults in `config.py`)

| parameter | value | reason |
|---|---|---|
| `n_envs`, `n_steps` | 4096, 64 | batched env on CUDA peaks at 4096 envs (docs/benchmark.md); 262,144 samples per rollout |
| `batch_size` | 8192 | 32 minibatches x 4 epochs per rollout; the PPO update is the bottleneck, so larger minibatches raise throughput (minibatch sweep in docs/benchmark.md, exp00) |
| `n_epochs` | 4 | rollouts, not gradient steps, are cheap; more epochs mostly cost time |
| `gamma` | 0.99 | a starting point from Andrychowicz et al. 2020 (https://arxiv.org/abs/2006.05990); higher discounts were not ablated here |
| `gae_lambda` | 0.95 | SB3 default (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters) |
| learning rate | linear 3e-4 -> 1e-5 | chosen schedule; its causal contribution has not been isolated in these runs |
| `clip_range` | linear 0.2 -> 0.05 | chosen schedule; no fixed-schedule ablation |
| `ent_coef` | 0.01 | with heavy masking the entropy collapses fast; keep exploring |
| `target_kl` | 0.03 | a KL target as in Schulman et al. 2017 (https://arxiv.org/abs/1707.06347, adaptive-KL penalty); SB3's `target_kl` implements it as early stopping of the update (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters), a guard against late blow-ups under the decaying clip schedule |
| `vf_coef`, `max_grad_norm` | 0.5, 0.5 | SB3 defaults (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters) |

Schedules use SB3's `LinearSchedule(start, end, 1.0)` (`get_linear_fn` is deprecated since SB3
2.7.0: https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html).

## Reward

`+1` food, `-1` death, `+10` win, `-0.001` per step including the truncating step, never a death
penalty on truncation (see docs/game_rules.md).
The completion bonus does not dominate the food-reward total even on the smallest evaluated
board: filling a 2^4 board from length 1 earns 15 food rewards and a 10-point bonus. PPO optimizes
discounted return (`gamma`), whereas the scientific outcome is completion within the caps;
maximizing one is not proven equivalent to maximizing the other. The step penalty discourages
longer trajectories but does not guarantee completion. Checkpoint selection uses mean
undiscounted evaluation return, not the completion metric ([evaluation protocol](evaluation.md)).
Distance-based progress bonuses are off in these experiments. Optional potential-based shaping
is implemented in `physics.py` but disabled by default; its
policy-invariance requires the appropriate assumptions and terminal treatment (Ng, Harada and
Russell 1999,
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
  `max(1, C // 64)`; at `hi = 1` the curriculum is over. This describes the configured gate,
  not verified mastery of each stage;
- evaluation always starts from length 1 (`MaskableEvalCallback` uses a separate env that never
  receives a frontier), and `rollout/fill_mean_true_start` / `rollout/win_rate_true_start`
  track the true-start episodes during training separately. These scores do not isolate the
  effect of start-distribution differences.

The current callback groups episodes by completion time and start length, without retaining the
frontier active when each episode reset. An in-flight episode from an older, easier frontier can
therefore enter the next gate after the result list is cleared. The
[scientific audit](../reports/scientific_audit.md) demonstrates contaminated exp04 gates; historical
advances cannot be interpreted as current-stage mastery. The exp02b/exp02d comparison also changes
threshold, window and minimum count together, so it does not isolate a threshold effect. Reset
probabilities are not transition shares: longer episodes contribute more training steps.

Before a future claim about stage mastery, attach a generation identifier at each reset, preserve
it in terminal episode records, and count only episodes from the active frontier generation in
its gate. This is a required future change, not an implemented fix; existing checkpoints and logs
retain their historical behavior. Schedule decay and failure to transfer from long to short
starts remain hypotheses without controlled ablations.

Board-size curriculum (2^4 -> 3^4 -> 4^4) is done by separate runs: the observation size differs,
so weights are not transferred.

## Imitation warm start (`snake4d imitate`)

exp04's selected checkpoint recorded no true-start completions on 4^4; the contaminated gates and
single training run prevent a general conclusion about reverse curricula. The
`imitate` phase behaviour-clones (Pomerleau 1988,
https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html;
covariate-shift caveat and the DAgger remedy: Ross, Gordon and Bagnell 2011,
https://arxiv.org/abs/1011.0686) the same MaskablePPO policy on route-follower decisions collected
from the batched env with starts spread over every snake length (negative log-likelihood of the
expert action under the masked policy, Adam, `bc_epochs` passes over `n_envs * n_steps` samples),
saves `bc_model.zip`, and `train --set model_path=...` fine-tunes it with PPO. Logged BC accuracy
is training-minibatch accuracy, not held-out accuracy. The recorded clone completed all observed
deterministic evaluation episodes; successful generalization is limited to that evaluation
condition. The fine-tuning run did not establish an efficiency improvement, and its limited
trajectory coverage prevents identifying a cause ([audit](../reports/scientific_audit.md)).

## Evaluation

See docs/evaluation.md.
