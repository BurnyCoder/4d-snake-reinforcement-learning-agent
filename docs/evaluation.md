# Evaluation protocol

## Definition of "complete"

An episode is a success when the snake length reaches `C = size**ndim` within the episode caps
(`idle_mult * C` steps without eating, `C*C` steps in total). `info["is_success"]` and
`info["fill"] = length / C` are emitted on every step, so SB3's `rollout/success_rate`,
`eval/success_rate` and the VecMonitor CSV all see them.

## Headline metrics

| metric | where | meaning |
|---|---|---|
| `success_rate` | `summary.json`, `eval/success_rate` | fraction of episodes that fill the board |
| `fill_mean` | `summary.json`, `rollout/fill_mean` | mean final `length / C` |
| `steps_to_complete_mean` | `summary.json` | mean episode length over successful episodes (efficiency vs the route follower) |
| `win_within` | `summary.json` | fraction of episodes won within `C`, `2C`, `4C`, `C*C/2` steps (a step-capped win rate as in AlphaSnake) |

## Procedure (`snake4d evaluate`)

- `sb3_contrib.common.maskable.evaluation.evaluate_policy` (masks applied) on the batched env
  with `n_envs = eval_episodes = 100`, so each env plays exactly one episode and the policy is
  queried once per step for the whole batch.
- Seeds `SNAKE_EVAL_SEEDS = 0,1,2` (default): the env is seeded at construction and
  `set_random_seed(seed)` is called on the policy; results are reported per seed and as mean +-
  std across seeds.
- Trained models are evaluated twice: `deterministic=True` (argmax) and `deterministic=False`
  (sampling), because a deterministic policy can cycle forever in a deterministic environment
  while a stochastic one breaks such loops (Sutton and Barto 2018, ch. 13,
  http://incompleteideas.net/book/the-book-2nd.html); the step caps end such episodes as failures.
- Scripted policies (`route`, `random`) run through the same function, so baseline and agent
  numbers are directly comparable.

Outputs: `runs/<ts>_evaluate_<name>/summary.json` (per run and per mode) and
`eval_episodes.csv` (one row per episode: seed, deterministic, success, fill, length, return).

## During training

`MaskableEvalCallback` evaluates `eval_episodes` true-start episodes every `eval_every` steps
(default 2,097,152 = 8 rollouts), records `eval/success_rate`, `eval/mean_reward`,
`eval/mean_ep_length`, saves `best_model.zip` on improvement and appends to `eval/evaluations.npz`
(`timesteps`, `results`, `ep_lengths`, `successes`). Curriculum episodes never enter the
evaluation env.
