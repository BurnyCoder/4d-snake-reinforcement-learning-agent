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
| `length_mean` | `summary.json` | mean episode length in steps over all episodes, successes and failures |
| `steps_to_complete_mean` | `summary.json` | mean episode length over successful episodes (efficiency vs the route follower) |
| `win_within` | `summary.json` | fraction of episodes won within `C`, `2C`, `4C`, `C*C/2` steps (a step-capped win rate as in AlphaSnake, https://arxiv.org/abs/2211.09622) |

## Derived efficiency metrics

`reports/networks.md` compares every trained network with the scripted baselines through three
ratios computed from the fields above. They are the authors' own arithmetic (an estimate of route
efficiency, not a measured quantity); `tests/test_docs_numbers.py` recomputes them from
`summary.json`.

- Steps per food = `length_mean / (fill_mean * C - 1)`: mean episode length divided by the mean
  number of foods eaten (final length minus the starting length of 1), i.e. how far the snake walks
  between meals.
- Geometric floor = `ndim * (n^2 - 1) / (3 * n)` for a board with `size = n`: the mean Manhattan
  distance between two uniformly random cells, 2.00 / 3.56 / 5.00 for n = 2, 3, 4 (the mean absolute
  difference of two independent uniform draws from `{0, ..., n - 1}` is `(n^2 - 1) / (3 * n)` per
  axis). Food spawns on a uniformly random free cell (docs/game_rules.md), so this is roughly the
  fewest steps per food any policy could average; "times floor" is steps per food divided by it. A
  yardstick rather than a bound: the head is not uniformly placed, the free cells are not the whole
  board, and the body may block the shortest path.
- Non-eating move share = `1 - (fill_mean * C - 1) / length_mean`: the fraction of moves that did
  not end on food.

For policies that never complete the board these ratios describe only the part of the game they
survive (a policy that dies at fill 0.40 is measured on the moves before it died), so read them
together with `fill_mean`. "Won within `4C` moves" is the `4C` entry of `win_within` above.

## Procedure (`snake4d evaluate`)

- `sb3_contrib.common.maskable.evaluation.evaluate_policy` (masks applied) on the batched env
  with `n_envs = eval_episodes = 100`, so each env plays exactly one episode and the policy is
  queried once per step for the whole batch.
- Seeds `SNAKE_EVAL_SEEDS = 0,1,2` (default): the env is seeded at construction and
  `set_random_seed(seed)` is called on the policy; results are reported per seed and as mean +-
  std across seeds.
- Trained models are evaluated twice: `deterministic=True` (argmax) and `deterministic=False`
  (sampling), because a deterministic policy can cycle forever in a deterministic environment,
  and under function approximation the best policy may itself be stochastic (Sutton and Barto
  2018, ch. 13, http://incompleteideas.net/book/the-book-2nd.html); the step caps end such
  episodes as failures.
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
