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
  distance between two uniformly random cells, 2.00 / 3.56 / 5.00 for n = 2, 3, 4. Per axis, the
  mean absolute difference of two independent uniform draws from `{0, ..., n - 1}` is
  `sum_{d=1}^{n-1} 2 d (n - d) / n^2 = (n^2 - 1) / (3 * n)`; `tests/test_docs_numbers.py` checks
  the closed form by brute force over all cell pairs. "Times floor" is steps per food divided by
  this reference distance. Despite the historical label, this is not a policy-specific lower
  bound: food spawns on free cells (docs/game_rules.md, `physics._spawn_food`), the head is not
  uniformly placed, the free cells are not the whole board, and the body may block shortest paths.
- Non-eating move share = `1 - (fill_mean * C - 1) / length_mean`: the fraction of moves that did
  not end on food.

For policies with no observed completions these ratios describe only the part of the game they
survive (a policy that dies at fill 0.40 is measured on the moves before it died), so read them
together with `fill_mean`. "Won within `4C` moves" is the `4C` entry of `win_within` above.

## Procedure (`snake4d evaluate`)

- `sb3_contrib.common.maskable.evaluation.evaluate_policy` (masks applied) on the batched env
  with `n_envs = eval_episodes = 100`, so exactly one episode per env is counted and the policy
  is queried once per step for the whole batch. Finished envs continue stepping and autoresetting
  while other envs finish; their additional episodes are not included in the reported sample.
- Seeds `SNAKE_EVAL_SEEDS = 0,1,2` (default): the env is seeded at construction and
  `set_random_seed(seed)` is called on the policy; results are reported per seed and as mean +-
  population std (`numpy.std`, `ddof=0`) across evaluation seeds for the same checkpoint. All
  ten archived networks use training seed 0; these three evaluations are not three independently
  trained replications.
- Trained models are evaluated twice: `deterministic=True` (argmax) and `deterministic=False`
  (sampling), because a deterministic policy can cycle forever in a deterministic environment,
  and under function approximation the best policy may itself be stochastic (Sutton and Barto
  2018, ch. 13, http://incompleteideas.net/book/the-book-2nd.html); the step caps end such
  episodes as failures.
- Scripted policies (`route`, `random`) run through the same function, so baseline and agent
  numbers are directly comparable.

Outputs: `runs/<ts>_evaluate_<name>/summary.json` (per run and per mode) and
`eval_episodes.csv` (one row per episode: seed, deterministic, success, fill, length, return).

## Interpretation and uncertainty

The archived scores describe the evaluated checkpoints under the recorded board, caps, masking,
and action-selection mode. The reported std is descriptive evaluation-seed variation, not a
confidence interval or uncertainty across training runs. Algorithm-level superiority, causal
effects of individual curriculum parameters, and robustness to training seeds remain unresolved;
see [Patterson et al. (2024)](https://jmlr.org/papers/v25/23-0183.html) for the experimental-design
distinction and the [scientific audit](../reports/scientific_audit.md) for this project's evidence.

For a fixed checkpoint, a Wilson interval computed from pooled episode successes is a conditional
binomial interval if those episodes are treated as independent trials with a common success
probability ([NIST formula](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)). It
does not account for training variation, checkpoint selection, model development using earlier
results, or dependence between episodes; shared numeric seeds also do not guarantee matched food
trajectories across policies. The audit reports these intervals as a post-hoc supplement without
changing archived `summary.json` fields. Zero observed successes and all observed successes do
not establish population probabilities of zero and one, respectively.

## During training

`MaskableEvalCallback` evaluates `eval_episodes` true-start episodes every `eval_every` steps
(default 2,097,152 = 8 rollouts), records `eval/success_rate`, `eval/mean_reward`,
`eval/mean_ep_length`, saves `best_model.zip` on improvement in **mean evaluation return**, and
appends to `eval/evaluations.npz`
(`timesteps`, `results`, `ep_lengths`, `successes`). Curriculum episodes never enter the
evaluation env. The selection criterion is not completion rate: the callback compares
`mean_reward > best_mean_reward`
([sb3-contrib source](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/callbacks.py)).
The discounted training objective and the undiscounted return used for selection both
differ from the headline completion metric; see [the reward design](rl_design.md#reward).

An evaluation of a saved checkpoint is distinct from a training replicate. Existing results are
exploratory: neither chronological preregistration nor exact historical source reconstruction is
established by the saved configuration and HEAD alone. The required provenance additions for
future runs are specified in [architecture.md](architecture.md#future-run-provenance-requirements).
