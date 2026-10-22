# Independent scientific audit of the archived experiments

Author: Libor Burian. Audit date: 2026-10-22. Code baseline: `cd30fd7`.

## Scope, method and conclusion

This is a retrospective audit, not a new training experiment or a preregistration. It independently
checks the source, historical Git trees, CSV/JSON/NPZ results and existing local monitor logs;
previous audit conclusions are not treated as evidence. No model, training, inference or policy
rollout was run. Checkpoint ZIP metadata was read as JSON without deserializing model objects.

The numerical evaluation summaries agree with their episode records. However, the curriculum gate
accepts stale episodes, all trained configurations have one training seed (0), and several causal
and reproducibility claims exceed the available evidence. The surviving findings describe the
particular evaluated checkpoints, boards, episode caps and action-selection modes. They do not
establish algorithm-level superiority, training robustness, universal completion or impossibility.

The audit checked 16 evaluation archives: 7,800 episode records, 78 seed/mode groups, 26 pooled
groups and 780 aggregate fields. Each seed/mode group has 100 episodes. The count treats each `win_within`
dictionary as one field while checking all four entries. Success, final fill, return, episode
length, successful-episode length, step-capped wins and evaluation-seed mean/standard deviation
all agree. Every recorded success has fill 1; fills are valid cell counts and lengths respect the
configured caps. See [the standalone checks](../tests/test_scientific_audit.py) and the original
`reports/data/exp*/eval_episodes.csv` and `summary.json` files; those files are unchanged.

## 1. Curriculum gates mix episodes from different frontiers

[`Backplay._on_step`](../src/snake4d/callbacks.py) accepts every finished episode with
`start_len > 1`. Advancing clears the result list, but [`SnakeBatch.set_curriculum`](../src/snake4d/physics.py)
only changes future resets. Episodes already in flight retain their easier starting states and
can enter the next gate. There is no reset-time frontier identifier in the episode records.

Saved exp04 records demonstrate the problem, independently of any proposed explanation for the
later collapse:

| Frontier change | Transition count | Mixed gate wins/episodes | Definitely stale starts/wins | Remaining wins/episodes |
|---|---:|---:|---:|---:|
| 239 -> 231 | 22,544,384 | 740/892 = 82.96% | 355 / 346 | 394/537 = 73.37% |
| 215 -> 207 | 46,923,776 | 799/963 = 82.97% | 471 / 444 | 355/492 = 72.15% |

"Definitely stale" means `start_len` exceeds the frontier being tested. Both remaining rates
are below the configured 80% gate; the second has fewer than the required 500 episodes. Remaining
episodes can still be stale when old and new length windows overlap. Therefore these exclusions
do not recover an uncontaminated current-frontier estimate.

The [1,855 archived curriculum rows](data/scientific_audit/gate_episodes.csv) preserve their original
monitor line numbers. The [evidence manifest](data/scientific_audit/evidence.json) records the
complete source monitor's SHA-256, timestamp origin, extraction windows and gate counts.
The mixed rates match [exp04 progress](data/exp04_ppo_4x4/progress.csv) exactly.
Windows are between consecutive frontier changes in [run.log](data/exp04_ppo_4x4/run.log):
21:13:49.303 to 21:13:57.664 and 21:25:55.578 to 21:26:12.089 on 2026-09-04, UTC+02:00.
Subtracting the monitor's Unix `t_start` gives the manifest's lower-exclusive, upper-inclusive
elapsed-time windows; only `start_len > 1` rows enter the gate.

This defect invalidates the interpretation that advancement demonstrated mastery at each stage.
It does not change the independently recorded true-start evaluation scores. Schedule decay and
start-distribution mismatch remain hypotheses; neither was isolated by an ablation, and this
implementation defect is an additional possible contributor. No causal share is estimated here.
Future implementation requirements are specified in [the RL design](../docs/rl_design.md).

## 2. Uncertainty and the unit of replication

All ten training/imitation `config.json` files specify `seed = 0`. Final evaluation reseeds the
same loaded checkpoint with 0, 1 and 2; `success_rate_std` is the population standard deviation
of three evaluation-batch rates, not a confidence interval or variability across trained networks.
The strict versus loose curriculum also changes threshold, window and minimum episode count
together. Their scores compare two configurations and two particular checkpoints, not an isolated
threshold intervention. See [exp02](experiments/exp02_ppo_2x4.md).

Illustrative two-sided 95% Wilson intervals for a fixed checkpoint are:

| Observed deterministic wins | Example | Conditional completion interval |
|---|---|---:|
| 289/300 | exp02d | 93.55-97.94% |
| 300/300 | exp05 and exp05b, separately | 98.74-100.00% |
| 0/300 | each exp03/exp04 checkpoint, separately | 0.00-1.26% |

These use the [NIST Wilson formula](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
under an independent Bernoulli-episode interpretation for the fixed checkpoint and evaluation
distribution. They do not account for training variance, adaptive experiment selection or multiple
comparisons, and are not algorithm-comparison significance tests. More episode resampling cannot
recover missing training replications. See [Patterson et al.](https://jmlr.org/papers/v25/23-0183.html),
[Agarwal et al.](https://arxiv.org/abs/2108.13264) and the [evaluation protocol](../docs/evaluation.md).

Training validation uses seed 1000, while final evaluation uses 0, 1 and 2. Shared seed 0 alone
does not establish train/test trajectory leakage: the batch sizes and random-call schedules
differ. No direct trajectory leak was demonstrated. Conversely, common evaluation seeds do not
establish matched episode pairs: food spawning and autoresets consume a shared batch generator
according to policy-dependent events, and CSV rows are in completion order without environment
IDs. Pairing rows by ordinal position would be unjustified. Reused evaluation results and adaptive
configuration choices make these comparisons exploratory.

## 3. Fine-tuning and behaviour-cloning claims

[Exp05b's episode summary](data/exp05b_ppo_4x4_from_bc/episodes.json) contains one completed
training episode, a failure, and zero successful episodes. Its 20,185,088 transitions over 4,096
environments give 4,928 steps per environment, compared with about 16,414 steps per successful
evaluation game. PPO still learns from partial trajectories and bootstrapped values, but this run
did not observe a successful complete training trajectory. Limited trajectory coverage is a major
alternative explanation for the lack of an established efficiency improvement.

The [progress log](data/exp05b_ppo_4x4_from_bc/progress.csv) has nonzero recorded KL, entropy and
policy-gradient terms. Stochastic evaluations contain 64/300 failures for
[BC](data/exp05_bc_4x4_eval/summary.json) and 71/300 for
[fine-tuning](data/exp05b_ppo_4x4_from_bc_best/summary.json). The deterministic mean successful-game
length changes from 16,447.84 to 16,413.5567 steps. These observations contradict "every sampled
action equals argmax" and do not support "the policy did not change at all". The small sample-mean
speed difference is descriptive; a reliable improvement has not been established.

[`imitation.py`](../src/snake4d/imitation.py) logs loss before each minibatch update and accuracy
after updating on that same minibatch. Its rounded 1.000 accuracy is online training accuracy,
not held-out accuracy of one fixed final checkpoint. Exponentiating the averaged NLL produces a
geometric mean over training minibatches/parameter states, not a measured deployment-wide action
probability. Matching the route follower's 300 deterministic episode records supports behavioural
agreement on that evaluation; it does not identify an internal head-position-only mechanism or
prove the absence of distribution shift. The archived 300 deterministic BC wins remain valid
observations. See [the imitation report](experiments/exp05_bc_4x4.md).

## 4. Reset probability is not transition share

The 20% true-start reset probability does not imply a fifth of the training transitions. Longer
true-start episodes contribute more steps. Exact sums from the local completed-episode monitors:

| Run | True-start transitions | All completed-episode transitions | True-start share |
|---|---:|---:|---:|
| exp03b | 16,112,937 | 29,880,185 | 53.93% |
| exp03c | 15,582,883 | 29,883,203 | 52.15% |
| exp04 | 51,186,306 | 99,301,360 | 51.55% |

Counts and source hashes are in [evidence.json](data/scientific_audit/evidence.json). They exclude
unfinished episodes at the training cutoff, so they are not exact fractions of the entire rollout
buffer. The original archived `episodes.json` files round mean lengths and cannot recover these
integer totals exactly. Those rounded summaries corroborate the approximate shares. Comparing
final fill across curriculum and true-start episodes also confounds initial fill: curriculum
episodes begin with much of the board occupied. These comparisons do not isolate transfer.

## 5. Preregistration and source provenance

Repository history does not establish preregistration of the executed configurations or analysis:

| Evidence, 2026-09-04 UTC+02:00 | Run completed | First corresponding commit |
|---|---|---|
| exp02d configuration | 20:42:29 | `67167f0`, 20:47:43 |
| exp05 configuration and imitation implementation | 22:17:46.647 | `575430d`, 22:18:16 |
| exp04 executed rho 0.8/window 16/step 8 | 21:56:36 | `d06d4a0`, 22:14:34 |

These are Git chronology checks against the runs' archived `run.log` files. Earlier exp04
configuration existed with different settings. Unseen external preregistration cannot be ruled
out, but is not established by this repository. Describe these as exploratory experiments.

The imitation train/evaluation [versions file](data/exp05_bc_4x4/versions.json) records
`e483fe513be150f7ec5439b68d694e727b6b61dd`; that Git tree lacks `src/snake4d/imitation.py`.
The fine-tuning recorded revision also lacks the loading overrides used by the actual run;
its checkpoint JSON metadata and progress logs nevertheless agree with the stated learning-rate,
clip and entropy settings. This is a source-provenance gap, not evidence of a hyperparameter
mismatch. Recording HEAD without dirty-source contents does not identify executed code.

Historical recorded PPO revisions also contain terminal-observation aliasing: `obs[i]` was stored
as a view and subsequently overwritten during autoreset. Current code uses `.copy()`. The
MaskablePPO timeout bootstrap consumes that observation. Saved rewards identify 26, 25, 23, 23,
1, 0, 1 and 0 truncations in exp02, exp02b, exp02c, exp02d, exp03a, exp03b, exp03c and exp04,
respectively. These are retrospective classifications: collision reward is
`foods - 1 - .001*(steps - 1)`, truncation reward `foods - .001*steps`; their separation is .999
and the largest observed residual is .00266. Thus the recorded-code defect had limited potential
exposure and cannot explain exp04's failure, which has zero detected training truncations.
Missing exact source snapshots prevent complete historical attribution.

## Reproduce the checks and understand their limits

```bash
uv run --no-sync pytest tests/test_scientific_audit.py -q
uv run --no-sync pytest tests/test_docs_numbers.py -q \
  -k "not test_mlp_params_matches_the_built_policy"
```

The first command imports no model code. The second selects only artifact, prose and arithmetic
checks; it excludes the model-construction test. Do not run `train`, `imitate`, `evaluate`, `bench`,
`pipeline`, or the full test suite to reproduce this audit: some existing tests run models.

Clones can recompute the evaluation summaries, intervals, fine-tuning coverage and gate counts
from committed artifacts. The three full monitor logs remain ignored; tests skip full-monitor
corroboration when they are absent. Their hashes identify the inspected originals but do not
substitute for distributing those originals. The committed gate excerpts support the local gate
calculation; independently checking extraction completeness requires the full hashed monitor.
Git chronology can be inspected with `git show` and `git log --follow` without executing code.

No new experimental outcomes are asserted. Future frontier identifiers, source manifests,
checkpoint hashes and replication requirements are specifications in
[the methodology](../docs/rl_design.md) and [architecture](../docs/architecture.md), not features
implemented by this documentation correction. Existing checkpoints and historical scores remain
unchanged; stronger causal and algorithm-level claims require new, separately authorized work.
