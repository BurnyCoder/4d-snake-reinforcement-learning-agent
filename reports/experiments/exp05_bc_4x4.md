# Experiment 05 - Imitation warm start on 4^4: behaviour cloning, then PPO fine-tuning

Date: 2026-09-04. Arms: `experiments/exp05_bc_4x4.env` (`snake4d imitate`) and
`experiments/exp05b_ppo_4x4_from_bc.env` (`snake4d train --set model_path=<exp05>/bc_model.zip`).

## Question

exp04 recorded no true-start completions in its 100M-step run, while the scripted route follower
completed every sampled episode. Can the *same* policy
network (MLP [512, 512] on the `4C + 2` observation) learn to complete the board by imitating the
route follower, and does PPO fine-tuning then keep the completion rate while reducing the ~16k
steps per game?

## Hypotheses

These are retrospective hypotheses; independent preregistration is not established. Both arms
use training seed 0. The three evaluation seeds measure each fixed checkpoint, not variation
across independent training runs ([scientific audit](../scientific_audit.md)).

- H1: the network can represent the route follower on 4^4 (the action is a function of the head
  cell only, a 256-entry lookup) and behaviour cloning reaches near-100 % expert-action accuracy.
- H2: the cloned policy completes the board from the true start in 100 % of deterministic
  evaluations; sampled actions occasionally leave the cycle and lower the rate.
- H3: PPO fine-tuning with gentle settings (lr 1e-4 -> 1e-5, clip 0.1 -> 0.05, no entropy bonus,
  no curriculum) keeps completion near 100 % and shortens games by taking shortcuts.

## Setup

- Data: 4096 batched envs x 64 steps = 262,144 (observation, expert action, mask) samples. The
  reset mixture uses length-1 starts with probability 0.2 and curriculum starts sampled across
  cycle lengths; subsequent expert transitions are not uniformly distributed over lengths.
  Expert = `RoutePolicy` (Hamiltonian cycle follower); there is no held-out action dataset.
- Cloning: negative log-likelihood of the expert action under the masked policy distribution,
  Adam lr 1e-3, minibatch 8192, 20 epochs; the value head is untouched. Wall time about 12 s for the model build plus the 20 epochs, about
  17 s including expert-data collection (`run.log`).
- Evaluation: docs/evaluation.md protocol, 100 episodes x 3 seeds, deterministic and stochastic.

## Results

| policy | deterministic success | stochastic success | fill (det / stoch) | steps to complete (det) |
|---|---|---|---|---|
| behaviour-cloned network (exp05) | **1.000 +- 0.000** | 0.787 +- 0.041 | 1.000 / 0.901 | 16,448 |
| clone + PPO fine-tuning, best checkpoint (exp05b) | **1.000 +- 0.000** | 0.763 +- 0.017 | 1.000 / 0.884 | 16,414 |
| route follower (exp01, scripted) | 1.000 | - | 1.000 | 16,448 |
| PPO + Backplay from scratch (exp04, 100M steps) | 0.000 | 0.000 | 0.401 / 0.395 | - |

Training-minibatch expert-action accuracy after cloning: 1.000 (loss 2e-4 after 20 epochs).
This logged accuracy aggregates minibatch accuracies during the final training epoch, each
measured immediately after its update. It is neither held-out generalization accuracy nor a
fixed-final-model dataset evaluation ([imitation.py](../../src/snake4d/imitation.py)).

### PPO fine-tuning (exp05b): 20M steps, 22.5 min, from `bc_model.zip`

| metric | value |
|---|---|
| in-training evaluations (100 true-start episodes every 2.1M steps) | success 1.00 at every one of the 9 evaluations |
| mean episode length at those evaluations | 16,218 - 16,514 steps (`evaluations.npz`) |
| `train/approx_kl` | 7.9e-7 to 1.7e-5 at every update (prints as 0.000; `progress.csv`) |
| throughput | 15k fps; no profiling ablation isolates the difference from exp04 |

H3's efficiency component is **unestablished**. Both selected checkpoints completed all
300 deterministic evaluation episodes, but the mean step-count difference alone does not
establish an improvement. Nonzero KL and entropy contradict the earlier claims that the policy
did not change and every sampled action equalled argmax. Stochastic evaluations recorded
**64/300 failures before** and **71/300 after** fine-tuning; absence of an entropy bonus is not
zero policy entropy. Exponentiating a mean training loss would give a geometric mean of assigned
expert probabilities on those training examples, not a universal action probability at test time.

Training coverage is a substantial confound: 4096 environments each received **4,928 steps**,
whereas a successful deterministic evaluation game took roughly 16,414 steps. Across training,
only **one completed episode, a failure**, was recorded and there were **zero successful training
episodes** ([episodes.json](../data/exp05b_ppo_4x4_from_bc/episodes.json)). PPO still updates on
partial trajectories, but this run supplies little evidence about fine-tuning with complete
successful trajectories. The [audit](../scientific_audit.md) links the saved-log derivation;
it does not establish insufficient exploration as the cause.

## Analysis and learnings

- H1 and H2 are consistent with the observed training accuracy and deterministic evaluations.
  The cloned checkpoint completed every sampled deterministic episode and matched the route
  follower's aggregate step counts (16,448 on average, seed by seed). This is consistent with
  route imitation, but aggregate equality does not prove identical trajectories or a head-only
  internal representation; that would require interventions on non-head observation features.
- Sampling actions completes 74-84 % across the clone's three seed groups. This establishes a
  mode-dependent outcome for this checkpoint. Saved aggregates alone do not identify which
  off-cycle actions led to each failure or guarantee that sharpening would improve the policy.
- Dense expert supervision and varied starting lengths are plausible reasons for BC's outcome.
  The comparison with exp04 also changes objective, initialization, data and training exposure,
  and exp04 has a curriculum defect. It does not isolate supervision or establish absence of
  a train/evaluation state-distribution shift.
- Cost: about twelve seconds for cloning versus 54 minutes for exp04. Expert collection takes
  additional time as recorded above, and the hand-written route is prior problem knowledge;
  this is not an equal-information or equal-compute algorithm comparison.

## Decisions

- The cloned checkpoints are demonstrated 4^4 completers on the sampled deterministic games;
  generalization to larger boards and robustness across training seeds remain open.
- A future fine-tuning comparison should first provide sufficient per-environment trajectory
  coverage, then vary exploration separately while preserving a completion metric and a
  prespecified analysis. Entropy bonuses or label smoothing are candidate interventions, not
  remedies demonstrated by exp05b. No new model execution was performed for these corrections.
