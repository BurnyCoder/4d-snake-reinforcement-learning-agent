# Experiment 05 - Imitation warm start on 4^4: behaviour cloning, then PPO fine-tuning

Date: 2026-09-04. Arms: `experiments/exp05_bc_4x4.env` (`snake4d imitate`) and
`experiments/exp05b_ppo_4x4_from_bc.env` (`snake4d train --set model_path=<exp05>/bc_model.zip`).

## Question

exp04 showed that model-free PPO with a reverse curriculum does not complete the 256-cell board in
100M steps, while the scripted route follower completes it every time. Can the *same* policy
network (MLP [512, 512] on the `4C + 2` observation) learn to complete the board by imitating the
route follower, and does PPO fine-tuning then keep the completion rate while reducing the ~16k
steps per game?

## Hypotheses

- H1: the network can represent the route follower on 4^4 (the action is a function of the head
  cell only, a 256-entry lookup) and behaviour cloning reaches near-100 % expert-action accuracy.
- H2: the cloned policy completes the board from the true start in 100 % of deterministic
  evaluations; sampled actions occasionally leave the cycle and lower the rate.
- H3: PPO fine-tuning with gentle settings (lr 1e-4 -> 1e-5, clip 0.1 -> 0.05, no entropy bonus,
  no curriculum) keeps completion near 100 % and shortens games by taking shortcuts.

## Setup

- Data: 4096 batched envs x 64 steps = 262,144 (observation, expert action, mask) samples, starts
  spread uniformly over every snake length via curriculum-style resets along the cycle
  (`p_true_start = 0.2`), expert = `RoutePolicy` (Hamiltonian cycle follower).
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

Expert-action accuracy after cloning: 1.000 (loss 2e-4 after 20 epochs).

### PPO fine-tuning (exp05b): 20M steps, 22.5 min, from `bc_model.zip`

| metric | value |
|---|---|
| in-training evaluations (100 true-start episodes every 2.1M steps) | success 1.00 at every one of the 9 evaluations |
| mean episode length at those evaluations | 16,218 - 16,514 steps (unchanged; `evaluations.npz`) |
| `train/approx_kl` | 7.9e-7 to 1.7e-5 at every update (prints as 0.000; `progress.csv`) |
| throughput | 15k fps (half of exp04: every rollout ends no episode, and each evaluation is 100 x 16k steps) |

H3 is **rejected in an instructive way**: the completion rate stayed at 100 %, but the policy did
not change at all. Behaviour cloning drove the expert action's probability to about 0.9998 (exp(-2e-4) from the
final cloning loss; not measured directly), so every
sampled action equals the argmax, the advantage estimates carry no information about alternative
moves, and with `ent_coef = 0` there is no force widening the distribution: PPO's update is
negligible (approx_kl below 2e-5). Fine-tuning a near-deterministic clone needs exploration on
purpose - an entropy bonus or a temperature on the cloned logits, or a less over-fitted clone
(early stopping, label smoothing: Müller, Kornblith and Hinton 2019,
https://arxiv.org/abs/1906.02629; or DAgger-style data aggregation: Ross, Gordon and Bagnell
2011, https://arxiv.org/abs/1011.0686) - which is the next experiment.

## Analysis and learnings

- H1 and H2 confirmed: a neural network policy now completes the 4D board from the true start in
  every deterministic evaluation episode - the first learned agent in this project to do so on
  256 cells. It is the route follower distilled into weights: identical step counts (16,448 on
  average, seed by seed), so it has learned the cycle as a head-position -> action lookup rather
  than any shortcut behaviour.
- Sampling actions (stochastic evaluation) completes 74-84 % of episodes: with 8 actions and a
  distribution that puts most but not all mass on the cycle move, a few thousand samples per game
  are enough for an off-cycle move that eventually traps the snake. Deterministic play is the
  right mode for this policy; fine-tuning should sharpen the distribution.
- Why imitation succeeds where the curriculum failed: the cloned policy is trained on states from
  *all* fill levels at once (uniform-length starts) with a dense supervised signal, so there is no
  frontier to advance and no distribution shift between "curriculum starts" and "true starts" -
  the true-start trajectory of the cycle follower is itself made of those states.
- Cost: about twelve seconds of cloning versus 54 minutes of PPO in exp04; the expert is free because the
  route follower runs inside the batched env.

## Decisions

- The imitation warm start is the recommended path for larger boards: it is the only learned
  4^4 completer so far.
- PPO's remaining job is efficiency (fewer steps than the cycle) and robustness under sampling.
  exp05b shows a zero-entropy clone cannot be fine-tuned as is; the next arm fine-tunes with
  `ent_coef > 0` (or clones with label smoothing) so that PPO has alternatives to evaluate, with
  the completion rate as a hard constraint (keep `best_model.zip` by evaluation success).
