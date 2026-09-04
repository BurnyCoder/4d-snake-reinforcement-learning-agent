# Experiment 02 - MaskablePPO on the 2^4 board (16 cells): plain PPO vs curriculum variants

Date: 2026-09-04. Arms (each an `experiments/*.env` file, trained with `uv run snake4d train`, best
checkpoint re-evaluated with `uv run snake4d evaluate --set model_path=...`):

| arm | curriculum | budget | wall |
|---|---|---|---|
| exp02 `exp02_ppo_2x4` | off | 5M steps | 2.2 min |
| exp02b `exp02b_ppo_2x4_backplay` | Backplay, `rho=0.2`, window 8, min 200 episodes | 5M | 2.3 min |
| exp02c `exp02c_ppo_2x4_long` | off | 20M | 8.6 min |
| exp02d `exp02d_ppo_2x4_backplay_strict` | Backplay, `rho=0.9`, window 4, min 500 episodes | 5M | 2.1 min |

Common settings: 1024 batched envs, `n_steps=32`, `batch_size=4096`, 4 epochs, MLP [512, 512],
gamma 0.99, lr 3e-4 -> 1e-5, clip 0.2 -> 0.05, `ent_coef` 0.01, evaluation of 100 true-start
episodes every 327,680 steps (655,360 for exp02c), CUDA (36-39k steps/s, final `time/fps`).

## Question

Does MaskablePPO alone learn to *complete* the smallest 4D board, how fast, and does the Backplay
reverse curriculum change the outcome?

## Hypotheses

- H1: plain PPO reaches 100 % completion within a few hundred thousand steps (the board is tiny).
- H2: the curriculum is unnecessary on 16 cells (exp02b ~ exp02).

## Results (best checkpoint, 100 episodes x 3 seeds)

| arm | deterministic success | stochastic success | steps to complete | best in-training eval (at step) |
|---|---|---|---|---|
| exp02 plain 5M | 0.877 +- 0.041 | 0.853 +- 0.034 | 34.4 | 0.91 (2.9M) |
| exp02b Backplay rho 0.2 | 0.850 +- 0.029 | 0.843 +- 0.026 | 34.6 | 0.91 (3.6M) |
| exp02c plain 20M | 0.933 +- 0.005 | 0.863 +- 0.025 | 35.7 | 0.96 (17.0M) |
| **exp02d Backplay rho 0.9** | **0.963 +- 0.009** | **0.960 +- 0.033** | 43.5 | **1.00 (3.6M)** |
| route follower (exp01) | 1.000 | - | 65.9 | - |
| random legal (exp01) | 0.293 | - | 129.4 | - |

Figures: `reports/figures/exp02*_curves.png` (fill, eval success, frontier, episode length) and
`exp02*_fill_hist.png`.

## Analysis and learnings

- H1 is **rejected**. Plain PPO climbs quickly (eval success 0.41 after 0.3M steps, 0.8 after
  1.3M) but plateaus around 0.85-0.93; 4x more steps only lifts the best checkpoint from 0.88 to
  0.93. Rollout fill is 0.98, so the agent almost always reaches 15/16 cells.
- Failure mode: of 61 failed evaluation episodes of the 20M model, 35 end at fill 15/16 by a
  collision (median length 33 for those 35, 31 over all 61 failures; far below the 64-step
  starvation cap; `eval_episodes.csv`), i.e. the snake boxes itself in
  with one cell left. Completing the last cell requires the body to be a Hamiltonian path whose
  head can still reach the free cell - a global constraint the one-hot MLP policy only partly learns
  from sparse terminal rewards.
- H2 is rejected in an instructive way. With the loose gate (`rho=0.2`) the frontier ran from 15
  to 1 in about 15 seconds of training - the curriculum advanced as soon as one fifth of the
  endgame episodes succeeded, so it taught nothing (exp02b = exp02 within noise). With a strict
  gate (`rho=0.9`, narrow window, 500 episodes per decision) the agent had to master each endgame
  frontier before moving back; the frontier still reached 1 within a minute, but the resulting
  policy is the best of all arms (0.963 deterministic, 0.960 stochastic, an in-training evaluation
  of 1.00 at 3.6M steps) at the same 5M budget. The strict gate is now the default
  (`SNAKE_CURRICULUM_RHO=0.9`, `WINDOW=4`, `MIN_EPS=500`).
- Deterministic (argmax) evaluation beats sampling for every arm except exp02d, where both agree:
  the strict-curriculum policy is confident in the endgame; the others rely on argmax to avoid
  their own exploration noise.
- Steps to complete (34.4-43.5) are a third to a half below the route follower's 65.9: the
  learned agents cut
  across the board instead of circling it, which is the efficiency gain RL buys.
- The eval curve is noisy at 100 episodes (about +-3 percentage points, the binomial standard
  error at p = 0.9 and n = 100 - an estimate, not a measurement), which is why the reported
  numbers use the best checkpoint re-evaluated with 300 episodes; the in-training "1.00" of
  exp02d was one such 100-episode sample.

## Decisions

- Curriculum defaults switched to the strict gate for exp03b (3^4) and exp04 (4^4).
- Open question carried forward: closing the last few percent on tiny boards likely needs either
  longer training with the strict gate, a lower final entropy, or an endgame-aware feature
  (reachability of the free region); tested next on the harder boards first.
