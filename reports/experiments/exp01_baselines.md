# Experiment 01 - Scripted baselines: is every board completable, and at what step cost?

Date: 2026-09-04. Command: `uv run snake4d evaluate --set size=<2|3|4> --set ndim=4 --set policy=<route|random>`
(runs `*_evaluate_exp01_*`; per-episode CSVs and summaries in `reports/data/exp01_*`).

## Question

Can a scripted policy fill the 2^4, 3^4 and 4^4 boards from a random length-1 start under the
episode caps used for training (starvation after `4 * cells` idle steps, `cells^2` total), and how
many steps does it need? What does masked random play achieve? These two numbers bracket every
learned agent: the route follower is the completability proof and the efficiency ceiling to beat;
random legal play is the floor.

## Hypotheses

- H1: following a Hamiltonian cycle completes every even board (4^4 = 256 cells, 2^4 = 16) in
  100 % of episodes, in about `cells^2 / 2` steps (the head must circle the board once per food on
  average).
- H2: the odd board 3^4 (81 cells, colour classes 41/40, no Hamiltonian cycle) is also completable
  by a scripted policy, but a plain Hamiltonian *path* follower is not enough.
- H3: masked random play completes the 16-cell board sometimes and never the larger ones.

## Setup

100 episodes x 3 seeds per policy and board, deterministic policies, the evaluation protocol of
docs/evaluation.md (masked `evaluate_policy`, one batched env per episode, explicit seeds).

## Results

| board | policy | success (mean +- std over seeds) | fill | mean steps to complete | win within `C^2/2` steps |
|---|---|---|---|---|---|
| 2^4 (16) | route | 1.000 +- 0.000 | 1.000 | 65.9 | 100 % |
| 2^4 (16) | random | 0.293 +- 0.048 | 0.816 | 129.4 | 15 % |
| 3^4 (81) | route | 1.000 +- 0.000 | 1.000 | 1,874.5 | 100 % |
| 3^4 (81) | random | 0.000 +- 0.000 | 0.226 | - | 0 % |
| 4^4 (256) | route | 1.000 +- 0.000 | 1.000 | 16,447.8 | 100 % |
| 4^4 (256) | random | 0.000 +- 0.000 | 0.077 | - | 0 % |

## Analysis and learnings

- H1 confirmed: the cycle follower never fails on even boards. The step cost is close to
  `C^2 / 4` (16,448 for C = 256), half the `C^2 / 2` worst case, because the food is on average half
  a lap ahead of the head. This is the efficiency bar: a learned agent should complete 4^4 in far
  fewer than ~16k steps.
- H2 confirmed only after a fix. The first version of the route follower used the reflected
  Gray-code *path* on odd boards and reached only 4 % fill on 3^4 (worse than random play at
  22 %): an open path has a dead end, so the head reaches the end, has to leave the route, and then
  keeps colliding with its own body near the end. The fix (`hamilton.cycle_minus_corner`) builds a
  Hamiltonian cycle over the 80 cells other than the corner - an explicit 2D construction plus a
  "ladder" splice of the corner's column into a neighbouring column for each extra dimension -
  and the policy enters the corner only when the food is there and at most as many cycle cells
  would be skipped as are currently free (otherwise the head could overtake its tail). With that,
  3^4 is completed in 100 % of episodes with zero fallbacks, in 1,874 steps on average. The stale
  Gray-path run was removed from `runs/`; this paragraph is its record.
- H3 confirmed: masked random play finishes the 16-cell board in 29 % of episodes (masking alone
  prevents most deaths on a tiny board) but never the 81- or 256-cell boards, where it fills
  23 % and 8 % before dying or starving.
- The starvation cap (`4 * cells`) is never binding for the route follower: the longest wait for
  food is one lap (`C` steps) plus the detour on odd boards.

## Decisions

- The route follower is the demonstration for the Backplay curriculum on every board size,
  including odd ones (now a cycle segment, not a path prefix).
- Report every learned agent's steps-to-complete against 65.9 / 1,874 / 16,448.
