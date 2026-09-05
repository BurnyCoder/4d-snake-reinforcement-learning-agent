# Experiment 03 - MaskablePPO on the 3^4 board (81 cells, odd parity)

Date: 2026-09-04. Arms (30M steps each, 2048 batched envs, `n_steps=64`, batch 8192, ~10 min each on CUDA):

| arm | curriculum | frontier reached | wall |
|---|---|---|---|
| exp03a `exp03a_ppo_3x4_nocur` | off | - | 9.9 min |
| exp03b `exp03b_ppo_3x4_backplay` | Backplay, strict gate `rho=0.9`, window 4, step 1 | 80 -> 78 | 9.7 min |
| exp03c `exp03c_ppo_3x4_backplay_relaxed` | Backplay, `rho=0.8`, window 8, step 4 | 80 -> 76 | 10.6 min |

The route on this board is the 80-cell cycle minus the corner (docs/game_rules.md), so the
frontier starts at 79 effective (the callback logged 80 until the read-back fix in the review PR).

## Question

Does PPO complete an odd board, where the last cell must be reached through the corner detour and
the final body must be a Hamiltonian path with both ends in the majority colour class? Does the
reverse curriculum that worked on 2^4 carry over?

## Hypotheses

- H1: without a curriculum, fill plateaus well below 1 with no completions (parity dead-ends).
- H2: the strict-gate Backplay curriculum reaches the true start and yields completions.

## Results (best checkpoint, 100 episodes x 3 seeds, deterministic)

| arm | success | fill (eval) | true-start fill during training (mean / best episode) | curriculum-start success |
|---|---|---|---|---|
| exp03a plain | 0.000 | 0.566 | 0.5395 / 0.914 (74/81) | - |
| exp03b strict gate | 0.000 | 0.549 | 0.5289 / 0.877 (71/81) | 0.78 mean; cleared the 0.9 gate twice (0.90 -> 79 at 14.9M, 0.91 -> 78 at 15.1M), peak 0.907 |
| exp03c relaxed gate | 0.000 | 0.542 | 0.5280 / 0.877 (71/81) | 0.60 |
| route follower (exp01) | 1.000 | 1.000 | - | - |

Figures: `reports/figures/exp03*_curves.png`, `exp03*_fill_hist.png`.

## Analysis and learnings

- H1 confirmed: every arm fills 54-57 % of the board on average and at most 74/81 cells
  (exp03a, 161,535 true-start episodes) or 71/81 (exp03b/c, ~80k each; `episodes.json`); no
  completion. Episodes end by collision at fill ~0.5-0.6, long
  before the starvation cap.
- H2 rejected: the curriculum never left the endgame. Starting as a 71-79-cell cycle segment
  (the frontier is clamped to `route_len - 1 = 79`; the callback logged 80 before the read-back
  fix; window 4 for exp03b, 8 for exp03c),
  the agent succeeded in 60-85 % of episodes on average and cleared the 90 % gate only twice
  (frontier 80 -> 78, then a plateau around 0.83); relaxing the gate to 80 % with larger steps
  moved the frontier only to 76 and lowered the curriculum success to 60 %.
  Completing from a near-full odd board needs the corner detour (enter the corner only when the
  food is there and the skipped arc is free) - a two-step plan the policy did not learn reliably in
  30M steps.
- The plain and curriculum arms end with the same true-start fill (0.53-0.55): the curriculum
  episodes (80 % of the batch) did not transfer to the true start at all, so the 20 % true-start
  episodes drove the same learning as in exp03a with a fifth of the data.
- The odd board is therefore not just "harder": it changes the endgame from "follow the free arc"
  to "follow the free arc and time a detour", and PPO with this observation does not get there.

## Decisions

- Treat 3^4 as the parity stress test, not a stepping stone: 4^4 (even, exp04) is the target.
- Next steps if the odd board is pursued: warm-start the policy by imitating the route follower
  (behaviour cloning on states from route rollouts), then fine-tune with PPO; or add an explicit
  observation feature for the corner detour condition.
