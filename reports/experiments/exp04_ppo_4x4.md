# Experiment 04 - MaskablePPO with Backplay on the 4^4 board (256 cells): the headline run

Date: 2026-09-04. `experiments/exp04_ppo_4x4.env`: 4096 batched envs, `n_steps=64`, batch 8192,
100M steps in 54 minutes on CUDA (31.1k steps/s, the final cumulative `time/fps`); Backplay along the 256-cell Hamiltonian cycle with
`rho=0.8`, window 16, frontier step 8, 500 episodes per gate decision (relaxed from the 2^4
defaults after exp03b stalled at a 90 % gate).

## Question

Can model-free MaskablePPO learn to fill the 4D board from length 1 within a 100M-step budget when
episodes are seeded along the Hamiltonian cycle and the start moves backwards as the agent masters
the endgame?

## Hypothesis

The frontier moves from 255 to 1 within the budget (32 advances of 8 cells at the 80 % gate,
each needing at least 500 curriculum episodes), and the true-start completion rate becomes
positive once the frontier passes roughly half the board.

## Results

| metric | value |
|---|---|
| curriculum frontier | 255 -> 247 (18.6M steps, 9.1 min) -> 239 (22.3M) -> 231 (22.5M) -> 223 (40.6M) -> 215 (46.4M) -> 207 (46.9M) -> 199 (53.7M, 27.1 min); no advance in the remaining 46.4M steps (`run.log`, `progress.csv`) |
| curriculum-start success at frontier 199 | 0.81 at the advance (53.7M), 0.60 at 56.1M, **0.00 from ~73M to 100M steps** |
| rollout fill (all starts) | 0.81 at 26M, falling to 0.71 at 100M |
| true-start fill (training) | mean 0.377, best episode 0.660 (169/256 cells); 81,995 true-start episodes of 624 steps on average (`episodes.json`) |
| true-start completion (eval, 47 evaluations of 100 episodes) | 0.000 throughout |
| best checkpoint, 100 episodes x 3 seeds, deterministic / stochastic | success 0.000 / 0.000, fill 0.401 / 0.395 |
| route follower (exp01) | success 1.000, 16,448 steps |

Figures: `reports/figures/exp04_ppo_4x4_curves.png`, `exp04_ppo_4x4_fill_hist.png`.

## Analysis and learnings

- **Negative result.** The agent did not complete the 4^4 board; the best true-start episode
  filled two thirds of it. The board is completable (the route follower proves it), so this is a
  learning failure, not a feasibility one.
- **The curriculum collapsed instead of advancing.** Seven advances happened between 18.6M and
  53.7M steps (gate success 0.80-0.84 each time), then the frontier stayed at 199 and the success
  rate of the 57-cell-free endgame *fell to zero* in the last quarter of training. The learning rate decays
  linearly to 1e-5 over the whole 100M budget regardless of curriculum progress, so by the time
  the frontier reached the hardest states the policy could no longer adapt; the clip range decays
  the same way (the entropy coefficient is a constant 0.01, `train.py`). Tying the schedules to the
  total budget was a design error for a curriculum whose pace is unknown in advance.
- **Distribution shift is visible.** Fill on curriculum starts (0.88, `episodes.json`) never transferred to the
  true start (0.38-0.40): a snake laid along the cycle is in a state the true-start policy never
  produces, and the true-start episodes (20 % of resets) learned no faster than plain PPO did on
  the smaller boards.
- **What the agent did learn**: from length 1 it eats about 100 cells (eval fill 0.40) before
  colliding - 5.2x the random floor (0.077) - in episodes of roughly 570-630 steps (evaluation
  566, training 624), i.e. it learned greedy, mostly safe foraging, not the long-horizon coverage
  strategy.

## Decisions and next steps

1. Decouple the schedules from the curriculum: constant learning rate and clip range until the
   frontier reaches 1, then decay; or make the decay a function of the frontier.
2. Warm-start by imitation: behaviour-clone the policy on (observation, action) pairs from the
   route follower (which is available in unlimited quantity from the batched env), then fine-tune
   with MaskablePPO - the network then starts from a policy that already completes the board and
   RL only has to make it faster. This is the most promising route to a learned completer.
3. Smaller frontier steps (4) with a longer budget (300M+ steps, ~3 h), and evaluation of the
   curriculum-start success on a fixed set of frontier states instead of the moving window.
4. As a fallback, search at decision time (AlphaSnake-style MCTS with the learned value function).
