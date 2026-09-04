# Experiment 04 - MaskablePPO with Backplay on the 4^4 board (256 cells): the headline run

Date: 2026-09-04. `experiments/exp04_ppo_4x4.env`: 4096 batched envs, `n_steps=64`, batch 8192,
100M steps in 54 minutes on CUDA (31-35k fps); Backplay along the 256-cell Hamiltonian cycle with
`rho=0.8`, window 16, frontier step 8, 500 episodes per gate decision (relaxed from the 2^4
defaults after exp03b stalled at a 90 % gate).

## Question

Can model-free MaskablePPO learn to fill the 4D board from length 1 within a 100M-step budget when
episodes are seeded along the Hamiltonian cycle and the start moves backwards as the agent masters
the endgame?

## Hypothesis

The frontier moves from 255 to 1 within the budget (64 advances of 8 cells at 90 % success would
need at least ~17M steps), and the true-start completion rate becomes positive once the frontier
passes roughly half the board.

## Results

| metric | value |
|---|---|
| curriculum frontier | 255 -> 247 (9.1M steps) -> ... -> 199 (28M steps), then no advance for 72M steps |
| curriculum-start success at frontier 199 | 0.62 at 50M steps, **0.00 at 75M-100M steps** |
| rollout fill (all starts) | 0.81 at 26M, falling to 0.71 at 100M |
| true-start fill (training) | mean 0.377, best episode 0.660 (169/256 cells) |
| true-start completion (eval, 47 evaluations of 100 episodes) | 0.000 throughout |
| best checkpoint, 100 episodes x 3 seeds, deterministic / stochastic | success 0.000 / 0.000, fill 0.401 / 0.394 |
| route follower (exp01) | success 1.000, 16,448 steps |

Figures: `reports/figures/exp04_ppo_4x4_curves.png`, `exp04_ppo_4x4_fill_hist.png`.

## Analysis and learnings

- **Negative result.** The agent did not complete the 4^4 board; the best true-start episode
  filled two thirds of it. The board is completable (the route follower proves it), so this is a
  learning failure, not a feasibility one.
- **The curriculum collapsed instead of advancing.** Eight advances happened in the first 28M steps
  (success 0.80-0.84 at each gate), then the frontier stayed at 199 and the success rate of the
  57-cell-free endgame *fell to zero* in the last quarter of training. The learning rate decays
  linearly to 1e-5 over the whole 100M budget regardless of curriculum progress, so by the time
  the frontier reached the hardest states the policy could no longer adapt; the entropy bonus and
  the clip range decay the same way. Tying the schedules to the total budget was a design error
  for a curriculum whose pace is unknown in advance.
- **Distribution shift is visible.** Fill on curriculum starts (0.9+) never transferred to the
  true start (0.38-0.40): a snake laid along the cycle is in a state the true-start policy never
  produces, and the true-start episodes (20 % of resets) learned no faster than plain PPO did on
  the smaller boards.
- **What the agent did learn**: from length 1 it eats about 100 cells (fill 0.40) before
  colliding, four times the random floor (0.08) in ~200 steps, i.e. it learned greedy, mostly safe
  foraging, not the long-horizon coverage strategy.

## Decisions and next steps

1. Decouple the schedules from the curriculum: constant learning rate, clip range and entropy
   until the frontier reaches 1, then decay; or make the decay a function of the frontier.
2. Warm-start by imitation: behaviour-clone the policy on (observation, action) pairs from the
   route follower (which is available in unlimited quantity from the batched env), then fine-tune
   with MaskablePPO - the network then starts from a policy that already completes the board and
   RL only has to make it faster. This is the most promising route to a learned completer.
3. Smaller frontier steps (4) with a longer budget (300M+ steps, ~3 h), and evaluation of the
   curriculum-start success on a fixed set of frontier states instead of the moving window.
4. As a fallback, search at decision time (AlphaSnake-style MCTS with the learned value function).
