# Completing 4D Snake with Masked PPO and a Hamiltonian Reverse Curriculum

Repository: https://github.com/BurnyCoder/4d-snake-rl - all numbers below come from the run
artifacts copied to `reports/data/` and the per-experiment write-ups in `reports/experiments/`.

## Abstract

We build an N-dimensional snake game (default 4D, 4^4 = 256 cells, 8 moves) whose rules are
implemented once as vectorised numpy over a body-age grid, so thousands of boards step together in
a single process (126k-235k environment steps per second, 33k PPO steps per second end to end on a
laptop GPU). We train sb3-contrib MaskablePPO agents to *complete* the game - fill every cell -
with a legal-action mask from the environment, a sparse reward (+1 food, -1 death, +10 win, -0.001
per step) and a Backplay-style reverse curriculum whose demonstration is a Hamiltonian route
(a cycle for even sizes; for odd sizes a cycle over all cells but one corner with a guarded
detour). Scripted route following completes 2^4, 3^4 and 4^4 in 100 % of episodes and defines
the step-count ceiling (66 / 1,875 / 16,448 steps). On 2^4 plain PPO plateaus at 0.88-0.93
completion; a strictly gated curriculum (advance only at 90 % success) reaches 0.963 at the same
5M-step budget while completing in 43.5 steps, a third fewer than the route follower's 65.9. On the
larger boards the result is negative within our budgets: on 3^4 (odd parity) every arm plateaus at
54-57 % fill with no completions, and on 4^4 a 100M-step curriculum run moves the start frontier
from 255 to 199 cells and then collapses, reaching 40 % fill from the true start (best episode
66 %) and zero completions. We attribute the collapse to schedules tied to the total budget rather
than to curriculum progress, and to the lack of transfer from cycle-shaped starts to true starts.
Behaviour-cloning the same network on 262k route-follower decisions spread over all fill levels
(about twelve seconds of supervised training) yields a neural policy that completes the full 4^4 board in
100 % of deterministic evaluation episodes (3 seeds x 100), at the follower's 16,448 steps; PPO
fine-tuning of this clone is the route to a faster learned completer.

## 1. Introduction

Snake is a coverage problem in disguise: to win, the body must at the end form a Hamiltonian path
of the grid, and every intermediate state must keep such a path reachable. In four dimensions the
grid graph has degree 8, the intuition of "corners" disappears, and the author of the projected
4D snake eugeneko/Snake4D reports being unable to play it beyond a certain snake length. This project asks whether a model-free policy can learn to complete the 4D
board, and builds the full pipeline needed to answer it: game, human play, scripted baselines,
batched training, evaluation protocol and reports.

## 2. Related work

- **Masked PPO for snake.** linyiLYi/snake-ai fills a 12x12 board with MaskablePPO, a body
  gradient in the observation, and decaying learning rate and clip range; we adopt the masking,
  the body-age channel and the schedules, but not its distance-based progress bonus.
- **Invalid-action masking.** Huang and Ontanon (2020) justify masking theoretically and show its
  value grows with the number of invalid actions - eight moves in 4D make it essential.
- **Reverse curricula.** Backplay (Resnick et al. 2018) starts near the end of a demonstration and
  moves the start backwards; Salimans and Chen (2018) gate the move on the success rate. Our
  demonstration is a Hamiltonian route rather than a human trajectory.
- **Hamiltonian-cycle agents.** twanvl/snake benchmarks cycle followers and perturbed cycles
  (0 % losses); the perturbed-cycle rule "any shortcut must result in the head not overtaking
  the tail" is from johnflux's Nokia-snake write-up (2015). A 2D grid graph with both sides >= 2
  is Hamiltonian when one side is even (Skiena 1990, via MathWorld), and an even cell count is
  necessary because grid graphs are bipartite (Itai, Papadimitriou and Szwarcfiter 1982, who also
  show the Hamilton-cycle problem NP-complete for general grid graphs); `hamilton.ham_cycle`
  lifts a 2D cycle to d dimensions and verifies the result, and the odd-board cycle over all
  cells but a corner is our own construction, also verified.
- **Imitation.** Behaviour cloning (Pomerleau 1988) learns the expert's action from the
  observation; its covariate-shift failure mode and the DAgger remedy are due to Ross, Gordon
  and Bagnell (2011).
- **AlphaSnake** (Du et al. 2022) is, by its authors' account, the first published learned agent
  with a >50 % win rate on a 10x10 board, using MCTS; it uses the Hamiltonian-cycle strategy as
  its scripted baseline, as we do.
- **Shaping.** Ng, Harada and Russell (1999): potential-based shaping is the only shaping that
  cannot change the optimal policy; we keep it available but off.

## 3. The game

State: a body-age grid `age[cell]` (0 empty, 1 tail, `length` head), head and food indices. A step
resolves the target cell, decrements every age unless the snake eats (so the tail vacates first
and following the tail is legal), tests occupancy, writes the head, checks `length == C` before
spawning food uniformly over free cells. Starvation (more than `4C` consecutive idle steps) and an absolute cap
(`C^2` steps) truncate the episode; the truncating step is paid `r_step` like any other and
there is no death penalty. Observation: `4C + 2` floats (body, time-to-vacate `age/length`,
head one-hot, food one-hot, `length/C`, hunger). Mask: inside the board, free after the tail
moves, not the neck; an all-false row falls back to in-bounds moves because MaskablePPO masks with
`-1e8`, not `-inf`. Rendering folds the 4D board into a z-by-w grid of x-by-y tiles.

Parity: the grid graph is bipartite, so a Hamiltonian cycle needs an even cell count. 4^4 has
one (constructed recursively and verified); 3^4 (41/40 colour split) has none, and completing it
requires the final body to be a Hamiltonian path with both ends in the larger colour class.

## 4. Method

**Agent.** MaskablePPO, MLP [512, 512] for policy and value, gamma 0.99, GAE 0.95, learning rate
3e-4 -> 1e-5 and clip range 0.2 -> 0.05 (both linear), entropy 0.01, target KL 0.03, 4 epochs,
minibatch 8192, 4096 batched environments with 64-step rollouts (262,144 samples per update).
Defaults were chosen by the throughput benchmark (exp00) and the reference agent's schedules.

**Batched environment.** One `SnakeBatch` holds all boards; a custom SB3 `VecEnv` exposes it and
`VecMonitor` records episodes. The benchmark measured 10x higher PPO throughput than
`DummyVecEnv`/`SubprocVecEnv` over single environments and showed the PPO update, not the
environment, to be the bottleneck (minibatch 2048 -> 8192 lifts throughput 19k -> 31k fps on the best row; `reports/data/exp00_benchmark/benchmark.json`).

**Curriculum.** 80 % of resets place the snake as a segment of length `U[hi - 4, hi]` along a
random rotation of the Hamiltonian route, starting at `hi = C - 1`; the frontier moves back by
`max(1, C/64)` cells whenever at least 500 curriculum episodes reached 90 % success; 20 % of resets
always start from length 1 and evaluation always starts from length 1.

**Baselines.** The route follower (Hamiltonian cycle, or cycle-minus-corner with a detour that
skips at most as many cycle cells as are free) and uniformly random legal play.

**Evaluation.** 100 episodes x 3 seeds, deterministic and stochastic, masked `evaluate_policy`;
metrics: completion rate, mean fill, steps to complete, win-within-K.

## 5. Experiments

Pre-registered arms live in `experiments/*.env`; every write-up follows question -> hypothesis ->
setup -> results -> learnings:

- exp00 - throughput benchmark: batched env on CUDA, 4096 envs; minibatch 8192 gives 31k fps
  (33k at 16384; 8192 adopted for more gradient steps per rollout).
- exp01 - scripted baselines: route follower 100 % on 2^4 / 3^4 / 4^4 (65.9 / 1,874.5 /
  16,447.8 steps); random legal play 29 % / 0 % / 0 % (`reports/data/exp01_*/summary.json`).
  The first odd-board follower (an open Gray-code path) reached about 4 % fill (run not
  archived, see exp01) and was replaced by the cycle-minus-corner construction.
- exp02 - 2^4: plain PPO 0.877 (5M) and 0.933 (20M) deterministic completion; loosely gated
  Backplay (rho 0.2) 0.850; strictly gated Backplay (rho 0.9, window 4) 0.963 at 5M with 43.5
  steps per completion (`reports/data/exp02*_best/summary.json`). Failures are collisions with
  one cell left. The strict gate became the default.
- exp03 - 3^4 (81 cells, odd): plain PPO, strict Backplay (gate 0.9) and relaxed Backplay (gate
  0.8) at 30M steps each all end at 0.54-0.57 fill and 0 completions; the curriculum frontier never
  left the endgame (80 -> 76-78) because the corner detour was never mastered above the gate
  (`reports/experiments/exp03_ppo_3x4.md`).
- exp04 - 4^4 (256 cells): Backplay (gate 0.8, window 16, step 8) for 100M steps / 54 min:
  seven frontier advances (255 -> 199) between 18.6M and 53.7M steps, none in the remaining
  46M, and the curriculum success rate fell to 0 from about 73M steps as the learning rate,
  clip range and entropy decayed towards their end values; true-start fill 0.40 (best episode
  0.66), completion 0.000 (`reports/experiments/exp04_ppo_4x4.md`, `reports/data/exp04_ppo_4x4*/`).
- exp05 - 4^4 imitation warm start: behaviour cloning of the route follower reaches 1.000
  expert-action accuracy and the cloned network completes the board in 1.000 +- 0.000 of
  deterministic episodes (0.787 +- 0.041 when sampling), 16,448 steps per game. exp05b fine-tuned
  it with PPO for 20M steps (lr 1e-4, clip 0.1, no entropy, no curriculum): completion stayed at
  1.00 in all nine evaluations but the step count did not move and `approx_kl` stayed below
  2e-5 throughout - a clone whose expert-move probability is about 0.9998 (from the final
  cloning loss) gives PPO nothing to compare, so fine-tuning needs deliberate exploration
  (`reports/experiments/exp05_bc_4x4.md`, `reports/data/exp05*`).

The generated cross-run table is `reports/all_experiments.md`.

## 6. Discussion

On the 16-cell board, masking plus a strictly gated reverse curriculum is what turns "eats a lot"
into "fills the board": the loose gate advanced the frontier before endgames were mastered and
was indistinguishable from no curriculum, while the strict gate gave the best policy at the same
budget. The remaining failures there are endgame traps (boxed in with one free cell), a
global-planning constraint that a one-hot MLP learns only partially from terminal rewards.

On 81 and 256 cells the same recipe fails, for two identifiable reasons. First, the curriculum's
starting states - snakes laid along the Hamiltonian cycle - are not states the true-start policy
ever reaches (cf. Florensa et al. 2017, who adapt the start distribution to the agent's
performance), so skill on curriculum starts (0.88-0.99 fill; 60-84 % success at each gate
crossing before the collapse) did not transfer to the true start (0.4-0.55 fill). Second, the learning-rate, clip-range and entropy schedules
decay with the total budget, not with curriculum progress, so the 4^4 agent lost the ability to
adapt exactly when the frontier reached the hardest states and its endgame success fell to zero.
The odd 3^4 board adds the parity constraint and a two-step corner detour that was never learned.

The imitation warm start (exp05) closes the feasibility gap: the route follower supplies
unlimited (observation, action) pairs on every board, the states cover every fill level at once,
and the resulting network completes the 256-cell board in every deterministic episode. What
remains for RL is efficiency - the clone plays the 16,448-step cycle, while the learned 2^4 agent
showed that PPO can cut a third of the steps by taking shortcuts - and robustness under sampling.
Other candidates for the from-scratch setting are schedules driven by the frontier rather than
the step count, longer budgets with smaller frontier steps, and decision-time search with the
learned value function (AlphaZero-style policy iteration, Silver et al. 2017), as in AlphaSnake.

## 7. Conclusion

The repository delivers a playable 4D snake, a batched training stack that runs 33k PPO steps per
second on one laptop GPU, scripted proofs that every board size is completable (with a new
cycle-minus-corner construction for odd sizes), a reproducible experiment loop, and two results:
model-free MaskablePPO with a Hamiltonian reverse curriculum completes the 16-cell 4D board in
96 % of episodes but not 3^4 or 4^4 within 30M and 100M steps; a neural network behaviour-cloned
from the Hamiltonian follower completes the full 4^4 board in 100 % of deterministic episodes,
and PPO fine-tuning from that clone is the path to a faster learned completer.

## References

- Huang, S. and Ontanon, S. (2020). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. arXiv:2006.14171.
- Resnick, C. et al. (2018). Backplay: "Man muss immer umkehren". arXiv:1807.06919.
- Salimans, T. and Chen, R. (2018). Learning Montezuma's Revenge from a Single Demonstration. arXiv:1812.03381.
- Ng, A., Harada, D. and Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. ICML, 278-287. https://dl.acm.org/doi/10.5555/645528.657613
- Grzes, M. (2017). Reward Shaping in Episodic Reinforcement Learning. AAMAS. https://dl.acm.org/doi/10.5555/3091125.3091208
- Itai, A., Papadimitriou, C. H. and Szwarcfiter, J. L. (1982). Hamilton Paths in Grid Graphs. SIAM J. Comput. 11(4), 676-686. https://doi.org/10.1137/0211056
- Skiena, S. (1990). Implementing Discrete Mathematics: Combinatorics and Graph Theory with Mathematica. Addison-Wesley, p. 148 (as cited by MathWorld, https://mathworld.wolfram.com/GridGraph.html).
- Du, Y. et al. (2022). AlphaSnake: Policy Iteration on a Nondeterministic NP-hard Markov Decision Process. arXiv:2211.09622.
- Andrychowicz, M. et al. (2020). What Matters in On-Policy Reinforcement Learning? arXiv:2006.05990.
- Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. JMLR 22(268), 1-8. https://jmlr.org/papers/v22/20-1364.html
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Towers, M. et al. (2024). Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv:2407.17032.
- Pomerleau, D. A. (1988). ALVINN: An Autonomous Land Vehicle in a Neural Network. NeurIPS. https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html
- Ross, S., Gordon, G. and Bagnell, D. (2011). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger). AISTATS. arXiv:1011.0686.
- Müller, R., Kornblith, S. and Hinton, G. (2019). When Does Label Smoothing Help? NeurIPS. arXiv:1906.02629.
- Florensa, C. et al. (2017). Reverse Curriculum Generation for Reinforcement Learning. CoRL. arXiv:1707.05300.
- Silver, D. et al. (2017). Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm. arXiv:1712.01815.
- Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press. http://incompleteideas.net/book/the-book-2nd.html
- johnflux (2015). Nokia 6110 Part 3 - Algorithms. https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/
- linyiLYi/snake-ai, twanvl/snake, instadeepai/jumanji, Pella86/Snake4d (see THIRD_PARTY_NOTICES.md).
