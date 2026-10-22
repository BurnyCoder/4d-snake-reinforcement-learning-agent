# Completing 4D Snake with Masked PPO and a Hamiltonian Reverse Curriculum

Author: Libor Burian

Repository: https://github.com/BurnyCoder/4d-snake-reinforcement-learning-agent - all numbers below come from the run
artifacts copied to `reports/data/` and the per-experiment write-ups in `reports/experiments/`.
This is an exploratory report of fixed checkpoints, each trained with seed 0. The
[scientific audit](scientific_audit.md) records the artifact checks, methodological defects and
source-provenance limits; no new training or policy evaluation was run for these corrections.

## Abstract

We build an N-dimensional snake game (default 4D, 4^4 = 256 cells, 8 moves) whose rules are
implemented once as vectorised numpy over a body-age grid, so thousands of boards step together in
a single process (126k-235k environment steps per second, 33k PPO steps per second end to end on a
laptop GPU). We train sb3-contrib MaskablePPO agents to *complete* the game - fill every cell -
with a legal-action mask from the environment, a sparse reward (+1 food, -1 death, +10 win, -0.001
per step) and a Backplay-style reverse curriculum whose demonstration is a Hamiltonian route
(a cycle for even sizes; for odd sizes a cycle over all cells but one corner with a guarded
detour). Scripted route following completed all evaluated 2^4, 3^4 and 4^4 episodes, providing
a step-count reference (66 / 1,875 / 16,448 steps). On 2^4 the plain PPO checkpoints scored 0.88-0.93
completion; the nominally strict curriculum checkpoint scored 0.963 at the
5M-step budget while completing in 43.5 steps, a third fewer than the route follower's 65.9 on
successful episodes. The curriculum gate mixes episodes begun at different frontiers, so its
advances do not establish mastery of each stage. The strict arm also changed multiple curriculum
parameters, and there are no independent training replications. On the larger boards the
evaluated checkpoints recorded no completions within our budgets: on 3^4 (odd parity) all arms
scored 54-57 % fill, and on 4^4 a 100M-step curriculum run moved the start frontier
from 255 to 199 cells before its curriculum success collapsed; its selected checkpoint reached
40 % fill from the true start (best episode
66 %) and zero completions. Schedule decay and transfer from curriculum starts are hypotheses,
not isolated causes of collapse; gate contamination is an additional confound.
Behaviour-cloning the same network on 262k route-follower decisions spread over all fill levels
(about twelve seconds of supervised training) yields a neural policy that completes the full 4^4 board in
100 % of deterministic evaluation episodes (3 evaluation seeds x 100), at the follower's 16,448 steps.
The recorded PPO fine-tune did not establish an efficiency improvement and completed only one
training episode, a failure. These findings describe the evaluated checkpoints, not algorithm-level
superiority or guaranteed completion.

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
  value in settings with many invalid actions. We use masking but did not ablate it, so its
  necessity in this game is not established.
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
- **Shaping.** Ng, Harada and Russell (1999) give policy-invariance conditions for potential-based
  reward shaping; episodic boundary conditions also matter (Grzes 2017). Shaping is available
  but off in the recorded experiments.
- **Empirical design.** Patterson et al. (2024) distinguish experimental units and sources of
  variability in RL comparisons. Our multiple evaluation seeds quantify fixed-checkpoint
  variation; they do not replace independent training seeds.

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

**Curriculum.** With the current defaults, 80 % of resets place the snake as a segment of length
`U[hi - 4, hi]` along a random rotation of the Hamiltonian route, starting at `hi = C - 1`; the frontier moves back by
`max(1, C/64)` cells when at least 500 accumulated curriculum episodes exceed 90 % success; 20 % of resets
start from length 1 and evaluation always starts from length 1. Per-run configurations override
these defaults. The historical callback admits episodes begun before a frontier change into the
next gate's statistic. Its logged success rate is therefore a mixture of stages, not a valid
current-stage mastery measure. Reset proportions also differ from transition proportions because
episode durations differ; see the [audit](scientific_audit.md) for reconstructed examples.

**Baselines.** The route follower (Hamiltonian cycle, or cycle-minus-corner with a detour that
skips at most as many cycle cells as are free) and uniformly random legal play.

**Evaluation.** 100 episodes x 3 seeds, deterministic and stochastic, masked `evaluate_policy`;
metrics: completion rate, mean fill, steps to complete conditional on winning, win-within-K.
Each learned checkpoint has one training seed (0); `+-` values below are standard deviations
across evaluation-seed means, not confidence intervals or training-seed variability. Checkpoints
were selected using in-training evaluation. The [evaluation protocol](../docs/evaluation.md)
states the uncertainty interpretation and selection limitations.

## 5. Experiments

Experiment configurations live in `experiments/*.env`; every write-up follows question -> hypothesis ->
setup -> results -> learnings. This organization does not establish preregistration: some
configurations were committed after their runs, and recorded commit IDs do not always identify
the complete executed source. These analyses are retrospective.

- exp00 - throughput benchmark: batched env on CUDA, 4096 envs; minibatch 8192 gives 31k fps
  (33k at 16384; 8192 adopted for more gradient steps per rollout).
- exp01 - scripted baselines: route follower 100 % on 2^4 / 3^4 / 4^4 (65.9 / 1,874.5 /
  16,447.8 steps); random legal play 29 % / 0 % / 0 % (`reports/data/exp01_*/summary.json`).
  The first odd-board follower (an open Gray-code path) reached about 4 % fill (run not
  archived, see exp01) and was replaced by the cycle-minus-corner construction.
- exp02 - 2^4: plain PPO 0.877 (5M) and 0.933 (20M) deterministic completion; loosely gated
  Backplay (rho 0.2) 0.850; strictly gated Backplay (rho 0.9, window 4) 0.963 at 5M with 43.5
  steps per completion (`reports/data/exp02*_best/summary.json`). The strict checkpoint's failed
  episodes included endings with one cell left. Relative to the loose arm, the strict arm changed gate
  threshold, window width and minimum episode count together, so the score difference cannot be
  attributed to the threshold alone. Its settings became the defaults.
- exp03 - 3^4 (81 cells, odd): plain PPO, strict Backplay (gate 0.9) and relaxed Backplay (gate
  0.8) at 30M steps each all end at 0.54-0.57 fill and 0 completions; the curriculum frontier never
  left the endgame (80 -> 76-78); these records do not isolate failure to learn the corner detour
  (`reports/experiments/exp03_ppo_3x4.md`).
- exp04 - 4^4 (256 cells): Backplay (gate 0.8, window 16, step 8) for 100M steps / 54 min:
  seven frontier advances (255 -> 199) between 18.6M and 53.7M steps, none in the remaining
  46M, and the logged curriculum success rate fell below 0.001 from about 73M steps and to exactly
  zero from about 77M steps as the learning rate and
  clip range decayed towards their end values; true-start fill 0.40 (best episode
  0.66), completion 0.000 (`reports/experiments/exp04_ppo_4x4.md`, `reports/data/exp04_ppo_4x4*/`).
- exp05 - 4^4 imitation warm start: behaviour cloning of the route follower records 1.000
  expert-action accuracy on training minibatches, without a held-out accuracy measurement, and
  the cloned network completes the board in 1.000 +- 0.000 of
  deterministic episodes (0.787 +- 0.041 when sampling), 16,448 steps per game. exp05b fine-tuned
  it with PPO for 20M steps (lr 1e-4, clip 0.1, entropy coefficient 0, no curriculum): completion stayed at
  1.00 in all nine in-training evaluations, and the selected checkpoint's completion-step mean
  remained close to the clone's. `approx_kl` stayed below 2e-5 but was nonzero, as was policy
  entropy: this does not establish an unchanged policy or identical sampled and argmax actions.
  The run completed only one training episode, a failure, limiting evidence about learning from
  successful full trajectories. No efficiency improvement or causal explanation was established
  (`reports/experiments/exp05_bc_4x4.md`, `reports/data/exp05*`).

The generated cross-run table is `reports/all_experiments.md`; `reports/networks.md` puts every
trained network beside the two scripted baselines in plain words (completion, fill, steps, and
steps per food against a geometric floor).

## 6. Discussion

On the 16-cell board, the checkpoint labelled strict Backplay had the highest observed completion
among the tested from-scratch checkpoints. This is a descriptive ranking: one training seed per
arm, multiple simultaneous curriculum changes and contaminated stage gates prevent an
algorithm-level or threshold-specific conclusion. Some of the strict checkpoint's failures
ended with one free cell, but that outcome alone does not identify its internal planning mechanism.

On 81 and 256 cells the evaluated from-scratch checkpoints recorded no completions. Possible
explanations include limited transfer between curriculum and true-start distributions and
learning-rate/clip-range decay tied to the total budget. Neither was isolated by an ablation;
low true-start performance does not prove that curriculum states are unreachable. The curriculum
defect is a further explanation: stage advances can be supported by results from older, easier
stages. No causal decomposition is possible from these runs. The odd board has a parity
constraint and the scripted baseline uses a corner detour, but the learned policies' failure
mechanism is not established.

The imitation checkpoint demonstrates neural-policy completion in the archived deterministic
4^4 evaluation. Its training accuracy measures fit to expert minibatches; matching the follower's
completion score and mean duration does not establish a head-only internal rule or equivalence
on every state. Sampling produced failures, and the PPO fine-tune provides little evidence about
successful full-trajectory learning. Efficiency and sampling robustness remain research targets.
Frontier-driven schedules, more complete training trajectories, and decision-time search are
candidates for future controlled experiments, not validated remedies.

The [audit](scientific_audit.md) and [evaluation protocol](../docs/evaluation.md) give the complete
limitations: conditional fixed-checkpoint uncertainty, incomplete historical source provenance,
retrospective hypotheses and missing independent training replications. No demonstrated direct
train/test trajectory leakage was found; this does not establish the absence of all selection bias.
Historical timeout-target aliasing affected some runs, but the archived 4^4 curriculum run had no
timeouts and that defect cannot explain its collapse. The measured scores remain descriptive
evidence; neither statistical significance nor causal identification can be supplied by editing
the reports.

## 7. Conclusion

The repository delivers a playable 4D snake, a batched training stack measured at 33k PPO steps per
second on one laptop GPU, scripted route baselines and archived experiment artifacts. The selected
strict-curriculum MaskablePPO checkpoint completed the 16-cell board in 96 % of its evaluation
episodes; the evaluated from-scratch checkpoints recorded no 3^4 or 4^4 completions within the
30M- and 100M-step budgets. The behaviour-cloned 4^4 checkpoint completed 100 % of its archived
deterministic episodes. These observations survive arithmetic verification, while stage-mastery,
algorithm superiority and causal mechanism claims do not. Whether PPO can make the clone more
efficient remains unresolved.

## References

- Patterson, A., Neumann, S., White, M. and White, A. (2024). Empirical Design in Reinforcement Learning. JMLR 25(318), 1-63. https://jmlr.org/papers/v25/23-0183.html
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
