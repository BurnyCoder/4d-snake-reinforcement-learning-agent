# Experiment 04 - MaskablePPO with Backplay on the 4^4 board (256 cells): the headline run

Date: 2026-09-04. `experiments/exp04_ppo_4x4.env`: 4096 batched envs, `n_steps=64`, batch 8192,
100M steps in 54 minutes on CUDA (31.1k steps/s, the final cumulative `time/fps`); Backplay along the 256-cell Hamiltonian cycle with
`rho=0.8`, window 16, frontier step 8, 500 episodes per gate decision (relaxed from the 2^4
defaults after exp03b stalled at a 90 % gate).

## Question

Can model-free MaskablePPO learn to fill the 4D board from length 1 within a 100M-step budget when
episodes are seeded along the Hamiltonian cycle and a success gate moves the start backwards?

## Hypothesis

The frontier moves from 255 to 1 within the budget (32 advances of 8 cells at the 80 % gate,
each needing at least 500 curriculum episodes), and the true-start completion rate becomes
positive once the frontier passes roughly half the board.

This is the hypothesis stated in this retrospective report, not an independently verified
preregistration. There is one trained checkpoint lineage (training seed 0); evaluation seeds
measure its conditional performance, not training robustness. See the
[scientific audit](../scientific_audit.md) for source-provenance and uncertainty limits.

## Results

| metric | value |
|---|---|
| curriculum frontier | 255 -> 247 (18.6M steps, 9.1 min) -> 239 (22.3M) -> 231 (22.5M) -> 223 (40.6M) -> 215 (46.4M) -> 207 (46.9M) -> 199 (53.7M, 27.1 min); no advance in the remaining 46.4M steps (`run.log`, `progress.csv`) |
| logged curriculum-start success (frontier logged as 199; gate mixes starts) | 0.81 at the advance (53.7M), 0.60 at 56.1M, below 0.001 from ~73M, exactly zero from ~77M to 100M steps |
| rollout fill (all starts) | 0.81 at 26M, falling to 0.71 at 100M |
| true-start fill (training) | mean 0.377, best episode 0.660 (169/256 cells); 81,995 true-start episodes of 624 steps on average (`episodes.json`) |
| true-start completion (eval, 47 evaluations of 100 episodes) | 0.000 throughout |
| best checkpoint, 100 episodes x 3 seeds, deterministic / stochastic | success 0.000 / 0.000, fill 0.401 / 0.395 |
| route follower (exp01) | success 1.000, 16,448 steps |

Figures: `reports/figures/exp04_ppo_4x4_curves.png`, `exp04_ppo_4x4_fill_hist.png`.

## Analysis and learnings

- **Negative result for this run.** No recorded true-start episode completed the 4^4 board; the
  best true-start training episode filled two thirds of it. The route follower demonstrates
  feasibility, but these observations do not show that PPO cannot learn a completer.
- **The curriculum collapsed instead of advancing.** Seven advances happened between 18.6M and
  53.7M steps (gate success 0.80-0.84 each time), then the frontier stayed at 199 and the success
  rate for curriculum starts (57-73 free cells at that frontier) fell to zero in the last quarter
  of training. The learning
  rate and clip range decay over the 100M budget regardless of curriculum progress, but no
  constant-schedule control isolates that choice as the cause. Schedule decay is a hypothesis.
- **Advances did not establish stage mastery.** The callback counted episodes begun before a
  frontier change toward the next gate. Excluding definitely stale episodes changes the
  239 -> 231 gate from 740/892 to 394/537 successes (73.37 %), and the 215 -> 207 gate from
  799/963 to 355/492 (72.15 %). Both are below the required 80 %, and the latter also has fewer
  than 500 episodes. These are filtered retrospective counts, not evaluations of a fixed current
  stage. The [audit](../scientific_audit.md) records their derivation and the resulting
  alternative explanation for curriculum failure.
- **Transfer is unresolved.** Curriculum-start fill (0.88, `episodes.json`) and true-start fill
  (0.38-0.40) have different initial lengths, so their difference is not a measure of transfer.
  There is no matched no-curriculum 4^4 control or state-distribution ablation. The configured
  true-start probability is 20 % per reset, not 20 % of transitions.
- **What the agent did learn**: from length 1 it eats about 100 cells (eval fill 0.40) before
  colliding - 5.2x the random floor (0.077) - in episodes of roughly 570-630 steps (evaluation
  566, training 624). This describes observed food collection; it does not identify an internal
  greedy strategy.

## Decisions and next steps

These are proposals for future experiments; no new training or policy evaluation was performed
for the audit corrections.

1. First associate every episode with its reset frontier so stale episodes cannot establish
   current-stage mastery. Then compare schedule variants with the gate held fixed and independent
   training replications; this separates implementation repair from the schedule hypothesis.
2. Warm-start by imitation: behaviour-clone the policy on (observation, action) pairs from the
   route follower (which is available in unlimited quantity from the batched env), then fine-tune
   with MaskablePPO - the network then starts from a policy that already completes the board and
   RL can then be tested for an efficiency improvement. Exp05 reports the resulting observation;
   it does not isolate the causal reason for the difference from exp04.
3. Smaller frontier steps (4) with a longer budget (300M+ steps, ~3 h), and evaluation of the
   curriculum-start success on a fixed set of frontier states instead of the moving window.
4. As a fallback, search at decision time (AlphaSnake-style MCTS with the learned value function).
