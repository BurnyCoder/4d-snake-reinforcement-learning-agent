# The ten trained networks beside the two scripted baselines

(README.md has the overview and the headline results; [docs/game_rules.md](../docs/game_rules.md),
[docs/rl_design.md](../docs/rl_design.md) and [docs/evaluation.md](../docs/evaluation.md) define the
game, the learning setup and the metrics; [experiments/](experiments/) analyses each experiment.
This page only puts the ten trained networks side by side with the two scripted baselines, in plain
words, and repeats none of those texts.)

All scores are the `deterministic=True` entries of `data/<run>/summary.json` (100 episodes x 3 seeds
from length 1, protocol in [docs/evaluation.md](../docs/evaluation.md); the sampling-mode numbers
are in the experiment write-ups). The baselines are `data/exp01_route_<n>x4`, the Hamiltonian loop
follower that completes every board and sets the step count to beat, and
`data/exp01_random_<n>x4`, random legal play, the floor ([exp01](experiments/exp01_baselines.md)).
Budgets, curricula and wall times come from the `experiments/*.env` files and the `wall_min` column
of [all_experiments.md](all_experiments.md). Steps per food, the geometric floor, the non-eating
share and the weight counts are the authors' own arithmetic from those files, an estimate of route
efficiency rather than a measurement. `tests/test_docs_numbers.py` recomputes every bold number,
every baseline number and every verdict on this page from `reports/data/`.

## What all ten networks share

| component | technical, where it is defined | in plain words |
|---|---|---|
| Domain | 4-dimensional snake on an `n^4` hypercube with `n` = 2, 3 or 4 cells per side, so 16, 81 or 256 cells and 8 moves, plus or minus along each axis: [docs/game_rules.md](../docs/game_rules.md). | Ordinary snake with a 4D cube for a board. You cannot picture it, but the rules are the usual ones: eat, grow, never hit a wall or yourself. |
| Input | A flat vector of `4C + 2` numbers: body map, time-to-vacate map, head one-hot, food one-hot, length and hunger (the observation table of [docs/game_rules.md](../docs/game_rules.md)). | The network never sees a picture. It gets one spreadsheet row describing the whole board, 66 numbers on the smallest board and 1,026 on the largest. |
| Architecture | sb3-contrib `MaskablePPO` with `MlpPolicy` and `net_arch {"pi": [512, 512], "vf": [512, 512]}` ([train.py](../src/snake4d/train.py)); 8 move scores and 1 value out; 0.60M / 0.86M / 1.58M weights on 2^4 / 3^4 / 4^4, own arithmetic from the layer sizes. | The plainest kind of neural net: two layers of 512 units turn the input row into a score for each of the 8 moves, and a twin stack guesses how well the game is going. No convolution, no memory between steps. |
| Action masking | Moves into walls, occupied cells or the neck are removed before the choice (the action-mask section of [docs/game_rules.md](../docs/game_rules.md)). | The game hands the net a list of legal moves. It can never die on the very next step, only get boxed in a few steps later. |
| RL algorithm and reward | MaskablePPO on `+1` food, `-1` death, `+10` win, `-0.001` per step; discount 0.99, entropy bonus 0.01, linearly decaying learning rate and clip range: [docs/rl_design.md](../docs/rl_design.md). | Trial and error across thousands of games played at once. Moves that led to more reward get more probability next time, the rest less. A small bonus for variety keeps it exploring. |
| Imitation algorithm | Behaviour cloning of the route follower: cross-entropy of the expert move under the masked policy, Adam 1e-3, 20 epochs over 262,144 samples ([imitation.py](../src/snake4d/imitation.py), [exp05](experiments/exp05_bc_4x4.md)). | Flashcards. Show the board, reveal the teacher's move, adjust the weights until the net gives the same answer. The teacher is a fixed loop through every cell. |
| Training data | Nothing stored: the batched simulator generates experience live, tens of thousands of moves per second ([docs/benchmark.md](../docs/benchmark.md)). | There is no dataset. The game itself is the data source. |
| Curriculum | Backplay along the Hamiltonian route: 80 % of training episodes start as a route segment near a frontier close to a full board, and the frontier moves back only when the success rate clears a gate ([docs/rl_design.md](../docs/rl_design.md)). | Practise the last few moves of the game first, then the last ten, then twenty, and so on. Evaluation always starts from length 1. |
| Baselines | Route follower (walks the Hamiltonian cycle and completes every board) and masked random play, evaluated by the same protocol: [exp01](experiments/exp01_baselines.md). | The loop is the safe-but-slow reference every network is measured against. Random is what no skill looks like. |
| Evaluation | 100 episodes x 3 seeds through masked `evaluate_policy`, a deterministic and a sampling pass ([docs/evaluation.md](../docs/evaluation.md)); this page shows the deterministic pass. | Every number below comes from 300 fresh games with the network always taking its top-scored move. |
| Efficiency metrics | Steps per food, geometric floor (2.00 / 3.56 / 5.00 moves on 2^4 / 3^4 / 4^4), non-eating move share and won within `4C` moves: the derived-metrics section of [docs/evaluation.md](../docs/evaluation.md). | Steps per food is how far the snake walks between meals. Times floor is how many times farther than a player who always took the shortest path to the food. Non-eating moves is the share of moves that ate nothing. |

## The ten networks, scores beside baselines

How to read a score cell: the network's own score in bold, then the loop follower's score with a
verdict, then random play's score with a verdict. Verdicts are from the network's point of view and
are judged at the printed precision, so equal digits are a tie. `never` means no episode finished.
Budgets are environment moves.

| run | board | training | completion | fill | steps to finish |
|---|---|---|---|---|---|
| [exp02](data/exp02_ppo_2x4_best/summary.json) | 2^4, 16 cells | MaskablePPO from scratch, no curriculum, 5M moves, 2.2 min ([write-up](experiments/exp02_ppo_2x4.md)) | **87.7 %** (loop 100.0 %, worse; random 29.3 %, better) | **0.987** (loop 1.000, worse; random 0.816, better) | **34.4** (loop 65.9, better; random 129.4, better) |
| [exp02b](data/exp02b_ppo_2x4_backplay_best/summary.json) | 2^4 | PPO, loose Backplay (advance at 20 % success, window 8), 5M moves, 2.3 min | **85.0 %** (loop 100.0 %, worse; random 29.3 %, better) | **0.980** (loop 1.000, worse; random 0.816, better) | **34.6** (loop 65.9, better; random 129.4, better) |
| [exp02c](data/exp02c_ppo_2x4_long_best/summary.json) | 2^4 | PPO, no curriculum, 20M moves, 8.6 min | **93.3 %** (loop 100.0 %, worse; random 29.3 %, better) | **0.991** (loop 1.000, worse; random 0.816, better) | **35.7** (loop 65.9, better; random 129.4, better) |
| [exp02d](data/exp02d_ppo_2x4_backplay_strict_best/summary.json) | 2^4 | PPO, strict Backplay (advance at 90 % success, window 4), 5M moves, 2.1 min | **96.3 %** (loop 100.0 %, worse; random 29.3 %, better) | **0.994** (loop 1.000, worse; random 0.816, better) | **43.5** (loop 65.9, better; random 129.4, better) |
| [exp03a](data/exp03a_ppo_3x4_nocur_best/summary.json) | 3^4, 81 cells | PPO, no curriculum, 30M moves, 9.9 min ([write-up](experiments/exp03_ppo_3x4.md)) | **0.0 %** (loop 100.0 %, worse; random 0.0 %, tie) | **0.566** (loop 1.000, worse; random 0.226, better) | never (loop 1,874.5; random never) |
| [exp03b](data/exp03b_ppo_3x4_backplay_best/summary.json) | 3^4 | PPO, strict Backplay (90 % gate, window 4), 30M moves, 9.7 min | **0.0 %** (loop 100.0 %, worse; random 0.0 %, tie) | **0.549** (loop 1.000, worse; random 0.226, better) | never (loop 1,874.5; random never) |
| [exp03c](data/exp03c_ppo_3x4_backplay_relaxed_best/summary.json) | 3^4 | PPO, relaxed Backplay (80 % gate, window 8, frontier step 4), 30M moves, 10.6 min | **0.0 %** (loop 100.0 %, worse; random 0.0 %, tie) | **0.542** (loop 1.000, worse; random 0.226, better) | never (loop 1,874.5; random never) |
| [exp04](data/exp04_ppo_4x4_best/summary.json) | 4^4, 256 cells | PPO, Backplay (80 % gate, window 16, frontier step 8), 100M moves, 53.7 min ([write-up](experiments/exp04_ppo_4x4.md)) | **0.0 %** (loop 100.0 %, worse; random 0.0 %, tie) | **0.401** (loop 1.000, worse; random 0.077, better) | never (loop 16,447.8; random never) |
| [exp05](data/exp05_bc_4x4_eval/summary.json) | 4^4 | behaviour cloning of the route follower, 262,144 samples x 20 epochs, about 12 s ([write-up](experiments/exp05_bc_4x4.md)) | **100.0 %** (loop 100.0 %, tie; random 0.0 %, better) | **1.000** (loop 1.000, tie; random 0.077, better) | **16,447.8** (loop 16,447.8, tie; random never, better) |
| [exp05b](data/exp05b_ppo_4x4_from_bc_best/summary.json) | 4^4 | PPO fine-tune from the exp05 clone, no curriculum, no entropy bonus, lr 1e-4 to 1e-5, 20M moves, 22.5 min | **100.0 %** (loop 100.0 %, tie; random 0.0 %, better) | **1.000** (loop 1.000, tie; random 0.077, better) | **16,413.6** (loop 16,447.8, better; random never, better) |

What happened, one line per network (details and figures in the write-ups):

- exp02: learns fast, then plateaus; most losses are getting boxed in with one cell left.
- exp02b: the loose gate ran the curriculum frontier from 15 to 1 in about fifteen seconds of
  training, so it taught nothing; same as plain PPO within noise.
- exp02c: four times the budget buys about 5 points of completion.
- exp02d: the best genuinely learned agent; slower per game than exp02 but wins far more often, and
  equally good when its moves are sampled instead of taken by argmax.
- exp03a: eats about 45 cells and then traps itself; its best single game reached 74 of 81 cells.
- exp03b: the frontier moved back one cell in 30M moves; the odd board's corner detour, a two-step
  plan, was never learned.
- exp03c: the relaxed gate moved the frontier to 76 of 80 cells; endgame practice did not transfer
  to games started from length 1.
- exp04: the frontier went from 255 to 199 and then endgame success collapsed as the learning rate
  decayed; the policy learned safe foraging to about 100 cells (fill 0.40) but not coverage.
- exp05: expert-action accuracy 1.000 after cloning; a memorised head-position-to-move lookup with
  the loop's exact step count.
- exp05b: kept 100 % but changed almost nothing (`approx_kl` below 2e-5); a clone this certain of
  its moves gives PPO no alternatives to compare.

## Efficiency beside baselines

Same cell format. For the networks that never complete the board, these ratios describe only the
part of the game they survived, the emptier and easier part, so they flatter those networks against
the loop, which is measured over whole games.

| run | board | steps per food | x geometric floor | non-eating moves | won within 4C moves |
|---|---|---|---|---|---|
| [exp02](data/exp02_ppo_2x4_best/summary.json) | 2^4 | **2.31** (loop 4.39, better; random 9.82, better) | **1.16x** (loop 2.20x, better; random 4.91x, better) | **56.8 %** (loop 77.2 %, better; random 89.8 %, better) | **87.7 %** (loop 45.3 %, better; random 0.7 %, better) |
| [exp02b](data/exp02b_ppo_2x4_backplay_best/summary.json) | 2^4 | **2.32** (loop 4.39, better; random 9.82, better) | **1.16x** (loop 2.20x, better; random 4.91x, better) | **56.8 %** (loop 77.2 %, better; random 89.8 %, better) | **85.0 %** (loop 45.3 %, better; random 0.7 %, better) |
| [exp02c](data/exp02c_ppo_2x4_long_best/summary.json) | 2^4 | **2.38** (loop 4.39, better; random 9.82, better) | **1.19x** (loop 2.20x, better; random 4.91x, better) | **58.0 %** (loop 77.2 %, better; random 89.8 %, better) | **93.3 %** (loop 45.3 %, better; random 0.7 %, better) |
| [exp02d](data/exp02d_ppo_2x4_backplay_strict_best/summary.json) | 2^4 | **2.89** (loop 4.39, better; random 9.82, better) | **1.45x** (loop 2.20x, better; random 4.91x, better) | **65.4 %** (loop 77.2 %, better; random 89.8 %, better) | **96.3 %** (loop 45.3 %, better; random 0.7 %, better) |
| [exp03a](data/exp03a_ppo_3x4_nocur_best/summary.json) | 3^4 | **4.07** (loop 23.43, better; random 73.98, better) | **1.14x** (loop 6.59x, better; random 20.81x, better) | **75.4 %** (loop 95.7 %, better; random 98.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |
| [exp03b](data/exp03b_ppo_3x4_backplay_best/summary.json) | 3^4 | **4.11** (loop 23.43, better; random 73.98, better) | **1.16x** (loop 6.59x, better; random 20.81x, better) | **75.7 %** (loop 95.7 %, better; random 98.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |
| [exp03c](data/exp03c_ppo_3x4_backplay_relaxed_best/summary.json) | 3^4 | **4.18** (loop 23.43, better; random 73.98, better) | **1.18x** (loop 6.59x, better; random 20.81x, better) | **76.1 %** (loop 95.7 %, better; random 98.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |
| [exp04](data/exp04_ppo_4x4_best/summary.json) | 4^4 | **5.57** (loop 64.50, better; random 248.65, better) | **1.11x** (loop 12.90x, better; random 49.73x, better) | **82.0 %** (loop 98.4 %, better; random 99.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |
| [exp05](data/exp05_bc_4x4_eval/summary.json) | 4^4 | **64.50** (loop 64.50, tie; random 248.65, better) | **12.90x** (loop 12.90x, tie; random 49.73x, better) | **98.4 %** (loop 98.4 %, tie; random 99.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |
| [exp05b](data/exp05b_ppo_4x4_from_bc_best/summary.json) | 4^4 | **64.37** (loop 64.50, better; random 248.65, better) | **12.87x** (loop 12.90x, better; random 49.73x, better) | **98.4 %** (loop 98.4 %, tie; random 99.6 %, better) | **0.0 %** (loop 0.0 %, tie; random 0.0 %, tie) |

What the comparisons say:

- On 16 cells every network beats random on every measure and beats the loop on every efficiency
  measure, but none reaches the loop's 100 % completion. The strict-curriculum agent comes closest,
  3.7 points short, while finishing games in a third fewer moves than the loop.
- On 81 and 256 cells the from-scratch networks tie random on completion at 0 %. They beat random on
  fill by two and a half times and five times, and they beat the loop on every efficiency measure by
  a wide margin, but only over the games they survived, which end with the board 40 to 57 % full.
- The clone ties the loop on every single measure: it is the loop in disguise. It beats random by
  100 points of completion, and fine-tuning changed its step count by 0.2 %.
- The gap nobody fills: on 4^4 the network that walks efficiently, 5.57 moves per meal at 1.11 times
  the floor, never finishes, and the network that always finishes walks 64.5 moves per meal, 12.9
  times the floor. A player with both would finish in about 1,300 moves instead of 16,448.
