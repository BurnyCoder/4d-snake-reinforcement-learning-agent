# Experiment 00 - Throughput benchmark (which environment/device configuration to train with)

Date: 2026-09-05 (re-measured with the minibatch sweep built into `snake4d bench`; the first grid
of 2026-09-04 gave the same ordering). Command: `uv run snake4d bench --set batch_size=2048
--set run_name=exp00_benchmark` (run `20260905-010410_bench_exp00_benchmark`). Every number below is read from
[reports/data/exp00_benchmark/benchmark.json](../data/exp00_benchmark/benchmark.json); the table
[docs/benchmark.md](../../docs/benchmark.md) is generated from the same file.

## Question

On this laptop (16 logical CPUs, torch using 8 threads,
NVIDIA GeForce RTX 5070 Laptop GPU with 8.0 GB, torch 2.14.0+cu130;
[versions.json](../data/exp00_benchmark/versions.json)), which of the candidate vectorisation
schemes gives the most MaskablePPO training steps per second on the 4^4 board, and what budgets
follow?

## Hypothesis

The research brief predicted, but could not cite a measurement for, the ordering
batched single-process env >> DummyVecEnv >> SubprocVecEnv for a cheap environment like snake:
per-env Python overhead dominates single envs and inter-process communication dominates
subprocesses (SB3 docs on `SubprocVecEnv`,
https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html; Gymnasium paper Fig. 1,
https://arxiv.org/abs/2407.17032; openai/baselines#608, https://github.com/openai/baselines/issues/608).

## Setup

- Board 4^4 (256 cells, observation 1026 floats), MLP [512, 512], `n_epochs=4`, `batch_size=2048`
  for the grid.
- Grid rows: batched `SnakeVecEnv` with 256 / 1024 / 4096 envs (`n_steps=64`); `DummyVecEnv(16)`
  and `SubprocVecEnv(16)` over single `SnakeEnv`s (`n_steps=512`); each on cpu and cuda, with
  `torch.set_num_threads` 1 and 8; about 200k-262k timesteps per row.
- Minibatch sweep: the best grid row re-run with minibatch 2048 / 8192 / 16384.
- fps = `model.num_timesteps / wall time of learn()` (SB3's own `time/fps` is cumulative).
- Env-only rows step the batched env with random legal actions and no policy.

## Results

| configuration | fps | notes |
|---|---|---|
| batched 4096 envs, cuda, batch 2048 | 20,433 | best row of the grid |
| batched 1024 envs, cuda, batch 2048 | 18,843 | |
| batched 256 envs, cuda, batch 2048 | 14,577 | |
| batched 4096 envs, cpu, 8 threads | 5,954 | 1 thread: 1,612 |
| DummyVecEnv(16), best of cpu/cuda | 2,006 | |
| SubprocVecEnv(16), best of cpu/cuda | 1,881 | |
| env only, batched 256 / 1024 / 4096 | 234,823 / 136,074 / 126,304 steps/s | no policy |

Minibatch sweep on the best row: batch 2048 -> 8192 -> 16384 gives 18,735 -> 30,686 ->
32,967 fps (best measured throughput 32,967 fps). Run-to-run variance is about
8 %: the batch-2048 setting measured 20,433 fps in the grid and
18,735 in the sweep.

## Analysis and learnings

- The hypothesis holds with a wide margin: the batched env trains 10x faster than the better
  of the two SB3 vectorisers, and `SubprocVecEnv` is no better than `DummyVecEnv` here (spawned
  workers on Windows spend their time on pickling and pipes, not on the microsecond-scale env step).
- The environment is not the bottleneck: it steps at 126k-235k steps/s
  (4.3-7.9 microseconds per env-step) but PPO reaches
  20k at batch 2048. The minibatch sweep shows the update dominates: 4 epochs x
  128 minibatches = 512 gradient steps per 262,144-sample rollout at batch 2048, each with
  fixed Python overhead; batch 8192 (128 gradient steps) gives 31k fps and 16384
  (64 gradient steps) 33k fps with diminishing returns, so 8192 was adopted.
- Env-only throughput falls as the batch grows (235k at 256 envs to 126k at 4096)
  because the observation tensor (n_envs x 1026 float32, 16 MB at 4096) is rebuilt every step; the
  cost is memory bandwidth, not the game rules.
- CPU with one torch thread is slow (2k fps) because the 1026 -> 512 -> 512 forward pass and
  the update run single-threaded; 8 threads help 3.7x, the GPU another
  3.4x. Training on CPU is a fallback only.
- Larger env batches beat smaller ones on cuda (15k -> 20k from 256 to 4096 envs):
  the per-step fixed costs (get_action_masks list, VecMonitor loop, buffer add) are amortised over
  more envs.

## Decisions

- Train on the batched env with `SNAKE_N_ENVS=4096`, `SNAKE_BATCH_SIZE=8192`, `SNAKE_DEVICE=auto`
  (cuda), `SNAKE_TORCH_THREADS=8`; these are the defaults in `.env.example` and `config.py`.
- Budget arithmetic at ~31k fps: 10M steps = 5 min, 100M steps =
  54 min (the real 100M-step run of exp04 took 54 min at 31.1k fps with
  its evaluations included).
- Evaluation cadence: `SNAKE_EVAL_EVERY=2,097,152` (8 rollouts of 262,144, about a minute) by default;
  the generated recommendation in docs/benchmark.md targets one evaluation per ten minutes instead
  and is deliberately not adopted, because the small boards finish long before that.
