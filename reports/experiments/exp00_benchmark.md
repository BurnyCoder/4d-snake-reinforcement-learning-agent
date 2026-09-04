# Experiment 00 - Throughput benchmark (which environment/device configuration to train with)

Date: 2026-09-04. Command: `uv run snake4d bench` (run `20260904-185344_bench_run`), plus a
minibatch follow-up (see below). Raw numbers: [docs/benchmark.md](../../docs/benchmark.md) and
[reports/data/exp00_benchmark/benchmark.json](../data/exp00_benchmark/benchmark.json).

## Question

On this laptop (Ryzen 7 260, 16 threads, RTX 5070 Laptop 8 GB), which of the candidate
vectorisation schemes gives the most MaskablePPO training steps per second on the 4^4 board, and
what are the resulting budgets?

## Hypothesis

The research brief predicted, but could not cite a measurement for, the ordering
batched single-process env >> DummyVecEnv >> SubprocVecEnv for a cheap environment like snake:
per-env Python overhead dominates single envs and inter-process communication dominates
subprocesses (SB3 docs on `SubprocVecEnv`, Gymnasium paper Fig. 1, openai/baselines#608).

## Setup

- Board 4^4 (256 cells, observation 1026 floats), MLP [512, 512], `n_epochs=4`, `batch_size=2048`.
- Rows: batched `SnakeVecEnv` with 256 / 1024 / 4096 envs (`n_steps=64`); `DummyVecEnv(16)` and
  `SubprocVecEnv(16)` over single `SnakeEnv`s (`n_steps=512`); each on cpu and cuda, with
  `torch.set_num_threads` 1 and 8; about 200k-262k timesteps per row.
- fps = `model.num_timesteps / wall time of learn()` (SB3's own `time/fps` is cumulative).
- Env-only rows step the batched env with random legal actions and no policy.

## Results

| configuration | best fps | notes |
|---|---|---|
| batched 4096 envs, cuda | 20,787 | best row of the grid |
| batched 1024 envs, cuda | 19,371 | |
| batched 256 envs, cuda | 13,333 | |
| batched 4096 envs, cpu, 8 threads | 5,904 | 1 thread: 1,616 |
| DummyVecEnv(16), best of cpu/cuda | 1,978 | cpu 8 threads; cuda 1,725 |
| SubprocVecEnv(16), best of cpu/cuda | 2,223 | cpu 8 threads; cuda 2,084 |
| env only, batched 256 / 1024 / 4096 | 177,858 / 125,988 / 114,394 steps/s | no policy |

Minibatch follow-up on cuda (262,144 timesteps per row): batch 2048 -> 8192 -> 16384 gives
18.4k -> 30.5k -> 34.5k fps at 1024 envs and 20.8k -> 33.8k -> 38.1k fps at 4096 envs.

## Analysis and learnings

- The hypothesis holds with a wide margin: the batched env trains 10x faster than either SB3
  vectoriser, and `SubprocVecEnv` is no better than `DummyVecEnv` here (spawned workers on Windows
  spend their time on pickling and pipes, not on the 100-microsecond env step).
- The environment is not the bottleneck: it steps at 114k-178k steps/s but PPO reaches 21k at
  batch 2048. Profiling by elimination (the minibatch follow-up) shows the update dominates:
  4 epochs x 128 minibatches = 512 gradient steps per 262k-sample rollout, each with fixed Python
  overhead. Raising the minibatch to 8192 (128 gradient steps) gives 34k fps; 16384 gives 38k with
  diminishing returns and only 64 gradient steps per rollout, so 8192 was adopted.
- Env-only throughput falls as the batch grows (178k at 256 envs to 114k at 4096) because the
  observation tensor (n_envs x 1026 float32, 16 MB at 4096) is rebuilt every step; the cost is
  memory bandwidth, not the game rules.
- CPU with one torch thread is very slow (1.5k fps) because the 1026 -> 512 -> 512 forward pass and
  the update run single-threaded; 8 threads help 3.5x, the GPU another 3.5x. Training on CPU is a
  fallback only.
- Larger env batches beat smaller ones on cuda (13k -> 21k from 256 to 4096 envs): the per-step
  fixed costs (get_action_masks list, VecMonitor loop, buffer add) are amortised over more envs.

## Decisions

- Train on the batched env with `SNAKE_N_ENVS=4096`, `SNAKE_BATCH_SIZE=8192`, `SNAKE_DEVICE=auto`
  (cuda), `SNAKE_TORCH_THREADS=8`; these are now the defaults in `.env.example` and `config.py`.
- Budget arithmetic at ~34k fps: 10M steps = 5 min, 100M steps = 50 min, 300M steps = 2.5 h.
- Evaluation cadence: `SNAKE_EVAL_EVERY=2,097,152` (8 rollouts, about a minute) by default;
  experiment files override it for small boards.
