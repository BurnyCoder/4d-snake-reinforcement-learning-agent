# Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `torch.__version__` ends in `+cpu`, or `test_gpu` fails | torch came from PyPI (the Windows PyPI wheel is CPU-only) | keep the `[tool.uv.sources]` / `[[tool.uv.index]]` block for `https://download.pytorch.org/whl/cu130` in `pyproject.toml` and run `uv sync --all-groups`; CUDA 12.6 builds predate sm_120; the cu128 index stops at torch 2.11 (https://download.pytorch.org/whl/cu128/torch/) and cu130 carries 2.14 (https://download.pytorch.org/whl/cu130/torch/) |
| `uv add` / `uv sync` fails with "The process cannot access the file" or "Access is denied" on `.venv/Scripts/snake4d.exe` or `snake4d-0.1.0.dist-info` | a running `uv run snake4d ...` (training, benchmark) holds the console script; uv reinstalls the project on every `uv add` | wait for the run to finish, or use `uv add --no-sync <pkg>` + `uv pip install <pkg>` and `uv sync` later |
| `ValueError: Environment does not support action masking` | the env passed to MaskablePPO is not `SnakeVecEnv`/`SnakeEnv` or lost the `action_masks` method through a wrapper | use `vec_env.make_env` or `make_vec_env(SnakeEnv, ...)`; custom VecEnv wrappers must forward `has_attr`/`env_method` |
| `KeyError: 'fill'` from `Monitor`/`VecMonitor` | an env returned an info dict without `fill`/`is_success`/`start_len` | `SnakeBatch.infos` sets them on every step; keep that contract in any new env |
| `progress.csv` has NaN in `curriculum/*` or `rollout/*` on some rows | those rows were written by the evaluation callback's own `logger.dump` | expected; use `dropna()` per column (report.py does) |
| `n_steps * n_envs must be a multiple of batch_size` | experiment file changed `n_envs` but not `batch_size` | set both (rollout = `n_steps * n_envs`) |
| eval dominates wall time | `eval_every` too small for the measured fps, or 100 long episodes on a big board | raise `SNAKE_EVAL_EVERY` (docs/benchmark.md recommends ~10 min between evals) |
| `SubprocVecEnv` hangs or spawns endlessly | the calling script lacks `if __name__ == "__main__":` (Windows uses spawn) | run through the console script (`uv run snake4d bench`) or add the guard; training never uses SubprocVecEnv |
| pygame window does not open in tests / CI | no display | set `SDL_VIDEODRIVER=dummy` (tests do) |
| `python -c "import pandas"` fails outside `uv run` | the system Python is not the project venv | always prefix commands with `uv run` |
