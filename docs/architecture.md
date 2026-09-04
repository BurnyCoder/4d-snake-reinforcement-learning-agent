# Architecture

(README.md has the overview and the module graph; this page describes each module's contract and
the data flow between them.)

## One physics core, two adapters

`physics.SnakeBatch` is the only implementation of the game rules. It keeps `n` boards as numpy
arrays with a leading batch axis:

| array | shape | meaning |
|---|---|---|
| `age` | `(n, C)` int16 | body-age grid: 0 empty, 1 tail, `length` head |
| `head`, `food` | `(n,)` int64 | flat cell indices |
| `length`, `idle`, `t`, `start_len` | `(n,)` int32 | snake length, steps since food, episode steps, start length |

`step(actions)` returns `(reward, terminated, truncated)` arrays; `observe()` builds the
`(n, 4C + 2)` float32 observation; `action_masks()` the `(n, 2*ndim)` legal-move mask; `infos()`
the per-row dicts (`fill`, `is_success`, `start_len`); `reset(rows)` restarts the given rows
(with curriculum starts when a frontier is set); `board(row)` returns cell codes for rendering.

- `env.SnakeEnv(gym.Env)` = `SnakeBatch(n=1)` behind the Gymnasium API. Used by the env checkers,
  human play, and the DummyVecEnv/SubprocVecEnv rows of the benchmark.
- `vec_env.SnakeVecEnv(VecEnv)` = `SnakeBatch(n=N)` behind stable-baselines3's VecEnv API.
  `vec_env.make_env(cfg, n_envs, seed, monitor_path)` wraps it in `VecMonitor`, which writes
  `info["episode"] = {r, l, t, is_success, fill, start_len}` at episode end and the
  `*.monitor.csv` file. Training and evaluation both use this factory.

The custom-VecEnv contract (SB3 v2.9.0) is honoured explicitly: the eight abstract methods
(`reset, step_async, step_wait, close, get_attr, set_attr, env_method, env_is_wrapped`);
`render_mode`/`num_envs` set before `VecEnv.__init__` (which calls `get_attr("render_mode")`);
`get_attr` raising `AttributeError` for unknown names so `has_attr` / sb3-contrib's
`is_masking_supported` work; `env_method("action_masks")` returning one row per env;
`env_method("set_curriculum", hi, window, p_true_start)` for the curriculum; `step_wait`
auto-resetting finished rows with `terminal_observation` and `TimeLimit.truncated` in their info.

## Phases (all reachable through `snake4d.main`)

| phase | module | reads | writes |
|---|---|---|---|
| `bench` | `benchmark.py` | Config | `runs/<ts>_bench_*/benchmark.json`, `docs/benchmark.md` |
| `play` | `play.py` | Config | `runs/<ts>_play_*/run.log` |
| `train` | `train.py`, `callbacks.py` | Config, optional `SNAKE_MODEL_PATH` | `runs/<ts>_train_<name>/` (`run.log`, SB3 `log.txt`, `progress.csv`, TensorBoard events, `monitor.monitor.csv`, `eval/evaluations.npz`, `best_model.zip`, `checkpoints/`, `final_model.zip`, `config.json`, `versions.json`) |
| `imitate` | `imitation.py`, `agents.py`, `train.py` | Config | `runs/<ts>_imitate_<name>/bc_model.zip` (policy behaviour-cloned on route-follower data over all snake lengths) |
| `evaluate` | `evaluation.py`, `agents.py` | Config (`SNAKE_MODEL_PATH` or `SNAKE_POLICY`) | `runs/<ts>_evaluate_<name>/` (`summary.json`, `eval_episodes.csv`, `run.log`) |
| `report` | `report.py` | every run directory | `reports/figures/`, `reports/data/`, `reports/all_experiments.md`, optional `reports/paper.pdf` |
| `pipeline` | `main.py` | Config | train -> evaluate -> report |

`main.py` builds one `Config` (defaults -> `.env` -> `--env-file` -> `--set`) and calls the phase's
`run(cfg)`. Phases import each other's modules lazily so `play` never imports torch and `train`
never imports pygame.

## Data flow of a training step

1. `MaskablePPO.collect_rollouts` asks the env for `action_masks` (`get_action_masks` ->
   `VecMonitor.env_method` -> `SnakeVecEnv.env_method` -> `SnakeBatch.action_masks`).
2. The policy samples one action per board from the masked categorical distribution.
3. `SnakeVecEnv.step_wait` calls `SnakeBatch.step`, builds infos, marks done rows, resets them,
   and re-observes only those rows.
4. `VecMonitor` adds `info["episode"]` for done rows; SB3 feeds `is_success` into
   `rollout/success_rate`; `callbacks.FillLogger` records fill statistics and
   `callbacks.Backplay` moves the curriculum frontier from the same infos.
5. After `n_steps` vec-steps SB3 runs `n_epochs` of minibatch updates; the logger dumps a row to
   `progress.csv` / TensorBoard; every `eval_every` steps `MaskableEvalCallback` evaluates
   `eval_episodes` true-start episodes on a separate batched env.

## Routes and the curriculum

`hamilton.route_for(size, ndim)` returns the demonstration route: a Hamiltonian cycle for even
sizes, a cycle over every cell except the corner for odd sizes (see docs/game_rules.md), the
Gray-code path for 1D. `SnakeBatch.reset` lays curriculum snakes along a random rotation of this
route; `agents.RoutePolicy` follows it as the scripted baseline.
