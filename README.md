# 4D Snake + reinforcement learning

An N-dimensional snake game (default **4D**: a `4 x 4 x 4 x 4` board, 256 cells, 8 move
directions) and a **MaskablePPO** agent trained to *complete* it, i.e. grow until the snake fills
every cell. Everything runs on one Windows laptop (RTX 5070 Laptop GPU); the game is playable by a
human, the training pipeline is fully scripted, and every experiment is written up in
[reports/](reports/).

## Methodology in one page

- **Game.** `size**ndim` cells addressed by flat index; action `a` moves along axis `a >> 1`
  (+1 if `a` is even, else -1). The state is a *body-age grid* (tail = 1 ... head = length): moving
  the tail is one decrement, collisions are one lookup, and all `N` training boards are stepped
  together with vectorised numpy. Eating grows the snake by one; the board is complete when
  `length == size**ndim`. Starvation (more than `4 * cells` consecutive steps without food) and
  an absolute cap (`cells^2` steps) truncate the episode: the ordinary step cost, never the
  death penalty. Rules: [docs/game_rules.md](docs/game_rules.md).
- **Agent.** sb3-contrib `MaskablePPO` with an MLP `[512, 512]` on a flat observation of
  `4 * cells + 2` floats (body, time-to-vacate, head, food, length, hunger). The environment
  supplies a legal-action mask (walls, occupied cells, the neck), so the agent only chooses
  *which* safe move fills the board. Reward: +1 food, -1 death, +10 win, -0.001 per step; no
  distance shaping. Design and hyper-parameters: [docs/rl_design.md](docs/rl_design.md).
- **Curriculum.** A Hamiltonian route (a cycle for even sizes; for odd sizes a cycle over all
  cells but one corner) is both the scripted baseline that proves every board is completable and
  the demonstration for a **Backplay** reverse curriculum: episodes start as long route segments
  near a full board and the start moves backwards as the success rate allows. Evaluation always
  starts from length 1.
- **Evaluation.** 100 episodes x 3 seeds through masked `evaluate_policy` (trained models:
  deterministic and stochastic passes; scripted baselines: one deterministic pass); headline
  metrics are the completion rate, mean fill and steps to complete,
  compared with the route follower (ceiling) and masked random play (floor):
  [docs/evaluation.md](docs/evaluation.md).
- **Science loop.** Each experiment is pre-registered as an `.env` override file in
  [experiments/](experiments/) and written up (question, hypothesis, setup, results, learnings)
  in [reports/experiments/](reports/experiments/); [reports/all_experiments.md](reports/all_experiments.md)
  is the generated cross-run table and [reports/paper.md](reports/paper.md) the paper (compiled to
  `reports/paper.pdf` by `uv run --group docs snake4d report --pdf`).

## Install

Requirements: Windows/Linux, [uv](https://docs.astral.sh/uv/) 0.12+, an NVIDIA GPU is optional
(CPU works; [docs/benchmark.md](docs/benchmark.md) measured the same batched setup about 3.4x
slower on 8 CPU threads than on the GPU). Python 3.13 is installed by uv automatically.

```bash
git clone https://github.com/BurnyCoder/4d-snake-reinforcement-learning-agent.git
cd 4d-snake-reinforcement-learning-agent
uv sync --all-groups          # creates .venv with torch 2.14 (CUDA 13.0 wheels), SB3 2.9, gymnasium 1.3, ...
cp .env.example .env          # optional: edit any SNAKE_* key (all keys are documented there)
uv run pytest -m "not slow and not gpu"   # fast suite, about 20 s; run -m gpu for the CUDA smoke test
```

Every setting is a `SNAKE_<FIELD>` key of [src/snake4d/config.py](src/snake4d/config.py):
defaults < `.env` < `--env-file experiments/<exp>.env` < `--set field=value`.

## Use

```bash
uv run snake4d play                              # human play on the 4^4 board (--set size=2 for the small one): WASD = x/y, IJKL = z/w, R restart, Esc quit
uv run snake4d watch --set model_path=exp02d_ppo_2x4_backplay_strict   # watch the best learned network play 2^4 (a run name under runs/, or a .zip path)
uv run snake4d watch --set model_path=exp04_ppo_4x4                    # the 4^4 network trained from scratch: an efficient forager that never finishes
uv run snake4d watch --set model_path=exp05b_ppo_4x4_from_bc --set watch_speed=64   # the 4^4 finisher (a cloned loop follower; 16k-move games)
uv run snake4d watch --set policy=route                                # the scripted Hamiltonian loop itself
uv run snake4d bench                             # throughput grid -> docs/benchmark.md
uv run snake4d evaluate --set policy=route       # scripted baseline on the default 4^4 board
uv run snake4d train --env-file experiments/exp02_ppo_2x4.env
uv run snake4d imitate --env-file experiments/exp05_bc_4x4.env   # behaviour-clone the route follower -> bc_model.zip
uv run snake4d train --set model_path=runs/<imitate run>/bc_model.zip   # PPO fine-tune from the clone
uv run snake4d evaluate --set model_path=runs/<run>/best_model.zip --set size=2 --set ndim=4
uv run snake4d report                            # figures + reports/all_experiments.md
uv run --group docs snake4d report --pdf         # ... plus reports/paper.pdf
uv run --group hub snake4d publish               # checkpoints + model cards -> Hub repos and a collection (after `hf auth login`)
uv run snake4d pipeline --env-file experiments/exp04_ppo_4x4.env   # train -> evaluate -> report
uv run tensorboard --logdir runs                 # optional live curves
```

## Play and watch

`play` and `watch` share one window: the 4D board is drawn as a `z`-by-`w` grid of `x`-by-`y`
tiles (tile rows labelled `z0..`, tile columns `w0..`), the head yellow, the food red, the body
green shading from dark at the tail to bright at the neck. A move along `x` or `y` stays inside a
tile; a move along `z` or `w` jumps to the neighbouring tile. `watch` plays game after game (Space
pauses, N plays one move, `+`/`-` double or halve the speed, R restarts) and infers the board size
from the checkpoint. `SNAKE_MODEL_PATH` takes a run name (the newest `train`/`imitate` run of that
name under `runs/`) or a `.zip`; `SNAKE_DETERMINISTIC=0` samples moves instead of taking the
argmax; `SNAKE_WATCH_GIF=1` records the first game as `runs/<ts>_watch_<name>/game.gif`.

Which network to watch: [reports/networks.md](reports/networks.md) compares all ten. The
strict-curriculum 2^4 agent exp02d is the best genuinely learned one, completing 96.3 % of its
games in 43.5 moves against the loop follower's 65.9; the two 4^4 networks that always finish are
behaviour-cloned copies of that loop, and the 4^4 network trained from scratch walks efficiently
but never finishes (fill 0.40).

![exp02d playing one 2^4 game](reports/figures/exp02d_ppo_2x4_backplay_strict_game.gif)

Without training anything, the weights come from the Hub
(https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download):

```bash
uv run --group hub hf download BurnyCoder/4d-snake-exp02d-ppo-2x4-backplay-strict best_model.zip --local-dir weights
uv run snake4d watch --set model_path=weights/best_model.zip
```

## What happens when you run it (data flow)

1. `snake4d.main` builds one `Config` and calls the phase's `run(cfg)`; the phase creates
   `runs/<timestamp>_<phase>_<name>/` with `config.json`, `versions.json` (library versions, GPU,
   git commit) and `run.log` (every log line, timestamped, also printed to the terminal).
2. `train`: `vec_env.make_env` builds `N` boards in one `SnakeBatch` behind an SB3 `VecEnv` +
   `VecMonitor`. Each PPO step asks the env for action masks, samples one move per board, steps
   all boards with numpy, auto-resets finished ones (curriculum starts are laid along the route),
   and records episode fill/success. SB3 writes `progress.csv`, `log.txt` and TensorBoard events;
   `MaskableEvalCallback` evaluates 100 true-start episodes every `eval_every` steps into
   `eval/evaluations.npz` and keeps `best_model.zip`; checkpoints and `final_model.zip` are saved.
3. `evaluate`: the same batched env with one episode per board runs the model or a scripted
   policy through sb3-contrib's masked `evaluate_policy` for every seed; `summary.json` and
   `eval_episodes.csv` are written.
4. `watch`: loads the checkpoint (or a scripted policy), infers the board from its input and
   output sizes, and plays it in the `play` window, one masked `predict` per move; every game's
   outcome goes to `run.log` and, when asked, the first game to `game.gif`.
5. `report`: reads every experiment run directory, draws `reports/figures/<run>_curves.png` and
   `<run>_fill_hist.png`, copies the small artifacts to `reports/data/<run>/`, writes the table in
   `reports/all_experiments.md`, and with `--pdf` compiles `reports/paper.pdf`.
6. `publish`: for every run name in `SNAKE_PUBLISH_RUNS`, copies the evaluated checkpoint,
   `config.json`, `versions.json`, the evaluation files and, for training runs, the SB3 log and the
   figures into `runs/<ts>_publish_*/staging/<repo>/`, writes a model card, uploads the folder to the
   Hub repo `<namespace>/4d-snake-<run>`, adds it to the collection and records everything in
   `published.json`.

## Architecture

```mermaid
flowchart LR
    subgraph cli [CLI]
        main[main.py: phases bench / play / watch / train / imitate / evaluate / report / publish / pipeline]
        config[config.py: Config from defaults, .env, --env-file, --set]
        logging[logging_utils.py: run dirs, run.log, versions.json]
    end
    subgraph core [Game core]
        grid[grid.py: neighbour table, parity]
        hamilton[hamilton.py: Hamiltonian cycle / path]
        physics[physics.py: SnakeBatch, batched rules, masks, observation, curriculum resets]
        env[env.py: SnakeEnv, Gymnasium adapter, n=1]
        vec[vec_env.py: SnakeVecEnv + VecMonitor, n=N]
        render[render.py: slice montage, ASCII, RGB]
    end
    subgraph rl [Learning]
        train[train.py: MaskablePPO, SB3 logger, callbacks]
        imitation[imitation.py: behaviour cloning of the route follower]
        callbacks[callbacks.py: FillLogger, Backplay]
        agents[agents.py: RoutePolicy, RandomMaskedPolicy]
        evaluation[evaluation.py: masked evaluate_policy, summary.json]
        bench[benchmark.py: throughput grid]
    end
    subgraph out [Outputs]
        play[play.py: pygame window]
        watch[watch.py: a checkpoint plays in the window]
        report[report.py: figures, tables, PDF]
        publish[publish.py: model cards, Hub repos, collection]
        runs[(runs/ ... progress.csv, evaluations.npz, models)]
        reports[(reports/ ... figures, data, all_experiments.md, paper.pdf)]
        hub[(Hugging Face Hub)]
    end
    main --> config
    main --> train & evaluation & bench & play & watch & report & imitation & publish
    logging --> train & evaluation & bench & play & watch & report & imitation & publish
    evaluation --> watch
    play --> watch
    report --> publish
    runs --> publish --> hub
    agents --> imitation --> runs
    train --> imitation
    vec --> imitation
    env --> bench
    train --> bench
    grid --> physics
    hamilton --> physics
    physics --> env & vec
    physics --> render --> play
    env --> play
    vec --> train & evaluation & bench
    callbacks --> train
    agents --> evaluation
    hamilton --> agents
    train --> runs
    evaluation --> runs
    bench --> runs
    runs --> report --> reports
```

Module contracts and the training data flow in detail: [docs/architecture.md](docs/architecture.md).
Measured throughput and the resulting defaults: [docs/benchmark.md](docs/benchmark.md).
Known pitfalls (CUDA wheels, Windows file locks, masking errors): [docs/troubleshooting.md](docs/troubleshooting.md).

## Results

The cross-experiment table is generated in [reports/all_experiments.md](reports/all_experiments.md);
each experiment's write-up is under [reports/experiments/](reports/experiments/); the paper is
[reports/paper.md](reports/paper.md) / [reports/paper.pdf](reports/paper.pdf); all ten trained
networks beside both scripted baselines, in plain words: [reports/networks.md](reports/networks.md).
The weights, configs and evaluation files of those ten networks are on the Hugging Face Hub, one
repo per run (`BurnyCoder/4d-snake-<run>`, uploaded by `snake4d publish`), in the collection
[4D Snake RL: all evaluated networks](https://huggingface.co/collections/BurnyCoder/4d-snake-rl-all-evaluated-networks-6a9d0a0a66c7efcd101b7741).

| board | route follower ([exp01](reports/data/exp01_route_4x4/summary.json)) | random legal play ([exp01](reports/data/exp01_random_4x4/summary.json)) | best learned agent | write-up, data |
|---|---|---|---|---|
| 2^4 (16 cells) | 100 %, 66 steps | 29 % | **96.3 %** completion, 43.5 steps (MaskablePPO + strict-gate Backplay, 5M steps) | [exp02](reports/experiments/exp02_ppo_2x4.md), [data](reports/data/exp02d_ppo_2x4_backplay_strict_best/summary.json) |
| 3^4 (81 cells, odd) | 100 %, 1,875 steps | 0 % (fill 0.23) | 0 % completion, fill 0.54-0.57 (30M steps, with or without curriculum) | [exp03](reports/experiments/exp03_ppo_3x4.md), [data](reports/data/exp03a_ppo_3x4_nocur_best/summary.json) |
| 4^4 (256 cells) | 100 %, 16,448 steps | 0 % (fill 0.08) | PPO + curriculum from scratch: 0 %, fill 0.40 (100M steps). **Behaviour-cloned network: 100 % completion** (deterministic, 3 x 100 episodes), 16,448 steps; PPO fine-tuning kept 100 % but left the policy unchanged (near-deterministic clone, approx_kl below 2e-5) | [exp04](reports/experiments/exp04_ppo_4x4.md), [exp05](reports/experiments/exp05_bc_4x4.md), [data](reports/data/exp05_bc_4x4_eval/summary.json) |

Model-free PPO with a reverse curriculum completes the smallest 4D board but not the 81- or
256-cell boards ([exp04](reports/experiments/exp04_ppo_4x4.md) analyses why: schedules tied to
the step budget instead of curriculum progress, no transfer from cycle-shaped starts). Cloning
the same network on the scripted Hamiltonian follower's decisions (`snake4d imitate`, about twelve
seconds)
yields a neural policy that completes the full 4^4 board every time; PPO then only has to make it
faster. Throughput on this laptop: 33k PPO steps/s (4096 batched envs on CUDA,
minibatch 16384), see [exp00](reports/experiments/exp00_benchmark.md) and
[benchmark.json](reports/data/exp00_benchmark/benchmark.json).

## Repository layout

`src/snake4d/` package (see the graph above), `tests/` (pytest; markers `slow`, `gpu`),
`experiments/` (pre-registered `.env` arms), `docs/`, `reports/`, `runs/` (git-ignored run
artifacts). Working conventions for contributors and agents: [AGENTS.md](AGENTS.md). Licence: MIT;
credits in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
