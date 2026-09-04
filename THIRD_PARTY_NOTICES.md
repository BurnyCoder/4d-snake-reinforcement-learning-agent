# Third-party notices

This project is MIT licensed (see LICENSE). No source code was copied from other repositories;
the following projects informed the design and are credited here:

| project | licence | what was used |
|---|---|---|
| [Pella86/Snake4d](https://github.com/Pella86/Snake4d) | MIT | the 2*ndim action scheme (axis = action // 2, sign by parity, [src/snake.py](https://github.com/Pella86/Snake4d/blob/master/src/snake.py)) and the WASD + IJKL key layout ([main.py](https://github.com/Pella86/Snake4d/blob/master/main.py)) |
| [twanvl/snake](https://github.com/twanvl/snake) | MIT | the Hamiltonian-cycle follower as the perfect-play baseline (its README benchmarks fixed-cycle and perturbed-cycle agents at 0 % losses on 30x30) |
| [johnflux, "Nokia 6110 Part 3 - Algorithms"](https://johnflux.com/2015/05/02/nokia-6110-part-3-algorithms/) | blog post, idea only | the perturbed-Hamiltonian-cycle rule "the head must never overtake the tail", applied to the corner detour of `agents.RoutePolicy` |
| [instadeepai/jumanji](https://github.com/instadeepai/jumanji) (snake env, [env.py](https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/snake/env.py)) | Apache-2.0 | ideas only: the body-age grid (`body_state` / `norm_body_state`: tail lowest, head highest, tail moves by decrement), per-entity observation channels, uniform food over free cells, `+1` per food reward |
| [oscarknagg/wurm](https://github.com/oscarknagg/wurm) | no licence | idea only (not code): its README describes the same body-age encoding |
| [linyiLYi/snake-ai](https://github.com/linyiLYi/snake-ai) | Apache-2.0 | ideas only: MaskablePPO for board-filling snake, the body-gradient observation channel, decaying learning rate and clip range, the `4 * cells` starvation cap |
| [Checkmate6659/4d-minesweeper](https://github.com/Checkmate6659/4d-minesweeper) | no licence | idea only (not code): rendering a 4D grid as a 2D grid of 2D layers |
| [eugeneko/Snake4D](https://github.com/eugeneko/Snake4D) | (see its repository) | cited only for its README statement that a projected 4D snake board is unplayable beyond a certain length |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [Stable-Baselines-Team/stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | MIT | library dependencies (PPO, MaskablePPO, VecEnv, VecMonitor, callbacks, logger) |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | MIT | library dependency (Env API, env checker) |
| [vb64/markdown-pdf](https://github.com/vb64/markdown-pdf) (uses PyMuPDF) | AGPL-3.0 | documentation tooling only (`docs` dependency group), used to build `reports/paper.pdf`; not a runtime dependency |

Papers (full references in [reports/paper.md](reports/paper.md)): Huang & Ontanon 2020 (invalid
action masking), Ng, Harada & Russell 1999 and Grzes 2017 (potential-based shaping, episodic
condition), Resnick et al. 2018 (Backplay), Salimans & Chen 2018 (success-gated reverse
curriculum), Florensa et al. 2017 (reverse curricula), Itai, Papadimitriou & Szwarcfiter 1982
(Hamiltonicity of grid graphs), Du et al. 2022 (AlphaSnake), Pomerleau 1988 and Ross, Gordon &
Bagnell 2011 (behaviour cloning, DAgger), Towers et al. 2024 (Gymnasium).
