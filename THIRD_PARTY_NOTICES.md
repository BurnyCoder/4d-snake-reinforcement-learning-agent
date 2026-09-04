# Third-party notices

This project is MIT licensed (see LICENSE). No source code was copied from other repositories;
the following projects informed the design and are credited here:

| project | licence | what was used |
|---|---|---|
| [Pella86/Snake4d](https://github.com/Pella86/Snake4d) | MIT | the 2*ndim action scheme (axis = action // 2, sign by parity) and the WASD + IJKL key layout |
| [twanvl/snake](https://github.com/twanvl/snake) | MIT | the Hamiltonian-cycle follower as the perfect-play baseline and the "never let the head overtake the tail" rule for shortcuts/detours |
| [PufferAI/PufferLib](https://github.com/PufferAI/PufferLib) (`ocean/snake`) | MIT | the body-age grid representation (tail = 1, head = length, decrement to move the tail) re-expressed in numpy |
| [linyiLYi/snake-ai](https://github.com/linyiLYi/snake-ai) | Apache-2.0 | ideas only: MaskablePPO for board-filling snake, the body-age observation channel, decaying learning rate and clip range, the `4 * cells` starvation cap |
| [instadeepai/jumanji](https://github.com/instadeepai/jumanji) (snake env) | Apache-2.0 | ideas only: per-entity observation channels, uniform food over free cells, `+1` per food reward |
| [Checkmate6659/4d-minesweeper](https://github.com/Checkmate6659/4d-minesweeper) | no licence | idea only (not code): rendering a 4D grid as a 2D grid of 2D layers |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3), [Stable-Baselines-Team/stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) | MIT | library dependencies (PPO, MaskablePPO, VecEnv, VecMonitor, callbacks, logger) |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | MIT | library dependency (Env API, env checker) |
| [vb64/markdown-pdf](https://github.com/vb64/markdown-pdf) (uses PyMuPDF) | AGPL-3.0 | documentation tooling only (`docs` dependency group), used to build `reports/paper.pdf`; not a runtime dependency |

Papers: Huang & Ontanon 2020 (invalid action masking), Ng, Harada & Russell 1999 (potential-based
shaping), Resnick et al. 2018 (Backplay), Salimans & Chen 2018 (success-gated reverse curriculum),
Ruskey & Sawada 2003 (bent Hamilton cycles in d-dimensional grids), Du et al. 2022 (AlphaSnake).
