# Game rules, observation, reward and masks

## Board and moves

- The board is an `ndim`-dimensional cube with `size` cells per axis, `C = size**ndim` cells
  (default 4^4 = 256). Cells are addressed by flat index; `grid.neighbour_table` maps
  `(cell, action)` to the neighbouring cell or `-1` at a wall.
- There are `2*ndim` actions: action `a` moves along axis `a >> 1` by `+1` if `a` is even, else
  `-1` (the direction scheme of Pella86/Snake4d). No direction state is kept: the snake moves in
  whatever direction the chosen action says, every step.
- The snake starts with length 1 at a random cell; one food item is always present, drawn
  uniformly over free cells (masked random scores + argmax, never a rejection loop).

## One step (post-tail-move occupancy)

1. Target cell = neighbour of the head along the action; a wall (`-1`) is a death.
2. If the target holds the food the snake eats: the tail does **not** move this step.
3. Otherwise every body cell's age is decremented (the tail cell becomes empty).
4. If the target cell is still occupied after step 3 the snake dies (so following the tail is
   legal, as in classic snake).
5. The head is written with age `length` (+1 when eating).
6. If `length == C` the game is won - checked **before** spawning food, so no spawn is attempted
   on a full board.
7. Otherwise, after eating, new food is spawned on a free cell.

## Episode end

| event | terminated | truncated | reward |
|---|---|---|---|
| wall or body collision | yes | no | `r_death = -1` |
| board filled (`length == C`) | yes | no | `r_food + r_win = 1 + 10` |
| more than `idle_mult * C` consecutive steps without eating (`idle > idle_cap`; default 4C) | no | yes | `r_step` |
| `C*C` steps in one episode | no | yes | `r_step` |

Truncations are not deaths: SB3 bootstraps the value of the last state for them
(`TimeLimit.truncated`), following Gymnasium's terminated/truncated semantics
(https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Every step also
pays `r_step = -0.001` (except the winning step), which makes wandering forever worse than
finishing; `Config` checks `|r_step| / (1 - gamma) < |r_death|` so dying never beats living.

Optional potential-based shaping (`SNAKE_SHAPING_COEF > 0`) adds
`coef * (gamma * Phi(s') - Phi(s))` with `Phi = -L1(head, food) / (ndim * size)` and
`Phi(terminal) = 0` (the episodic-task condition of Grzes 2017,
https://dl.acm.org/doi/10.5555/3091125.3091208); by Ng, Harada and Russell (1999,
https://dl.acm.org/doi/10.5555/645528.657613) this cannot change the optimal policy. It is off
by default.

## Observation (`4C + 2` floats in [0, 1])

| block | size | content |
|---|---|---|
| body | C | 1 where a body cell (including the head) is |
| time-to-vacate | C | `age / length`: 1 at the head, `1/length` at the tail, 0 elsewhere |
| head | C | one-hot head position |
| food | C | one-hot food position |
| scalars | 2 | `length / C`, `min(idle / idle_cap, 1)` |

The time-to-vacate channel tells the agent when each body cell will free up, the information the
reference 2D agent (linyiLYi/snake-ai) encodes as a brightness gradient along the body.

## Action mask

An action is legal when the target is inside the board, its cell is free once the tail has moved
(`age <= 1`), and it is not the neck (for length 2 the tail is the neck, so the immediate
reversal is forbidden like in classic snake; for length >= 3 the neck is already occupied). If no
action is legal the mask falls back to every in-bounds move: MaskablePPO masks with `-1e8`, not
`-inf` (`HUGE_NEG` in sb3-contrib's `MaskableCategorical`,
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/maskable/distributions.py),
so an all-false row would otherwise sample near-uniformly (and possibly a wall).

## Parity and completability

The grid graph is bipartite (cells coloured by coordinate-sum parity;
https://mathworld.wolfram.com/GridGraph.html). A Hamiltonian cycle alternates colours, so it
needs equal colour classes, hence an even number of cells:

- `4^4 = 256` splits 128/128 and has a Hamiltonian cycle (every rectangular grid graph with an
  even number of cells has one: Itai, Papadimitriou and Szwarcfiter 1982,
  https://doi.org/10.1137/0211056; `hamilton.ham_cycle` builds the 2D cycle and lifts it
  dimension by dimension, `check_route` verifies the result). Following it fills the board
  every time (`RoutePolicy`), which is the completability proof and the step-count ceiling.
- `3^4 = 81` splits 41/40: no Hamiltonian cycle exists, and a snake filling the board must end with
  its body as a Hamiltonian path whose ends both lie in the 41-cell class. The scripted baseline
  uses a Hamiltonian cycle over the 80 cells other than the corner `(0,0,0,0)`
  (`hamilton.cycle_minus_corner`: an explicit 2D construction, then the corner's column woven into
  a neighbouring column as a ladder in each extra dimension) and enters the corner only when the
  food is there and the detour cannot let the head overtake the tail (it skips at most as many
  cycle cells as are free).
