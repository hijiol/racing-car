# Optimal path solver + ghost line

## Context

The whole sim was built toward this: the car, the track geometry, and the move rules
are all in place, and the last piece is working out the fastest way round. Every
verification step so far has run a throwaway BFS over `(position, velocity,
checkpoint)` states in a scratch script — this makes that a real part of the game.

**Optimal** = fewest turns, free choice of start slot. The solver searches from all
5 start-line nodes and finishes anywhere across the finish line, so it reports the
genuine track record for the map rather than the best run from one grid slot.

**Display** = a ghost line the player toggles on and off. It draws the solved path
over the track; the player keeps driving their own line and can compare.

## The search

State is `(pos, velocity, checkpoint_passed)` — position alone is not enough, since
the same node arrived at different speeds has completely different futures.

- **Sources**: every node from `track.line_nodes("start")`, velocity `(0, 0)`, cost 0.
- **Moves**: exactly the game's rules. Reuse `Car.moves(track)` (car.py:37) rather
  than re-deriving them, so the solver can never drift from what the player may do.
- **Goal**: a hop where `checkpoint_passed and track.finished(origin, dest)`
  (track.py:340). Checkpoint progress advances via `track.hop_crosses(track.checkpoint, ...)`
  (track.py:333), mirroring `Track.lap_state` (track.py:344).
- **Cost**: 1 per turn.

Prior BFS runs cost ~1.5s over ~16k states — too slow to sit in a click handler, so
this uses **A\*** with an admissible heuristic:

> With Chebyshev distance `d` still to cover and current speed `s`, a car that
> accelerates flat out covers `s*t + t*(t+1)/2` in `t` turns. The heuristic is the
> smallest `t` satisfying `s*t + t*(t+1)/2 >= d`.

`d` is the Chebyshev distance to the nearest finish-line node, plus — when the
checkpoint is still ahead — the distance to the checkpoint first, since that leg is
mandatory. Both are lower bounds on real distance, so the heuristic never
overestimates and A\* still returns a genuine optimum.

## Files

**`solver.py`** (new)

```python
@dataclass
class Solution:
    path: list[Node]              # start node .. final node, one per turn
    velocities: list[tuple[int, int]]
    turns: int                    # == len(path) - 1
    explored: int                 # states popped, for the status line

def solve(track) -> Solution | None   # None when the map is unwinnable
```

A\* over a `heapq`, `came_from` dict for reconstruction, `best_cost` dict keyed by
state. Uses `Car(pos, velocity).moves(track)` for successors — `Track.allows` is
already memoised (track.py:293), which is what makes repeated expansion cheap.

**`main.py`**

- `PATH_COLOR` constant; ghost drawn as a polyline plus a small dot per turn node,
  under the car and option rings so it never hides live state.
- `draw_path_button()` beside the existing map button (mirrors `draw_map_button`,
  main.py:170), plus `P` as the keyboard toggle.
- `solutions: dict[int, Solution | None]` cached per map variant; cleared implicitly
  by keying on `variant`, recomputed on first toggle after a map change.
- Solving paints a "solving…" frame first, then runs. If A\* lands above ~0.3s,
  move it to a `threading.Thread` and draw the ghost when it arrives — the plan is
  to measure first and only add the thread if the pause is visible.
- Status line gains `optimal N turns` when a solution is loaded, and the ghost's own
  start node gets a faint marker, since it may differ from where the player parked.

Nothing in `track.py`, `car.py`, `grid.py` or `course.py` needs to change.

## Verification

1. **Correctness against ground truth** — headless script: run `solver.solve()` and
   the existing plain BFS on variants 0, 1, 2, 5; assert identical turn counts. A\*
   must match BFS exactly or the heuristic is inadmissible.
2. **The path is actually drivable** — replay `Solution.path` through a real `Car`
   with `drive_to()`; every hop must be accepted, and `track.lap_state(history)`
   must end at `"finished"`. This catches any drift between solver and game rules.
3. **Speed** — time `solve()` per variant; report against the ~1.5s BFS baseline.
4. **Visual** — render a frame with the ghost on, crop the start area, confirm the
   line follows the corridor and hugs the inside of bends.
5. **In the app** — `python main.py`, press `P` to toggle, cycle maps with `M` and
   confirm the ghost re-solves for the new circuit and toggles cleanly off.
