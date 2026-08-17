"""Finding the fastest way round: fewest turns, any start slot.

The search is over states of (position, velocity, checkpoint passed), because a
node reached at different speeds has completely different futures — arriving at
a hairpin flat out is not the same place as crawling into it. Every start-line
slot is seeded at once and the goal is any hop that reaches the finish line, so
the answer is the best run the map allows rather than the best from one slot.

A plain breadth-first sweep gets the same answer but visits everything; A* with
the heuristic below only walks towards the finish, which is the difference
between a visible pause and an instant one.
"""

import heapq
from dataclasses import dataclass

from car import Car
from grid import Node

Velocity = tuple[int, int]
State = tuple[Node, Velocity, bool]


@dataclass
class Solution:
    """One optimal run: where the car goes, and how fast it is going there."""

    path: list[Node]  # start node .. finishing node, one entry per turn
    velocities: list[Velocity]  # velocity the car arrives at each node with

    @property
    def turns(self) -> int:
        return len(self.path) - 1


def turns_to_cover(distance: int, speed: int) -> int:
    """Fewest turns to cover `distance` when already moving at `speed`.

    Flat out, a car adds one to its speed every turn, so in t turns it covers at
    most speed*t + t*(t+1)/2. The smallest t that reaches the distance is a
    lower bound on the turns really needed, which is what makes it a safe (never
    overestimating) heuristic for A*.
    """
    covered = 0
    turns = 0
    while covered < distance:
        turns += 1
        speed += 1
        covered += speed
    return turns


def chebyshev(a: Node, b: Node) -> int:
    """Moves needed to bridge two nodes if speed were no object.

    Chebyshev, not Euclidean: one diagonal step closes a column and a row at
    once, so the diagonal is as cheap as the straight.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


class _Distances:
    """Distances to the things the car must still reach. Cached per node."""

    def __init__(self, track):
        # Track guarantees the finish line has on-track nodes; the checkpoint is
        # only ever crossed, so fall back to its full span if none are on track.
        self.finish_nodes = track.line_nodes("finish")
        self.checkpoint_nodes = []
        if track.checkpoint is not None:
            on_track = [node for node in track.checkpoint.nodes() if track.is_open(node)]
            self.checkpoint_nodes = on_track or list(track.checkpoint.nodes())
        self._to_finish: dict[Node, int] = {}
        self._to_checkpoint: dict[Node, int] = {}
        # Both legs are mandatory, so the shortest way through the checkpoint is
        # a lower bound on the distance left when it has not been passed yet.
        self._checkpoint_to_finish = min(
            (chebyshev(cp, fin) for cp in self.checkpoint_nodes for fin in self.finish_nodes),
            default=0,
        )

    def to_finish(self, node: Node) -> int:
        value = self._to_finish.get(node)
        if value is None:
            value = min(chebyshev(node, target) for target in self.finish_nodes)
            self._to_finish[node] = value
        return value

    def remaining(self, node: Node, checkpoint_passed: bool) -> int:
        if checkpoint_passed or not self.checkpoint_nodes:
            return self.to_finish(node)
        value = self._to_checkpoint.get(node)
        if value is None:
            value = min(chebyshev(node, target) for target in self.checkpoint_nodes)
            self._to_checkpoint[node] = value
        return value + self._checkpoint_to_finish


def solve(track) -> Solution | None:
    """The fewest-turn run from any start slot. None if the map cannot be won."""
    distances = _Distances(track)
    checkpoint = track.checkpoint

    def heuristic(node: Node, velocity: Velocity, passed: bool) -> int:
        distance = distances.remaining(node, passed)
        if distance == 0:
            return 0
        # Speed along the axis that matters, never more than the car really has.
        speed = max(abs(velocity[0]), abs(velocity[1]))
        return turns_to_cover(distance, speed)

    queue: list[tuple[int, int, State]] = []
    best_cost: dict[State, int] = {}
    came_from: dict[State, State | None] = {}

    for node in track.line_nodes("start"):
        state = (node, (0, 0), False)
        best_cost[state] = 0
        came_from[state] = None
        heapq.heappush(queue, (heuristic(node, (0, 0), False), 0, state))

    while queue:
        _, cost, state = heapq.heappop(queue)
        if cost > best_cost.get(state, float("inf")):
            continue  # a cheaper route to this state was queued later

        node, velocity, passed = state
        for destination, next_velocity in Car(node, velocity).moves(track).items():
            next_passed = passed or (
                checkpoint is not None and track.hop_crosses(checkpoint, node, destination)
            )
            if passed and track.finished(node, destination):
                final = (destination, next_velocity, True)
                came_from[final] = state
                return _reconstruct(final, came_from)

            next_state = (destination, next_velocity, next_passed)
            next_cost = cost + 1
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                came_from[next_state] = state
                priority = next_cost + heuristic(destination, next_velocity, next_passed)
                heapq.heappush(queue, (priority, next_cost, next_state))

    return None


def _reconstruct(final: State, came_from: dict[State, State | None]) -> Solution:
    path, velocities = [], []
    state = final
    while state is not None:
        node, velocity, _ = state
        path.append(node)
        velocities.append(velocity)
        state = came_from[state]
    path.reverse()
    velocities.reverse()
    return Solution(path=path, velocities=velocities)
