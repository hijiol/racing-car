"""Finding the fastest way round: fewest turns, any start slot.

The search is over states of (position, velocity, gates passed), because a node
reached at different speeds has completely different futures — arriving at a
hairpin flat out is not the same place as crawling into it — and because a lap
is only a lap if the gates round the circuit went by in order. Every start-line
slot is seeded at once and the goal is any hop that reaches the finish line
with every gate behind it, so the answer is the best run the map allows rather
than the best from one slot.

A plain breadth-first sweep gets the same answer but visits everything; A* with
the heuristic below only walks towards the finish, which is the difference
between a visible pause and an instant one.
"""

import heapq
from dataclasses import dataclass

from car import Car
from grid import Node

Velocity = tuple[int, int]
State = tuple[Node, Velocity, int]  # where, how fast, and how many gates behind it


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
    """How far the car still has to travel, gate by gate. Cached per node.

    The route is forced: every remaining gate in order, then the finish line. So
    the distance left is at least the hop to the next gate plus the chain of
    gate-to-gate legs after it — a lower bound, which is what A* needs, and a
    far tighter one than the straight line to the finish.
    """

    def __init__(self, track):
        self.gates = [
            [node for node in gate.nodes() if track.is_open(node)] or list(gate.nodes())
            for gate in track.checkpoints
        ]
        self.gates.append(track.line_nodes())  # the last leg ends back at the line
        # Distance from each gate to the one after it, summed backwards, so
        # `self.chain[i]` is the whole route left once gate i has been reached.
        self.chain = [0] * (len(self.gates) + 1)
        for i in range(len(self.gates) - 2, -1, -1):
            leg = min(chebyshev(a, b) for a in self.gates[i] for b in self.gates[i + 1])
            self.chain[i] = leg + self.chain[i + 1]
        self._to_gate: list[dict[Node, int]] = [{} for _ in self.gates]

    def remaining(self, node: Node, gate_index: int) -> int:
        cached = self._to_gate[gate_index]
        value = cached.get(node)
        if value is None:
            value = min(chebyshev(node, target) for target in self.gates[gate_index])
            cached[node] = value
        return value + self.chain[gate_index]


def solve(track) -> Solution | None:
    """The fewest-turn run from any start slot. None if the map cannot be won."""
    distances = _Distances(track)
    gate_count = len(track.checkpoints)

    def heuristic(node: Node, velocity: Velocity, passed: int) -> int:
        distance = distances.remaining(node, passed)
        if distance == 0:
            return 0
        # Speed along the axis that matters, never more than the car really has.
        speed = max(abs(velocity[0]), abs(velocity[1]))
        return turns_to_cover(distance, speed)

    queue: list[tuple[int, int, State]] = []
    best_cost: dict[State, int] = {}
    came_from: dict[State, State | None] = {}

    for node in track.line_nodes():
        state = (node, (0, 0), 0)
        best_cost[state] = 0
        came_from[state] = None
        heapq.heappush(queue, (heuristic(node, (0, 0), 0), 0, state))

    while queue:
        _, cost, state = heapq.heappop(queue)
        if cost > best_cost.get(state, float("inf")):
            continue  # a cheaper route to this state was queued later

        node, velocity, passed = state
        for destination, next_velocity in Car(node, velocity).moves(track).items():
            if passed == gate_count and track.finished(node, destination):
                final = (destination, next_velocity, passed)
                came_from[final] = state
                return _reconstruct(final, came_from)

            next_passed = track.gates_passed(node, destination, passed)
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
