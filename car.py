"""The car and how it is allowed to move.

The car carries a velocity, not just a position. Each turn the velocity may
change by at most one step in each axis, then the car moves by that velocity:

    velocity += (dcol, drow)   with dcol, drow in {-1, 0, +1}   -> 9 choices
    position += velocity

Standing still at the start (velocity zero) that yields the nine nodes around
the car, so the first move is one tile in any direction. Once it is rolling the
nine choices sit around wherever the current velocity would carry it, so speed
has to be built up and shed gradually — the car cannot stop on a coin.
"""

from dataclasses import dataclass, field

from grid import Node

# The nine ways velocity may change in a turn, including leaving it alone.
STEERING = [(dcol, drow) for dcol in (-1, 0, 1) for drow in (-1, 0, 1)]


@dataclass
class Car:
    pos: Node
    velocity: tuple[int, int] = (0, 0)
    history: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.history:
            self.history = [self.pos]

    def coasting_to(self) -> Node:
        """Where the car ends up if it does not steer at all this turn."""
        return self.pos[0] + self.velocity[0], self.pos[1] + self.velocity[1]

    def moves(self, track) -> dict[Node, tuple[int, int]]:
        """Legal next nodes -> the velocity that gets there.

        These are the nine nodes around the coasting node, minus any the track
        refuses — off the tarmac, or reached by a line that clips a wall. A dict
        because two steering choices never share a destination, and the planner
        wants the velocity the car arrives with.

        An empty result means the car has crashed: too fast into a corner, with
        nowhere legal left to put itself.
        """
        options = {}
        for dcol, drow in STEERING:
            vel = self.velocity[0] + dcol, self.velocity[1] + drow
            node = self.pos[0] + vel[0], self.pos[1] + vel[1]
            if track.grid.contains(*node) and track.allows(self.pos, node):
                options[node] = vel
        return options

    def drive_to(self, node: Node, track) -> None:
        """Take one turn, ending on `node`. Raises if that is not a legal move."""
        options = self.moves(track)
        if node not in options:
            raise ValueError(f"{node} is not reachable from {self.pos} at velocity {self.velocity}")
        self.velocity = options[node]
        self.pos = node
        self.history.append(node)

    def undo(self) -> None:
        """Step back one turn, restoring the velocity the car had arrived with."""
        if len(self.history) < 2:
            return
        self.history.pop()
        self.pos = self.history[-1]
        previous = self.history[-2] if len(self.history) > 1 else self.pos
        self.velocity = self.pos[0] - previous[0], self.pos[1] - previous[1]

    def reset(self, pos: Node) -> None:
        self.pos = pos
        self.velocity = (0, 0)
        self.history = [pos]

    @property
    def speed(self) -> float:
        return (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
