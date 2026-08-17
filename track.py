"""The circuit: where the car may drive, and the lines it runs between.

The track is ordinary polygon geometry in world coordinates and owes nothing to
the grid — it curves wherever it likes and the grid nodes simply fall inside or
outside it. Two rules follow from that:

  * a node is drivable only if it lies inside the track;
  * a move is legal only if the straight line the car travels stays inside too.

That second rule matters because a fast car jumps several cells in one turn and
could otherwise cut a corner straight through a wall.

A Line is a run of grid nodes between two ends. The start and finish lines are
kept horizontal or vertical by course.py, so every cell along them is a node the
car can line up on; the checkpoint, which is only ever crossed, sits at a true
right angle to the track instead.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache

from grid import Grid, Node

Point = tuple[float, float]


def point_in_polygon(point: Point, polygon) -> bool:
    """Ray casting: count edge crossings to the right of the point."""
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i - 1]
        x2, y2 = polygon[i]
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) / (y2 - y1) * (x2 - x1)
            if crossing_x > x:
                inside = not inside
    return inside


def cross(a: Point, b: Point, c: Point) -> float:
    """Which side of a->b does c fall on? Positive one way, negative the other.

    Twice the signed area of the triangle, which makes it a turn test, a
    left/right test and an area formula all at once.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_cross(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 touches or crosses segment p3-p4.

    Touching counts, deliberately: a move that grazes a wall is refused rather
    than allowed through on a rounding error.
    """
    d1, d2 = cross(p3, p4, p1), cross(p3, p4, p2)
    d3, d4 = cross(p1, p2, p3), cross(p1, p2, p4)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear-and-overlapping cases show up as a zero orientation on-segment.
    for d, (a, b, c) in (
        (d1, (p3, p4, p1)),
        (d2, (p3, p4, p2)),
        (d3, (p1, p2, p3)),
        (d4, (p1, p2, p4)),
    ):
        if d == 0 and min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]):
            return True
    return False


@dataclass(frozen=True)
class Boundary:
    """Tarmac: inside `outer`, but outside every hole (an infield, say)."""

    outer: tuple[Point, ...]
    holes: tuple[tuple[Point, ...], ...] = ()
    walls: tuple = field(default=(), repr=False, compare=False)  # filled in below

    def __post_init__(self) -> None:
        # Every wall segment, outer and holes alike, built once up front: this
        # list is walked for every legality check in the game.
        walls = tuple((ring[i - 1], ring[i]) for ring in (self.outer, *self.holes) for i in range(len(ring)))
        object.__setattr__(self, "walls", walls)

    def contains_point(self, point: Point) -> bool:
        if not point_in_polygon(point, self.outer):
            return False
        return not any(point_in_polygon(point, hole) for hole in self.holes)

    def blocks(self, p: Point, q: Point) -> bool:
        """True if travelling p -> q would clip a wall.

        A circuit has hundreds of wall segments and almost none of them are
        anywhere near any given hop, so bounding boxes throw out the vast
        majority before the real intersection test runs.
        """
        lo_x, hi_x = (p[0], q[0]) if p[0] <= q[0] else (q[0], p[0])
        lo_y, hi_y = (p[1], q[1]) if p[1] <= q[1] else (q[1], p[1])
        for a, b in self.walls:
            if max(a[0], b[0]) < lo_x or min(a[0], b[0]) > hi_x:
                continue
            if max(a[1], b[1]) < lo_y or min(a[1], b[1]) > hi_y:
                continue
            if segments_cross(p, q, a, b):
                return True
        return False


def normal_at(centerline, i: int) -> Point:
    """Unit vector square across the centreline at point i.

    Averaging the incoming and outgoing direction keeps the width honest around
    bends. Circuits are closed loops, so the ends wrap.
    """
    n = len(centerline)
    prev = centerline[(i - 1) % n]
    nxt = centerline[(i + 1) % n]
    dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
    length = math.hypot(dx, dy) or 1.0
    return -dy / length, dx / length


def loop_corridor(centerline, half_width: float) -> tuple[tuple[Point, ...], tuple[Point, ...]]:
    """A closed circuit: (outer ring, infield ring).

    The centreline wraps, so the two offsets close into rings rather than
    joining at the ends. The infield becomes a hole in the boundary — drive
    around it, not across it.
    """
    left, right = [], []
    for i, (x, y) in enumerate(centerline):
        nx, ny = normal_at(centerline, i)
        left.append((x + nx * half_width, y + ny * half_width))
        right.append((x - nx * half_width, y - ny * half_width))
    # Which offset ends up outside depends on which way the centreline winds,
    # so pick by area rather than trusting the normal's sign.
    if abs(polygon_area(left)) < abs(polygon_area(right)):
        left, right = right, left
    return tuple(left), tuple(right)


def ray_hit(origin: Point, direction: Point, walls) -> float:
    """Distance from origin along direction to the nearest wall, or inf."""
    ox, oy = origin
    dx, dy = direction
    nearest = math.inf
    for (ax, ay), (bx, by) in walls:
        ex, ey = bx - ax, by - ay
        denominator = dx * ey - dy * ex
        if denominator == 0:
            continue  # parallel to this wall
        t = ((ax - ox) * ey - (ay - oy) * ex) / denominator
        u = ((ax - ox) * dy - (ay - oy) * dx) / denominator
        if t > 0 and 0 <= u <= 1:
            nearest = min(nearest, t)
    return nearest


def is_simple(polygon) -> bool:
    """True if no two non-adjacent edges of the ring cross.

    Offsetting a curve inevitably folds it over itself where the bend is tighter
    than the offset distance, which turns a track into a knot.
    """
    n = len(polygon)
    edges = [(polygon[i - 1], polygon[i]) for i in range(n)]
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # these two share a vertex
            if segments_cross(*edges[i], *edges[j]):
                return False
    return True


def polygon_area(polygon) -> float:
    """Shoelace area; the sign carries the winding direction."""
    total = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i - 1]
        x2, y2 = polygon[i]
        total += x1 * y2 - x2 * y1
    return total / 2


@lru_cache(maxsize=None)
def bresenham(a: Node, b: Node) -> tuple[Node, ...]:
    """Grid nodes along a line in any direction, from a to b inclusive.

    Cached: the same handful of lines are walked over and over during a search.
    """
    (c1, r1), (c2, r2) = a, b
    dc, dr = abs(c2 - c1), abs(r2 - r1)
    step_c = 1 if c2 >= c1 else -1
    step_r = 1 if r2 >= r1 else -1
    error = dc - dr
    nodes = []
    while True:
        nodes.append((c1, r1))
        if (c1, r1) == (c2, r2):
            return tuple(nodes)
        doubled = 2 * error
        if doubled > -dr:
            error -= dr
            c1 += step_c
        if doubled < dc:
            error += dc
            r1 += step_r


@dataclass(frozen=True)
class Line:
    """A start or finish line: two ends, and the nodes strung between them.

    Horizontal and vertical lines pass through a node at every cell along their
    length, which is what makes every point on them selectable. A diagonal one
    only meets a node where Bresenham puts it, so start and finish lines are
    kept axis-aligned (see `axis_aligned`); the checkpoint, which is only ever
    crossed and never picked, is free to sit at a true right angle to the track.
    """

    a: Node
    b: Node

    @property
    def axis_aligned(self) -> bool:
        return self.a[0] == self.b[0] or self.a[1] == self.b[1]

    def nodes(self) -> tuple[Node, ...]:
        return bresenham(self.a, self.b)

    def direction(self, grid: Grid) -> tuple[float, float]:
        """Unit vector from end a towards end b, in world units."""
        ax, ay = grid.world_pos(*self.a)
        bx, by = grid.world_pos(*self.b)
        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy) or 1.0
        return dx / length, dy / length


@dataclass
class Track:
    """Grid + boundary + the two lines, and the node chosen on each."""

    grid: Grid
    boundary: Boundary
    start_line: Line
    finish_line: Line
    checkpoints: tuple[Line, ...] = ()  # gates that must be crossed in order
    start: Node = field(default=None)
    finish: Node = field(default=None)

    def __post_init__(self) -> None:
        self._open = {node for node in self.grid.nodes() if self.boundary.contains_point(self.grid.world_pos(*node))}
        self._allowed: dict[tuple[Node, Node], bool] = {}
        self._spans: dict[Line, tuple[Point, Point]] = {}
        for which in ("start", "finish"):
            options = self.line_nodes(which)
            if not options:
                raise ValueError(f"the {which} line has no nodes inside the track")
            chosen = getattr(self, which)
            self.choose(which, chosen if chosen in options else options[len(options) // 2])

    def is_open(self, node: Node) -> bool:
        return node in self._open

    def line_for(self, which: str) -> Line:
        return self.start_line if which == "start" else self.finish_line

    def line_nodes(self, which: str) -> list[Node]:
        """Every node on the line that is on the track — all of them selectable.

        No thinning: the car moves on the grid, so the places it can line up on
        are exactly the grid nodes the line passes through. Keeping the count
        sensible is the corridor's job, not the picker's.
        """
        return [node for node in self.line_for(which).nodes() if self.is_open(node)]

    def allows(self, origin: Node, destination: Node) -> bool:
        """May the car travel this straight hop? Both ends on track, no wall clipped.

        Answers are cached: a search revisits the same hop from thousands of
        states, and the geometry behind the answer never changes.
        """
        key = (origin, destination)
        answer = self._allowed.get(key)
        if answer is None:
            answer = self.is_open(origin) and self.is_open(destination) and not self.boundary.blocks(
                self.grid.world_pos(*origin), self.grid.world_pos(*destination)
            )
            self._allowed[key] = answer
        return answer

    def line_span(self, line: Line) -> tuple[Point, Point]:
        """Where the line meets the walls: wall to wall, not node to node.

        The end nodes stop a fraction short of the tarmac's edge, which both
        looks untidy and leaves a sliver a fast car could slip through without
        registering. Casting a ray each way from the middle finds the real edge.
        """
        cached = self._spans.get(line)
        if cached is None:
            grid = self.grid
            middle_node = line.nodes()[len(line.nodes()) // 2]
            middle = grid.world_pos(*middle_node)
            dx, dy = line.direction(grid)
            ends = []
            for sign in (1, -1):
                reach = ray_hit(middle, (dx * sign, dy * sign), self.boundary.walls)
                if reach == math.inf:  # no wall that way; fall back to the end node
                    end = grid.world_pos(*(line.b if sign > 0 else line.a))
                else:
                    end = middle[0] + dx * sign * reach, middle[1] + dy * sign * reach
                ends.append(end)
            cached = (ends[0], ends[1])
            self._spans[line] = cached
        return cached

    def hop_crosses(self, line: Line, origin: Node, destination: Node) -> bool:
        """True if travelling origin -> destination cuts or lands on `line`."""
        if destination in line.nodes():
            return True
        span = self.line_span(line)
        return segments_cross(self.grid.world_pos(*origin), self.grid.world_pos(*destination), *span)

    def finished(self, origin: Node, destination: Node) -> bool:
        """True if this single hop reaches the finish line."""
        return self.hop_crosses(self.finish_line, origin, destination)

    def gates_passed(self, origin: Node, destination: Node, passed: int) -> int:
        """How many gates are behind the car after this hop.

        Gates only count in order, and a single fast hop can clear more than one
        where they are close together, so this keeps advancing while it can.
        """
        while passed < len(self.checkpoints) and self.hop_crosses(self.checkpoints[passed], origin, destination):
            passed += 1
        return passed

    def lap_progress(self, history: list[Node]) -> tuple[int, bool]:
        """Gates cleared so far, and whether the lap is done.

        A single gate on the far side is not enough to prove a lap: with the
        finish line sitting just behind the start, a car could run out to that
        gate, turn round and come back to the finish having driven half the
        circuit twice. Gates spread round the lap and counted in order can only
        be cleared by going round the way the track runs.
        """
        passed = 0
        for origin, destination in zip(history, history[1:]):
            if passed == len(self.checkpoints) and self.finished(origin, destination):
                return passed, True
            passed = self.gates_passed(origin, destination, passed)
        return passed, False

    def choose(self, which: str, node: Node) -> None:
        """Pick where the car sets off from / aims for. Must be on that line."""
        if node not in self.line_nodes(which):
            raise ValueError(f"{node} is not a track node on the {which} line")
        other = self.finish if which == "start" else self.start
        if other is not None and node == other:
            raise ValueError(f"start and finish cannot both be {node}")
        setattr(self, which, node)
