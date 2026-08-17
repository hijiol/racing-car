"""The circuits: closed loops, one per map variant.

Circuits are grown the way a real one sprawls, not stamped from a formula:

  1. scatter random points and take their convex hull — a simple loop to start
  2. shove each edge's midpoint sideways, which bites concave bends into it
  3. push neighbours apart and open out the sharpest corners, so no two parts of
     the circuit end up closer than the track is wide
  4. round the whole thing off with Chaikin smoothing, then resample evenly

That gives hairpins, chicanes and long straights — and, importantly, sections
that double back past each other, which a radial curve can never do.

Fattening the finished curve gives two rings, the outer wall and the infield,
and the car has to go round the infield rather than across it. The finish line
sits just behind the start line, the way a real circuit's does, so the run is
one full lap; the checkpoint on the far side is what stops the car simply
reversing over the finish on turn one (see Track.lap_state).

Shapes that fold their own walls into a knot are rejected rather than shown.
"""

import math
import random

from grid import Grid
from track import Boundary, Line, Track, cross, is_simple, loop_corridor, normal_at

# Twice the node density of the original 40x28 grid, over the same world: the
# circuit keeps its size and shape, the lattice measuring it just got finer.
GRID = Grid(cols=79, rows=55, spacing=13.0)

# World units, not cells, so changing the grid resolution never resizes the track.
# Two cells of tarmac either side of the racing line puts about five nodes across
# the corridor, which is how many places there are to line up on the start line.
HALF_WIDTH = 2.1 * GRID.spacing  # corridor half-width
EDGE_MARGIN = 55.0  # clear grid left showing outside the wall, on every side
WAYPOINTS = 260
CHECKPOINT_AT = 0.5  # the far side of the circuit
LINE_SPACING = 6  # centreline points between the finish line and the start line
MIN_LINE_GAP = 2.5  # clear cells between the two lines, so they never merge

SEED_POINTS = 14  # scattered points the hull is taken from
# Two rounds of displacement, the second gentler: one round gives a rounded blob,
# two give a lap with real corners on it. A third tangles more shapes than it is
# worth, since every bad shape is a rejection.
DISPLACEMENTS = (0.40, 0.30)
SMOOTHING = 3  # Chaikin rounds; more means gentler bends
MIN_GAP = 2.8  # closest two parts of the circuit may come, in corridor half-widths
MIN_CORNER = math.radians(72)  # sharpest corner allowed before smoothing rounds it
MIN_RADIUS = 1.3  # tightest bend, in corridor half-widths: below this the infield folds
MIN_LINE_NODES = 3  # a start line narrower than this is barely a choice
MAX_LINE_NODES = 8  # wider than this and the line is running along the track, not across


def _convex_hull(points):
    """Andrew's monotone chain. A simple loop to start growing the circuit from."""
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def half(ordered):
        chain = []
        for point in ordered:
            while len(chain) >= 2 and cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain[:-1]

    return half(points) + half(reversed(points))


def _displace_midpoints(points, rng, magnitude: float):
    """Insert a midpoint on every edge, pushed sideways. This is where bends come from."""
    grown = []
    for i, point in enumerate(points):
        nxt = points[(i + 1) % len(points)]
        dx, dy = nxt[0] - point[0], nxt[1] - point[1]
        length = math.hypot(dx, dy) or 1.0
        offset = rng.uniform(-magnitude, magnitude) * length
        grown.append(point)
        grown.append(
            (
                (point[0] + nxt[0]) / 2 - dy / length * offset,
                (point[1] + nxt[1]) / 2 + dx / length * offset,
            )
        )
    return grown


def _push_apart(points, min_distance: float, rounds: int = 12):
    """Separate points that stray within a track's width of each other.

    Two stretches of circuit running closer than the corridor is wide would
    merge into one lump of tarmac when fattened, so they are eased apart first.
    """
    points = [list(p) for p in points]
    count = len(points)
    for _ in range(rounds):
        moved = False
        for i in range(count):
            for j in range(i + 2, count):
                if i == 0 and j == count - 1:
                    continue  # neighbours around the loop
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                distance = math.hypot(dx, dy)
                if 0 < distance < min_distance:
                    shift = (min_distance - distance) / distance / 2
                    points[i][0] -= dx * shift
                    points[i][1] -= dy * shift
                    points[j][0] += dx * shift
                    points[j][1] += dy * shift
                    moved = True
        if not moved:
            break
    return [tuple(p) for p in points]


def _open_corners(points, limit: float, rounds: int = 6):
    """Ease open any corner sharper than `limit`, so hairpins stay drivable."""
    points = [list(p) for p in points]
    count = len(points)
    for _ in range(rounds):
        for i in range(count):
            before = points[i - 1]
            here = points[i]
            after = points[(i + 1) % count]
            ax, ay = before[0] - here[0], before[1] - here[1]
            bx, by = after[0] - here[0], after[1] - here[1]
            la, lb = math.hypot(ax, ay) or 1.0, math.hypot(bx, by) or 1.0
            angle = math.acos(max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb))))
            if angle >= limit:
                continue
            # Swing the following point outwards around this corner until it opens up.
            turn = (limit - angle) * (1 if cross(before, here, after) > 0 else -1)
            cos_t, sin_t = math.cos(turn), math.sin(turn)
            points[(i + 1) % count] = [
                here[0] + bx * cos_t - by * sin_t,
                here[1] + bx * sin_t + by * cos_t,
            ]
    return [tuple(p) for p in points]


def _chaikin(points, rounds: int):
    """Corner cutting: each round replaces every corner with two gentler ones."""
    for _ in range(rounds):
        smoothed = []
        for i, point in enumerate(points):
            nxt = points[(i + 1) % len(points)]
            smoothed.append((point[0] * 0.75 + nxt[0] * 0.25, point[1] * 0.75 + nxt[1] * 0.25))
            smoothed.append((point[0] * 0.25 + nxt[0] * 0.75, point[1] * 0.25 + nxt[1] * 0.75))
        points = smoothed
    return points


def _resample(points, count: int):
    """Space `count` points evenly along the closed curve, by arc length."""
    closed = list(points) + [points[0]]
    lengths = [0.0]
    for a, b in zip(closed, closed[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1]
    step = total / count

    resampled = []
    j = 0
    for i in range(count):
        target = i * step
        while j < len(lengths) - 2 and lengths[j + 1] < target:
            j += 1
        span = lengths[j + 1] - lengths[j] or 1.0
        t = (target - lengths[j]) / span
        a, b = closed[j], closed[j + 1]
        resampled.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return resampled


def _circumradius(a, b, c) -> float:
    """Radius of the circle through three points — the local bend radius."""
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
    area = abs(cross(a, b, c)) / 2
    if area < 1e-9:
        return math.inf  # straight ahead
    return ab * bc * ca / (4 * area)


def _relax_curvature(points, min_radius: float, rounds: int = 60):
    """Ease any bend tighter than the corridor can survive.

    Offsetting a curve inwards by more than its bend radius turns the infield
    inside out, which is what rejects most raw shapes. Averaging the offending
    point with its neighbours opens the bend out; a bend that is already wide
    enough is left exactly as it is, so straights stay straight.
    """
    points = list(points)
    count = len(points)
    for _ in range(rounds):
        tightest = math.inf
        relaxed = list(points)
        for i in range(count):
            before, here, after = points[i - 1], points[i], points[(i + 1) % count]
            radius = _circumradius(before, here, after)
            tightest = min(tightest, radius)
            if radius < min_radius:
                relaxed[i] = (
                    (before[0] + 2 * here[0] + after[0]) / 4,
                    (before[1] + 2 * here[1] + after[1]) / 4,
                )
        points = relaxed
        if tightest >= min_radius:
            break
    return points


def separation(points, half_width: float) -> float:
    """Closest approach between parts of the circuit that are not neighbours.

    Measured in half-widths, so anything under 2 means the tarmac overlaps itself.
    """
    count = len(points)
    skip = max(4, count // 12)  # ignore the curve's own immediate neighbourhood
    closest = math.inf
    for i in range(count):
        for j in range(i + skip, count):
            # The loop wraps, so points either side of the seam are neighbours too.
            if count - (j - i) < skip:
                continue
            closest = min(closest, math.dist(points[i], points[j]))
    return closest / half_width


def centerline(grid: Grid, variant: int = 0):
    """Grow one circuit, sized so the finished track sits inside the grid.

    Sizing the centreline is not enough to place the wall: the wall is offset
    perpendicular to the curve, which around a bend reaches further than the
    radius plus the corridor width suggests. So the shape is measured after it
    has been fattened, then scaled and recentred until the actual wall clears
    the grid edge by EDGE_MARGIN. A couple of passes is all it takes.
    """
    mid_x = (grid.cols - 1) / 2 * grid.spacing
    mid_y = (grid.rows - 1) / 2 * grid.spacing
    width, height = grid.world_size

    rng = random.Random(variant * 7919 + 13)
    scattered = [
        (rng.uniform(0.08, 0.92) * width, rng.uniform(0.08, 0.92) * height) for _ in range(SEED_POINTS)
    ]
    points = _convex_hull(scattered)
    for magnitude in DISPLACEMENTS:
        points = _displace_midpoints(points, rng, magnitude)
        points = _push_apart(points, MIN_GAP * HALF_WIDTH)
        points = _open_corners(points, MIN_CORNER)
    points = _resample(_chaikin(points, SMOOTHING), WAYPOINTS)
    points = _resample(_relax_curvature(points, MIN_RADIUS * HALF_WIDTH), WAYPOINTS)

    for _ in range(6):
        outer, _ = loop_corridor(points, HALF_WIDTH)
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        scale = min(
            (width - 2 * EDGE_MARGIN) / max(max(xs) - min(xs), 1e-9),
            (height - 2 * EDGE_MARGIN) / max(max(ys) - min(ys), 1e-9),
        )
        # Recentre on the wall's own bounding box, so the margin is even all round.
        shift_x = mid_x - (max(xs) + min(xs)) / 2
        shift_y = mid_y - (max(ys) + min(ys)) / 2
        points = [
            (mid_x + (x + shift_x - mid_x) * scale, mid_y + (y + shift_y - mid_y) * scale) for x, y in points
        ]
        if 0.999 <= scale <= 1.0 and abs(shift_x) < 0.5 and abs(shift_y) < 0.5:
            break
    return points


def _line_across(grid: Grid, boundary: Boundary, points, index: int) -> Line:
    """A horizontal or vertical line spanning the corridor at a centreline point.

    Whichever axis lies closer to square across the track is chosen, then the
    line is grown outwards node by node until it runs into a wall. Staying
    axis-aligned means every cell along it is a node the car can be placed on.
    """
    x, y = points[index]
    nx, ny = normal_at(points, index)
    step = (1, 0) if abs(nx) > abs(ny) else (0, 1)

    middle = _nearest_open_node(grid, boundary, (x, y))
    ends = []
    for sign in (1, -1):
        col, row = middle
        while True:
            ahead = col + step[0] * sign, row + step[1] * sign
            if not grid.contains(*ahead) or not boundary.contains_point(grid.world_pos(*ahead)):
                break
            col, row = ahead
        ends.append((col, row))
    return Line(*ends)


def _tangent(points, i: int) -> tuple[float, float]:
    """Unit vector along the track at point i."""
    before, after = points[(i - 1) % len(points)], points[(i + 1) % len(points)]
    dx, dy = after[0] - before[0], after[1] - before[1]
    length = math.hypot(dx, dy) or 1.0
    return dx / length, dy / length


def _pick_start_straight(grid: Grid, boundary: Boundary, points) -> tuple[Line, Line, int]:
    """The start and finish lines, drawn across the same straight.

    They belong together the way a real circuit's do — the finish a few cells
    before the start, both square across the same stretch, so the pair reads as
    one start/finish. That needs a length of track which is straight (the two
    lines end up parallel rather than splayed around a bend) and running roughly
    along an axis (an axis-aligned line cuts it squarely rather than running
    away down the track). Every point on the lap is scored on those two counts
    and the best is tried first.
    """
    count = len(points)

    def score(index: int) -> float:
        behind = (index - LINE_SPACING) % count
        ahead_dir, behind_dir = _tangent(points, index), _tangent(points, behind)
        straightness = ahead_dir[0] * behind_dir[0] + ahead_dir[1] * behind_dir[1]
        squareness = min(
            max(abs(component) for component in normal_at(points, i)) for i in (index, behind)
        )
        return straightness * squareness

    for index in sorted(range(count), key=score, reverse=True):
        behind = (index - LINE_SPACING) % count
        lines = [_line_across(grid, boundary, points, i) for i in (index, behind)]
        if not all(_usable_line(line) for line in lines):
            continue
        start_line, finish_line = lines
        gap = min(
            math.dist(grid.world_pos(*a), grid.world_pos(*b))
            for a in start_line.nodes()
            for b in finish_line.nodes()
        )
        if gap < MIN_LINE_GAP * grid.spacing:
            continue  # nudged into each other by the walls
        return start_line, finish_line, index
    raise ValueError("no straight long enough to hold a start and finish line")


def _usable_line(line: Line) -> bool:
    """A line the car can line up on: axis-aligned, and only as wide as the track."""
    return line.axis_aligned and MIN_LINE_NODES <= len(line.nodes()) <= MAX_LINE_NODES


def _nearest_open_node(grid: Grid, boundary: Boundary, point) -> tuple[int, int]:
    """Node closest to a world point that is actually on the tarmac."""
    col, row = round(point[0] / grid.spacing), round(point[1] / grid.spacing)
    candidates = [(col + dc, row + dr) for dc in (0, -1, 1) for dr in (0, -1, 1)]
    for node in candidates:
        if grid.contains(*node) and boundary.contains_point(grid.world_pos(*node)):
            return node
    raise ValueError(f"no drivable node near {point}")


def _perpendicular_line(grid: Grid, points, index: int) -> Line:
    """A line at a true right angle to the track — for crossing tests only."""
    x, y = points[index]
    nx, ny = normal_at(points, index)
    reach = HALF_WIDTH
    ends = [
        (round((x + nx * reach * sign) / grid.spacing), round((y + ny * reach * sign) / grid.spacing))
        for sign in (1, -1)
    ]
    return Line(*ends)


def build(grid: Grid = GRID, variant: int = 0) -> Track:
    """Build one circuit. Raises ValueError if this variant is not usable."""
    points = centerline(grid, variant)

    # Two stretches running closer than the track is wide would merge into a
    # blob of tarmac with no wall between them. Cheaper to catch here than to
    # discover as a hole in the circuit.
    if separation(points, HALF_WIDTH) < 2.4:
        raise ValueError(f"variant {variant}: the circuit runs too close to itself")

    outer, infield = loop_corridor(points, HALF_WIDTH)
    # A tight enough bend folds the offset ring over itself; that is not a track.
    for name, ring in (("outer wall", outer), ("infield", infield)):
        if not is_simple(ring):
            raise ValueError(f"variant {variant}: {name} crosses itself")

    width, height = grid.world_size
    edge = EDGE_MARGIN - 1.0  # a hair of slack for floating point
    if any(not (edge <= x <= width - edge and edge <= y <= height - edge) for x, y in outer):
        raise ValueError(f"variant {variant}: the wall crowds the grid edge")

    boundary = Boundary(outer=outer, holes=(infield,))
    start_line, finish_line, start_at = _pick_start_straight(grid, boundary, points)

    return Track(
        grid=grid,
        boundary=boundary,
        start_line=start_line,
        finish_line=finish_line,
        # Half a lap on, so the only way to reach the finish is the long way
        # round — a car that reversed over the line has not passed it.
        checkpoint=_perpendicular_line(grid, points, (start_at + int(WAYPOINTS * CHECKPOINT_AT)) % WAYPOINTS),
    )


def build_next(grid: Grid = GRID, after: int = -1) -> tuple[Track, int]:
    """The next usable circuit after `after`, and its variant number.

    Most random shapes tangle somewhere — a bend too tight for the corridor, two
    stretches run together, no clean line across the track — so roughly two in
    three are thrown away here rather than shown to the player. Pass after=-1
    for the first usable circuit of all.
    """
    for variant in range(after + 1, after + 200):
        try:
            return build(grid, variant), variant
        except ValueError:
            continue
    raise RuntimeError("no usable circuit found — the shape constraints are too tight")
