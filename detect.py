"""Read a circuit off a scan of a hand-drawn one.

The drawing already holds the track: two pen lines with the road between them.
So rather than tracing it by hand, this reads it.

The trick is which regions to take the walls from. The start/finish line drawn
across the road cuts the road region into a C, so the road's own outline is not
a clean ring. The area *outside* the circuit and the infield *inside* it are
untouched by that cut — so the outer wall is the hole in the outside, and each
inner wall is an infield's outline. The cut is not a nuisance either: walking
the centreline and finding where it leaves the road says exactly where the
start/finish line was drawn.

Everything here is in image pixels and knows nothing about the game.
"""

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

INK_CLOSE = 9  # kernel that seals the gaps in a biro line
MARK_SHARE = 0.05  # ink smaller than this share of the circuit is a label, not a wall
SPUR_PATH = 1.5  # road widths: a detour longer than this is real, not a drawing artefact
SPUR_CHORD = 0.35  # road widths: how near a detour must return to count as one
SPUR_LOOKAHEAD = 150  # wall points searched for the far end of a detour
MIN_REGION = 2000  # px: smaller enclosed areas are specks, not infields
SIMPLIFY = 1.0  # px of slack when thinning a wall's outline
SMOOTH_WINDOW = 5  # points averaged to take the spurs off a wall
CENTRELINE_POINTS = 400


@dataclass
class Drawing:
    """What a scan turned out to contain."""

    outer: list  # the outer wall, in image pixels
    inners: list  # one outline per infield
    centre: list  # midway round the road; only used to place gates
    start_at: int  # index into `centre` where the drawn start/finish line sits
    road_width: float  # pixels, measured
    lap: float  # length of the centreline, pixels
    size: tuple  # (width, height) of the scan


def read(path: str) -> Drawing:
    """Pull the circuit out of a scan. Raises ValueError if it cannot be read."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot open {path}")
    height, width = image.shape

    ink = _drop_marks(_ink(image))
    outside, road, infields = _regions(ink)
    if not infields:
        raise ValueError("no infield found — is a wall broken, letting the road leak?")

    centre, lap = _centreline(road)
    road_width = _road_width(road, centre)

    # The walls are the edge of the road itself, not the outline of the pen
    # stroke. Taking the pen's outline would hand the car the ink to drive on —
    # and, worse, anything the ink has closed over, such as the sliver between
    # the two legs of a hairpin, which is solid pen rather than a hole.
    ring = _heal_cut(road, centre, road_width)
    outer, inners = _walls(ring)

    outer = _prune_spurs(outer, road_width)
    inners = [_prune_spurs(inner, road_width) for inner in inners]
    return Drawing(
        outer=outer,
        inners=inners,
        centre=centre,
        # The centreline runs from one face of the drawn start/finish line round
        # to the other, so the seam between its ends is exactly that line.
        start_at=0,
        road_width=road_width,
        lap=lap,
        size=(width, height),
    )


@dataclass
class Ruling:
    """The graph paper's own grid, measured off the scan.

    `pitch` is how far apart the ruled lines are and `phase_x`/`phase_y` say
    where the first one falls, so the game's lattice can be laid on the lines
    the circuit was drawn over rather than at some arbitrary offset.

    A scanner rarely comes out perfectly square — here the ruling is 23.9px
    across and 23.5px down — so `y_scale` stretches the drawing very slightly
    to make its squares square. It is well under two percent, invisible in the
    shape, and it is what stops the lattice drifting off the ruling down the page.
    """

    pitch: float
    phase_x: float
    phase_y: float
    y_scale: float


def ruling(path: str) -> Ruling:
    """Measure the graph paper the circuit was drawn on."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot open {path}")
    # The pen would drown out the ruling, so blank it out first.
    near_pen = cv2.dilate(_ink(image), np.ones((15, 15), np.uint8)) > 0
    clean = np.where(near_pen, np.nan, image.astype(np.float32))

    with np.errstate(all="ignore"):
        across = _rule_line(np.nanmean(clean, axis=0))
        down = _rule_line(np.nanmean(clean, axis=1))
    if across is None or down is None:
        raise ValueError("no ruled paper to measure — is this graph paper?")
    (pitch_x, phase_x), (pitch_y, phase_y) = across, down
    return Ruling(
        pitch=pitch_x,
        phase_x=phase_x,
        phase_y=phase_y * pitch_x / pitch_y,
        y_scale=pitch_x / pitch_y,
    )


def _rule_line(profile, low: float = 15.0, high: float = 40.0):
    """Pitch and offset of the ruling along one axis.

    Every candidate spacing is scored by how strongly the profile repeats at it —
    one Fourier term each — and the winner's angle gives where the lines sit.
    """
    darkness = np.nan_to_num(255 - profile, nan=0.0)
    if not darkness.any():
        return None
    # Flatten out the slow shading so only the ruling's rhythm is left.
    darkness = darkness - cv2.GaussianBlur(darkness.reshape(-1, 1), (0, 0), 12).ravel()
    steps = np.arange(len(darkness))
    best = (0.0, None)
    for candidate in np.arange(low, high, 0.02):
        strength = abs((darkness * np.exp(-2j * np.pi * steps / candidate)).sum())
        if strength > best[0]:
            best = (strength, candidate)
    pitch = best[1]
    angle = np.angle((darkness * np.exp(-2j * np.pi * steps / pitch)).sum())
    return float(pitch), float((-angle / (2 * np.pi) * pitch) % pitch)


def _ink(image) -> np.ndarray:
    """Where the pen is. The scan is evenly lit, so a plain threshold does it."""
    _, ink = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((INK_CLOSE, INK_CLOSE), np.uint8))


def _drop_marks(ink):
    """Keep the drawing, drop the writing.

    The circuit is one enormous connected stroke; the corner labels are small
    separate ones. Anything a fraction of the circuit's size is a label, and
    letters left in would be read as walls in the middle of the track.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:
        return ink
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, count)]
    keep = max(areas) * MARK_SHARE
    wanted = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= keep]
    return np.isin(labels, wanted).astype(np.uint8) * 255


def _prune_spurs(points: list, road_width: float) -> list:
    """Drop the little detours the wall makes around things drawn on top of it.

    The arrows beside a start/finish line poke through the wall, and the wall as
    read wanders out around them and straight back. So does any label that
    happens to touch. What tells those apart from real geometry — the sliver
    between the two legs of a hairpin, say, which is every bit as thin — is how
    far the detour runs: an arrow is over within a road width or so, while a
    hairpin runs for many. Short there-and-back excursions go; long ones stay.
    """
    longest = SPUR_PATH * road_width
    reunion = SPUR_CHORD * road_width
    count = len(points)
    kept, i = [], 0
    while i < count:
        kept.append(points[i])
        travelled, rejoin = 0.0, None
        for j in range(i + 1, min(i + SPUR_LOOKAHEAD, count)):
            travelled += math.dist(points[j - 1], points[j])
            if travelled > longest:
                break
            # Back where it started, having gone a good way round: a detour.
            if travelled > reunion * 2 and math.dist(points[i], points[j]) < reunion:
                rejoin = j
        i = rejoin if rejoin is not None else i + 1
    return kept if len(kept) >= 8 else points


def _regions(ink):
    """Split the paper into outside, road and infields.

    Road and infield are told apart by how fat they are: everywhere in the road
    is within half a road width of a wall, while an infield has a deep middle.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - ink, connectivity=4)
    height, width = ink.shape
    outside_id, enclosed = None, []
    for i in range(1, count):
        x, y, w, h, area = (stats[i, k] for k in range(5))
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            if outside_id is None or area > stats[outside_id, cv2.CC_STAT_AREA]:
                outside_id = i
        elif area >= MIN_REGION:
            enclosed.append(i)
    if outside_id is None or not enclosed:
        raise ValueError("the scan does not hold a closed circuit")

    depths = {}
    for i in enclosed:
        mask = (labels == i).astype(np.uint8) * 255
        depths[i] = float(cv2.distanceTransform(mask, cv2.DIST_L2, 5).max())
    road_id = min(depths, key=depths.get)  # the thinnest region is the road
    road = (labels == road_id).astype(np.uint8) * 255
    infields = [(labels == i).astype(np.uint8) * 255 for i in enclosed if i != road_id]
    return labels == outside_id, road, infields


def _heal_cut(road, centre, road_width: float):
    """Close the gap the drawn start/finish line leaves in the road.

    The line cuts the road open, which is how its position is known — but a
    track with a slice taken out of it is not a track, so the slice goes back
    once it has been read. The patch is kept inside the corridor, so it fills
    the gap without spilling over a wall.
    """
    bridge = np.zeros_like(road)
    ends = [(int(round(p[0])), int(round(p[1]))) for p in (centre[0], centre[-1])]
    cv2.line(bridge, ends[0], ends[1], 255, thickness=int(road_width * 1.2))
    room = cv2.dilate(road, np.ones((INK_CLOSE * 3, INK_CLOSE * 3), np.uint8))
    return cv2.bitwise_or(road, cv2.bitwise_and(bridge, room))


def _walls(ring) -> tuple[list, list]:
    """The road's own two edges: its outline, and the infields inside it.

    The short spurs the arrows leave behind are cleaned off by the caller; what
    matters here is that these follow the tarmac, so what the car is given to
    drive on is the road and nothing else.
    """
    contours, hierarchy = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("the road has no outline")
    outer_index = max(
        (i for i in range(len(contours)) if hierarchy[0][i][3] == -1),
        key=lambda i: cv2.contourArea(contours[i]),
    )
    holes = [
        i
        for i in range(len(contours))
        if hierarchy[0][i][3] == outer_index and cv2.contourArea(contours[i]) >= MIN_REGION
    ]
    if not holes:
        raise ValueError("the road does not close into a ring — is a wall broken?")
    return _tidy(contours[outer_index]), [_tidy(contours[i]) for i in holes]


def _tidy(contour) -> list:
    """Thin and smooth a traced outline into a wall."""
    contour = cv2.approxPolyDP(contour, SIMPLIFY, closed=True)
    return _smooth([(float(p[0][0]), float(p[0][1])) for p in contour], SMOOTH_WINDOW)


def _wall(mask) -> list:
    """The outline of a filled region, thinned and smoothed into a wall."""
    mask = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("a region has no outline")
    contour = max(contours, key=cv2.contourArea)
    contour = cv2.approxPolyDP(contour, SIMPLIFY, closed=True)
    points = [(float(p[0][0]), float(p[0][1])) for p in contour]
    return _smooth(points, SMOOTH_WINDOW)


def _smooth(points: list, window: int) -> list:
    """Round off the spurs where a label or an arrow touches a wall."""
    count = len(points)
    if count < window * 2:
        return points
    half = window // 2
    smoothed = []
    for i in range(count):
        neighbours = [points[(i + d) % count] for d in range(-half, half + 1)]
        smoothed.append(
            (sum(p[0] for p in neighbours) / len(neighbours), sum(p[1] for p in neighbours) / len(neighbours))
        )
    return smoothed


def _centreline(road) -> tuple[list, float]:
    """The middle of the road, in order all the way round, and the lap length.

    The road is a C — the start/finish line you drew across it cuts the ring
    open — and that is what makes this easy. Measuring how far every pixel is
    from one cut face, *travelling within the road*, gives a value that climbs
    steadily round the lap. Pixels sharing a value form a cross-section of the
    road, so their centre of mass is a point on the centreline, and sorting
    those by value puts them in lap order.

    Pairing the two walls instead would cut every corner, because the inner wall
    is much the shorter way round.
    """
    small = cv2.resize(road, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    reach = _geodesic(small, _any_pixel(small))
    far_end = np.unravel_index(int(np.argmax(np.where(np.isfinite(reach), reach, -1))), reach.shape)
    reach = _geodesic(small, far_end)  # the two cut faces are the ends of the C
    finite = np.isfinite(reach)
    lap = float(reach[finite].max())

    points = []
    step = lap / CENTRELINE_POINTS
    values = reach[finite]
    ys, xs = np.nonzero(finite)
    order = np.argsort(values)
    values, ys, xs = values[order], ys[order], xs[order]
    edges = np.searchsorted(values, [k * step for k in range(CENTRELINE_POINTS + 1)])
    for i in range(CENTRELINE_POINTS):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        points.append((float(xs[lo:hi].mean()) * 2, float(ys[lo:hi].mean()) * 2))
    return points, lap * 2


def _road_width(road, centre) -> float:
    """How wide the road is, measured up the middle of it.

    Not the average distance-to-wall over the whole road: in a band of width w
    that averages about w/4, because most of the road is nearer one wall than
    the middle. Sampling on the centreline reads the half-width directly.
    """
    dist = cv2.distanceTransform(road, cv2.DIST_L2, 5)
    height, width = road.shape
    ridge = [
        dist[min(height - 1, max(0, int(y))), min(width - 1, max(0, int(x)))] for x, y in centre
    ]
    return 2 * float(np.median([v for v in ridge if v > 0]))


def _any_pixel(mask):
    ys, xs = np.nonzero(mask)
    return int(ys[len(ys) // 2]), int(xs[len(xs) // 2])


def _geodesic(mask, seed) -> np.ndarray:
    """Distance from `seed` to every pixel, travelling only inside the mask."""
    reach = np.full(mask.shape, np.inf, np.float32)
    reach[seed] = 0.0
    frontier = deque([seed])
    inside = mask > 0
    height, width = mask.shape
    while frontier:
        y, x = frontier.popleft()
        here = reach[y, x]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if dy == dx == 0 or not (0 <= ny < height and 0 <= nx < width):
                    continue
                if not inside[ny, nx]:
                    continue
                far = here + (1.41421356 if dy and dx else 1.0)
                if far < reach[ny, nx]:
                    reach[ny, nx] = far
                    frontier.append((ny, nx))
    return reach


def bounds(drawing: Drawing) -> tuple:
    """The circuit's extent in the scan: (x0, y0, x1, y1)."""
    xs = [p[0] for p in drawing.outer]
    ys = [p[1] for p in drawing.outer]
    return min(xs), min(ys), max(xs), max(ys)


def describe(drawing: Drawing) -> str:
    lap = drawing.lap
    return (
        f"{drawing.size[0]}x{drawing.size[1]} scan | road {drawing.road_width:.0f}px "
        f"| lap {lap:.0f}px | outer wall {len(drawing.outer)} points "
        f"| {len(drawing.inners)} infield(s)"
    )
