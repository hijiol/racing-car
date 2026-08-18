"""Circuits read off a drawing, kept on disk.

Each one is a JSON file in tracks/: the two walls as polygons, a centreline for
placing gates, and the grid they were measured onto — all in normalised (0..1)
coordinates. The scan itself is not needed again, so a saved circuit loads
without OpenCV and stays exactly the same forever.

The grid is taken from the paper's own ruling. `NODES_PER_SQUARE` nodes to a
square puts the lattice on the lines the circuit was drawn over.
"""

import json
import pathlib

from grid import Grid

FOLDER = pathlib.Path(__file__).parent / "tracks"

# One node per graph-paper square: the game's lattice *is* the paper's ruling, so
# the four corners of a square on the drawing are four nodes in the game.
NODES_PER_SQUARE = 1
GATES = 8
MARGIN_CELLS = 6  # clear grid around the drawing, so the walls are never at the very edge


def grid_for(scan_size, ruling) -> Grid:
    """A grid ruled like the paper the circuit was drawn on."""
    cell = ruling.pitch / NODES_PER_SQUARE
    width, height = scan_size
    return Grid(
        cols=round(width / cell) + 1 + 2 * MARGIN_CELLS,
        rows=round(height * ruling.y_scale / cell) + 1 + 2 * MARGIN_CELLS,
        spacing=cell,
    )


def placement(ruling, grid: Grid) -> tuple[float, float, float]:
    """Where to put the drawing in the world: (x, y, y_scale).

    Chosen so the paper's ruled lines fall on nodes. The margin sets roughly
    where the sheet sits; the fractional part is then nudged by the ruling's own
    offset, so that the first ruled line — and every one after it — lands on the
    lattice instead of a fraction of a cell away from it.
    """
    cell = grid.spacing
    rough = MARGIN_CELLS * cell
    return rough - (ruling.phase_x % cell), rough - (ruling.phase_y % cell), ruling.y_scale


def to_world(point, place) -> tuple[float, float]:
    """Scan pixels -> world units, squared up and sat on the ruling."""
    offset_x, offset_y, y_scale = place
    return point[0] + offset_x, point[1] * y_scale + offset_y


def save(name: str, drawing, grid: Grid, place, sketch: str = "") -> pathlib.Path:
    """Write a circuit read off a scan to tracks/<name>.json."""
    FOLDER.mkdir(exist_ok=True)
    width, height = grid.world_size

    def norm(points):
        return [[round(x / width, 6), round(y / height, 6)] for x, y in (to_world(p, place) for p in points)]

    path = FOLDER / f"{_slug(name)}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "sketch": sketch,
                "outer": norm(drawing.outer),
                "inners": [norm(inner) for inner in drawing.inners],
                "centre": norm(drawing.centre),
                "start_at": drawing.start_at,
                "road_width": round(drawing.road_width, 2),
                "grid": {"cols": grid.cols, "rows": grid.rows, "spacing": round(grid.spacing, 4)},
                "place": [round(v, 4) for v in place],
                "gates": GATES,
            }
        ),
        encoding="utf-8",
    )
    return path


def load_all() -> list[dict]:
    """Every saved circuit, by name. A half-written file is skipped, not fatal."""
    if not FOLDER.is_dir():
        return []
    saved = []
    for path in sorted(FOLDER.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if len(data.get("outer", [])) >= 8:
                saved.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return saved


def build(data: dict):
    """Rebuild a saved circuit into a Track."""
    import course  # here rather than at the top: course reads this module
    from track import Boundary

    grid = Grid(**data["grid"])
    width, height = grid.world_size

    def world(points):
        return tuple((x * width, y * height) for x, y in points)

    boundary = Boundary(outer=world(data["outer"]), holes=tuple(world(i) for i in data["inners"]))
    centre = list(world(data["centre"]))
    return course.from_walls(
        grid,
        boundary,
        centre,
        half_width=data["road_width"] / 2,
        gates=data.get("gates", GATES),
        prefer_start=data.get("start_at", 0),
        label=data["name"],
        sketch=data.get("sketch", ""),
        # Where the drawing sits in the world, so it can be laid back over the top.
        sketch_place=tuple(data.get("place", (0.0, 0.0, 1.0))),
    )


def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in name.strip()).strip("-") or "track"
