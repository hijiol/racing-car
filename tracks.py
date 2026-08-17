"""Circuits you drew yourself, kept on disk.

Each one is a JSON file in tracks/: the centreline in normalised (0..1)
coordinates, the grid it was drawn on, how wide the road is and where the
start/finish belongs. Normalised and grid-stamped so a track rebuilds exactly
the same on any run, whatever the window is doing.

The grid is laid out to match the ruling of graph paper — two nodes per square —
so a road drawn two squares wide comes out five nodes across, and the lattice
lines up with the paper underneath.
"""

import json
import math
import pathlib

from grid import Grid

FOLDER = pathlib.Path(__file__).parent / "tracks"

# The sketch is on 5mm A4 graph paper, whose squares came out 12 pixels across in
# the photo. Two grid cells to a square puts a node on every ruled line and one
# between, which is what makes a two-square road exactly five nodes wide.
CELLS_PER_SQUARE = 2
ROAD_SQUARES = 2.0  # how many paper squares wide the road is
HALF_WIDTH_CELLS = ROAD_SQUARES * CELLS_PER_SQUARE / 2
GATES = 8

# Grid left clear around the sheet. A drawing that runs to the edge of the paper
# still needs somewhere to put its outer wall, and the builder wants clear grid
# beyond that again — without this margin, tracing right to the paper's edge is
# refused for crowding the grid.
MARGIN_CELLS = 10


def canvas_grid(aspect: float, squares_tall: int = 57, spacing: float = 7.0) -> Grid:
    """A grid ruled like the paper: `squares_tall` squares down, `aspect` wide.

    Two nodes to a square, with a margin of clear grid all round the sheet.
    """
    rows = squares_tall * CELLS_PER_SQUARE + 1 + 2 * MARGIN_CELLS
    cols = round(squares_tall * aspect) * CELLS_PER_SQUARE + 1 + 2 * MARGIN_CELLS
    return Grid(cols=cols, rows=rows, spacing=spacing)


def paper_rect(grid: Grid) -> tuple[float, float, float, float]:
    """Where the sheet sits in the grid's world: (x, y, width, height)."""
    inset = MARGIN_CELLS * grid.spacing
    width, height = grid.world_size
    return inset, inset, width - 2 * inset, height - 2 * inset


def half_width(grid: Grid) -> float:
    return HALF_WIDTH_CELLS * grid.spacing


def save(name: str, points, start_near, grid: Grid) -> pathlib.Path:
    """Write a drawn circuit to tracks/<name>.json and return the path."""
    FOLDER.mkdir(exist_ok=True)
    path = FOLDER / f"{_slug(name)}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "points": [[round(x, 5), round(y, 5)] for x, y in points],
                "start_near": [round(start_near[0], 5), round(start_near[1], 5)],
                "grid": {"cols": grid.cols, "rows": grid.rows, "spacing": grid.spacing},
                "gates": GATES,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return path


def load_all() -> list[dict]:
    """Every saved circuit, oldest first. Unreadable files are skipped, not fatal."""
    if not FOLDER.is_dir():
        return []
    saved = []
    for path in sorted(FOLDER.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if len(data.get("points", [])) >= 8:
                saved.append(data)
        except (json.JSONDecodeError, OSError):
            continue  # a half-written file should not stop the game starting
    return saved


def build(data: dict):
    """Rebuild a saved circuit into a Track."""
    import course  # here rather than at the top: course reads this module

    grid = Grid(**data["grid"])
    return course.build_track(
        grid,
        data["points"],
        half_width=half_width(grid),
        gates=data.get("gates", GATES),
        start_near=data["start_near"],
        name=data["name"],
    )


def _slug(name: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in name.strip()]
    return "".join(keep).strip("-") or "track"


def closest_index(points, target) -> int:
    """Index of the point nearest `target` — used to place the start/finish."""
    return min(range(len(points)), key=lambda i: math.dist(points[i], target))
