"""Grid model: the lattice of points the world is measured against.

Nodes sit at integer coordinates (col, row) and edges connect orthogonal
neighbours, so the squares on screen are four nodes and the four edges between
them. This file knows nothing about the track or the car — it is pure geometry,
and track.py decides which of these nodes are actually drivable.
"""

from dataclasses import dataclass

Node = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    cols: int  # number of nodes horizontally
    rows: int  # number of nodes vertically
    spacing: float = 40.0  # world units between adjacent nodes

    def contains(self, col: int, row: int) -> bool:
        return 0 <= col < self.cols and 0 <= row < self.rows

    def nodes(self):
        """Every node in the grid, row-major."""
        for row in range(self.rows):
            for col in range(self.cols):
                yield col, row

    def edges(self):
        """Each connecting segment once, as ((col, row), (col, row))."""
        for col, row in self.nodes():
            if self.contains(col + 1, row):
                yield (col, row), (col + 1, row)
            if self.contains(col, row + 1):
                yield (col, row), (col, row + 1)

    def world_pos(self, col: int, row: int) -> tuple[float, float]:
        """Node index -> world position."""
        return col * self.spacing, row * self.spacing

    @property
    def world_size(self) -> tuple[float, float]:
        return (self.cols - 1) * self.spacing, (self.rows - 1) * self.spacing


