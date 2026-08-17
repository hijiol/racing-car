"""Draw a circuit by hand, tracing over a photo of one.

The sketch loads as a dimmed backdrop and you drag the mouse along its centre
line — the middle of the road, not its edges. Points are kept as the cursor
moves, the road is drawn around them at the width the paper implies, and trouble
is marked in red while the pen is still in your hand rather than reported as a
failure afterwards.

The grid is ruled like the paper underneath: two nodes to a 5mm square, so a road
two squares wide comes out five nodes across, and what you draw keeps its scale.

Controls
    drag left       draw; release to pause, drag again to carry on
    click the start close the loop (or press Enter)
    Backspace       undo the last stroke        C  clear and start again
    O               fade the sketch behind      Enter  save when the loop is closed
    click (closed)  put the start/finish line there
    right-drag      pan          wheel  zoom          Esc  leave without saving
"""

import math

import pygame

import course
import tracks

MIN_SPACING = 6.0  # world units between kept points: thins the mouse, keeps corners
CLOSE_RADIUS = 14  # pixels from the first point that count as closing the loop

BACKDROP_FADES = (0.6, 0.9, 0.0, 0.3)  # what O cycles through; the first is the default

INK = (150, 205, 255)
ROAD = (70, 80, 100)
TROUBLE = (240, 90, 110)
TEXT = (150, 160, 180)
HINT = (110, 120, 140)
START_MARK = (90, 220, 130)


def _crop_to_paper(image: pygame.Surface) -> pygame.Surface:
    """Trim a photo down to the sheet of paper in it.

    A photo of a drawing has desk around the edges. Cropping to the paper is
    what lets the grid line up with the ruling on it: the canvas is then the
    sheet itself, so many squares wide by so many tall, and a road drawn two
    squares wide really is two squares wide.
    """
    width, height = image.get_size()
    step = max(1, min(width, height) // 120)

    def bright_range(size: int, sample) -> tuple[int, int]:
        levels = [sample(i) for i in range(0, size, step)]
        threshold = (max(levels) + min(levels)) / 2
        lit = [i * step for i, level in enumerate(levels) if level > threshold]
        return (lit[0], lit[-1]) if lit else (0, size - 1)

    def column(x: int) -> float:
        return sum(sum(image.get_at((x, y))[:3]) for y in range(0, height, step)) / max(1, height // step)

    def row(y: int) -> float:
        return sum(sum(image.get_at((x, y))[:3]) for x in range(0, width, step)) / max(1, width // step)

    x0, x1 = bright_range(width, column)
    y0, y1 = bright_range(height, row)
    if x1 - x0 < width // 3 or y1 - y0 < height // 3:
        return image  # no clear sheet in there; trace the whole photo instead
    return image.subsurface(pygame.Rect(x0, y0, x1 - x0 + 1, y1 - y0 + 1)).copy()


class Editor:
    """Everything about the drawing in progress."""

    def __init__(self, backdrop: str, view_factory, window):
        self.grid = None
        self.points: list[tuple[float, float]] = []  # world coordinates
        self.stroke_starts: list[int] = []  # where each stroke began, for undo
        self.closed = False
        self.start_near = None
        self.drawing = False
        self.panning = False
        self.fade_step = 0
        self.message = ""
        self.built = None  # the finished Track, once saved

        self.image = self._load(backdrop)
        aspect = (self.image.get_width() / self.image.get_height()) if self.image else 0.75
        self.grid = tracks.canvas_grid(aspect)
        self.view = view_factory(self.grid, window)
        self.window = window

    @staticmethod
    def _load(path):
        try:
            image = pygame.image.load(path).convert_alpha()
        except (pygame.error, FileNotFoundError):
            return None  # drawing freehand with no photo to trace is still fine
        return _crop_to_paper(image)

    # ------------------------------------------------------------------ input

    def handle(self, event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._press(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drawing = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            self.panning = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            self.panning = False
        elif event.type == pygame.MOUSEMOTION:
            if self.panning:
                self.view.pan(*event.rel)
            elif self.drawing:
                self._extend(event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            self.view.zoom_at(*pygame.mouse.get_pos(), 1.1**event.y)
        elif event.type == pygame.KEYDOWN:
            self._key(event.key)

    def _press(self, pos) -> None:
        if self.closed:
            # The loop is done; a click now says where the start/finish goes.
            self.start_near = self.view.to_world(*pos)
            self.message = "start/finish set — Enter to save"
            return
        if len(self.points) > 8 and self._near_first(pos):
            self.close()
            return
        self.drawing = True
        self.stroke_starts.append(len(self.points))
        self._extend(pos, force=True)

    def _extend(self, pos, force: bool = False) -> None:
        point = self.view.to_world(*pos)
        if force or not self.points or math.dist(point, self.points[-1]) >= MIN_SPACING:
            self.points.append(point)

    def _key(self, key) -> None:
        if key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.close() if not self.closed else self.save()
        elif key == pygame.K_BACKSPACE:
            self.undo()
        elif key == pygame.K_c:
            self.points.clear()
            self.stroke_starts.clear()
            self.closed = False
            self.start_near = None
            self.message = ""
        elif key == pygame.K_o:
            self.fade_step = (self.fade_step + 1) % len(BACKDROP_FADES)

    def _near_first(self, pos) -> bool:
        if not self.points:
            return False
        first = self.view.to_screen(*self.points[0])
        return math.dist(first, pos) <= CLOSE_RADIUS

    def undo(self) -> None:
        if self.closed:
            self.closed = False
            self.start_near = None
            self.message = "loop reopened"
        elif self.stroke_starts:
            del self.points[self.stroke_starts.pop() :]

    # ----------------------------------------------------------------- output

    def close(self) -> None:
        if len(self.points) < 12:
            self.message = "too few points to make a circuit"
            return
        self.closed = True
        self.message = "click where the start/finish line goes, then Enter"

    def save(self, name: str = "SPA"):
        """Build the drawn circuit and write it out. Sets `built` on success."""
        width, height = self.grid.world_size
        normalised = [(x / width, y / height) for x, y in self.points]
        start = self.start_near or self.points[0]
        start_near = (start[0] / width, start[1] / height)
        try:
            self.built = course.build_track(
                self.grid,
                normalised,
                half_width=tracks.half_width(self.grid),
                gates=tracks.GATES,
                start_near=start_near,
                name=name,
            )
        except ValueError as problem:
            self.message = str(problem).split(":")[-1].strip()
            self.built = None
            return None
        tracks.save(name, normalised, start_near, self.grid)
        return self.built

    def trouble_spots(self) -> list[tuple[float, float]]:
        """Places the drawing cannot become a track: too tight, or too close to itself.

        Uses the same measures the builder will apply, so what is flagged here is
        exactly what would reject the shape later.
        """
        count = len(self.points)
        if count < 12:
            return []
        half = tracks.half_width(self.grid)
        spots = []

        # Once the loop is closed the two ends are neighbours, so both tests wrap.
        span = range(count) if self.closed else range(1, count - 1)
        for i in span:
            before, here, after = (self.points[(i + d) % count] for d in (-1, 0, 1))
            if course._circumradius(before, here, after) < course.MIN_RADIUS * half:
                spots.append(here)

        skip = max(6, count // 12)
        for i in range(count):
            for j in range(i + skip, count):
                if self.closed and count - (j - i) < skip:
                    continue  # neighbours around the seam
                if math.dist(self.points[i], self.points[j]) < course.TRACE_MIN_SEPARATION * half:
                    spots.append(self.points[i])
                    break
        return spots

    def draw(self, surface, font) -> None:
        surface.fill((18, 20, 26))
        self._draw_backdrop(surface)
        self._draw_lattice(surface)

        if len(self.points) > 1:
            screen_points = [self.view.to_screen(*p) for p in self.points]
            road = max(2, int(tracks.half_width(self.grid) * 2 * self.view.zoom))
            pygame.draw.lines(surface, ROAD, self.closed, screen_points, road)
            pygame.draw.lines(surface, INK, self.closed, screen_points, 2)

        for spot in self.trouble_spots():
            pygame.draw.circle(surface, TROUBLE, self.view.to_screen(*spot), 5, width=2)

        if self.points:
            first = self.view.to_screen(*self.points[0])
            colour = START_MARK if self.closed else INK
            pygame.draw.circle(surface, colour, first, CLOSE_RADIUS // 2, width=2)
        if self.start_near:
            pygame.draw.circle(surface, START_MARK, self.view.to_screen(*self.start_near), 9, width=3)

        self._draw_help(surface, font)

    def _draw_backdrop(self, surface) -> None:
        if not self.image:
            return
        fade = BACKDROP_FADES[self.fade_step]
        if fade <= 0:
            return
        # The sheet sits inside the grid's margin, so its ruling lines up with
        # the lattice and a trace at the paper's edge still has room for its wall.
        x, y, paper_w, paper_h = tracks.paper_rect(self.grid)
        size = (max(1, int(paper_w * self.view.zoom)), max(1, int(paper_h * self.view.zoom)))
        scaled = pygame.transform.smoothscale(self.image, size)
        scaled.set_alpha(int(255 * fade))
        surface.blit(scaled, self.view.to_screen(x, y))

    def _draw_lattice(self, surface) -> None:
        """The grid, with every second node brighter: those are the paper's ruling."""
        step = tracks.CELLS_PER_SQUARE
        for col, row in self.grid.nodes():
            on_rule = (col - tracks.MARGIN_CELLS) % step == 0 and (row - tracks.MARGIN_CELLS) % step == 0
            colour = (74, 82, 102) if on_rule else (40, 44, 54)
            pygame.draw.circle(surface, colour, self.view.node_screen((col, row)), 1.6 if on_rule else 1.0)

    def _draw_help(self, surface, font) -> None:
        panel = pygame.Surface((surface.get_width(), 96))
        panel.fill((18, 20, 26))
        panel.set_alpha(215)  # the sketch must not fight the instructions
        surface.blit(panel, (0, 0))
        lines = [
            "DRAW A CIRCUIT — drag along the middle of the road on the sketch",
            "click the green dot to close the loop   Backspace undo   C clear   O fade sketch",
            "closed: click where the start/finish goes, then Enter to save   Esc to leave",
            f"{len(self.points)} points   road {tracks.ROAD_SQUARES:.0f} squares wide "
            f"({round(tracks.HALF_WIDTH_CELLS * 2) + 1} nodes across)",
        ]
        for i, text in enumerate(lines):
            surface.blit(font.render(text, True, TEXT if i == 0 else HINT), (14, 12 + i * 18))
        if self.message:
            surface.blit(font.render(self.message, True, TROUBLE), (14, 12 + len(lines) * 18))
