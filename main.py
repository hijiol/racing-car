"""Drive the car by hand round the circuit: one lap, start line to finish line.

The track is a closed loop with an infield you have to go around. The finish
line sits just behind the start line, so a lap only counts once the gates spread
round the circuit have gone by in order — no reversing over the finish, and no
running half way out and back.

The car keeps a velocity, so each turn it may only nudge that velocity by one
step per axis — nine choices, drawn as rings. Leaving the tarmac is not one of
them, and neither is a move whose straight line would clip a wall. Pile into a
corner too fast and every option is gone: that is a crash, and the restart
button appears.

Run:   python main.py
Mouse: left click one of the rings to take that move,
       left click elsewhere on the start line to move where the car sets off,
       right click a node on the finish line to pick the finish,
       left drag to pan, wheel to zoom.
Keys:  Backspace undoes a move, R restarts the run, M loads the next map,
       P shows the optimal line as a ghost, V resets the view, Esc quits.
"""

import math
import threading

import pygame

import course
import solver
from car import Car
from track import Track

WINDOW_SIZE = (1200, 800)
BACKGROUND = (18, 20, 26)
TARMAC = (36, 39, 48)
WALL_COLOR = (110, 122, 150)
LINE_COLOR = (46, 52, 68)
NODE_COLOR = (86, 96, 120)
OFF_TRACK_COLOR = (40, 44, 54)
HOVER_COLOR = (255, 190, 90)
START_COLOR = (90, 220, 130)
FINISH_COLOR = (240, 90, 110)
CAR_COLOR = (255, 230, 120)
TRAIL_COLOR = (200, 160, 60)
OPTION_COLOR = (120, 190, 255)
TEXT_COLOR = (150, 160, 180)
CRASH_COLOR = (240, 90, 110)
CHECKPOINT_COLOR = (120, 130, 160)
CHECKPOINT_DONE = (58, 64, 80)
PATH_COLOR = (170, 140, 240)

NODE_RADIUS = 2.5
SLOT_RADIUS = 4.5
MARKER_RADIUS = 7.0
GATE_LABEL_OFFSET = 14.0  # pixels past the end of a gate to sit its number

# The two lines the car runs between, and the colour each is drawn in.
LINES = (("start", START_COLOR), ("finish", FINISH_COLOR))
DRAG_SLOP = 4  # pixels of movement still counted as a click, not a pan


class View:
    """Maps world coordinates to screen pixels."""

    def __init__(self, grid, window: tuple[int, int]):
        self.grid = grid
        self.window = window
        self.reset()

    def reset(self) -> None:
        world_w, world_h = self.grid.world_size
        self.zoom = min(self.window[0] / (world_w + 80), self.window[1] / (world_h + 80))
        self.offset_x = (self.window[0] - world_w * self.zoom) / 2
        self.offset_y = (self.window[1] - world_h * self.zoom) / 2

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return x * self.zoom + self.offset_x, y * self.zoom + self.offset_y

    def node_screen(self, node) -> tuple[float, float]:
        return self.to_screen(*self.grid.world_pos(*node))

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.offset_x) / self.zoom, (sy - self.offset_y) / self.zoom

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Zoom while keeping the world point under (sx, sy) pinned there."""
        wx, wy = self.to_world(sx, sy)
        self.zoom = max(0.2, min(6.0, self.zoom * factor))
        self.offset_x = sx - wx * self.zoom
        self.offset_y = sy - wy * self.zoom

    def pan(self, dx: float, dy: float) -> None:
        self.offset_x += dx
        self.offset_y += dy


def nearest_node(grid, view: View, mouse: tuple[int, int]):
    """Grid node closest to the mouse, or None if the mouse is well off-grid."""
    wx, wy = view.to_world(*mouse)
    col = round(wx / grid.spacing)
    row = round(wy / grid.spacing)
    if not grid.contains(col, row):
        return None
    nx, ny = grid.world_pos(col, row)
    if abs(wx - nx) > grid.spacing / 2 or abs(wy - ny) > grid.spacing / 2:
        return None
    return col, row


def draw_gate_number(surface, font, number: int, span, colour) -> None:
    """Number a gate, just off the end of it.

    Sat past the wall rather than on the tarmac: the numbers say which way round
    to go, and the driving line is busy enough without digits under it.
    """
    (x1, y1), (x2, y2) = span
    dx, dy = x1 - x2, y1 - y2
    length = math.hypot(dx, dy) or 1.0
    centre = (x1 + dx / length * GATE_LABEL_OFFSET, y1 + dy / length * GATE_LABEL_OFFSET)
    glyph = font.render(str(number), True, colour)
    box = glyph.get_rect(center=centre)
    pygame.draw.circle(surface, BACKGROUND, centre, max(glyph.get_width(), glyph.get_height()) * 0.8)
    surface.blit(glyph, box)


def draw_track(surface, track: Track, view: View, font, hovered, gates_passed: int = 0) -> None:
    """Tarmac, walls, the grid over the top, then the two lines."""
    grid = track.grid

    outline = [view.to_screen(*p) for p in track.boundary.outer]
    pygame.draw.polygon(surface, TARMAC, outline)
    for hole in track.boundary.holes:
        pygame.draw.polygon(surface, BACKGROUND, [view.to_screen(*p) for p in hole])

    for (c1, r1), (c2, r2) in grid.edges():
        # Only draw grid lines whose ends are both drivable, so the lattice
        # fades out at the walls instead of hiding the track shape.
        if track.is_open((c1, r1)) and track.is_open((c2, r2)):
            pygame.draw.line(surface, LINE_COLOR, view.node_screen((c1, r1)), view.node_screen((c2, r2)), 1)

    for ring in (track.boundary.outer, *track.boundary.holes):
        pygame.draw.polygon(surface, WALL_COLOR, [view.to_screen(*p) for p in ring], width=2)

    radius = max(1.2, NODE_RADIUS * view.zoom)
    for node in grid.nodes():
        color = NODE_COLOR if track.is_open(node) else OFF_TRACK_COLOR
        if node == hovered:
            color = HOVER_COLOR
        pygame.draw.circle(surface, color, view.node_screen(node), radius)

    for index, gate in enumerate(track.checkpoints):
        span = [view.to_screen(*p) for p in track.line_span(gate)]
        # Gates already behind the car fade out, so the next one to aim for reads.
        colour = CHECKPOINT_COLOR if index >= gates_passed else CHECKPOINT_DONE
        pygame.draw.line(surface, colour, *span, width=max(1, int(2 * view.zoom)))
        draw_gate_number(surface, font, index + 1, span, colour)

    for which, color in LINES:
        line = track.line_for(which)
        # Drawn wall to wall rather than node to node, so it meets the tarmac edge.
        span = [view.to_screen(*p) for p in track.line_span(line)]
        pygame.draw.line(surface, color, *span, width=max(1, int(2 * view.zoom)))
        # The slots have to out-read the line they sit on, or the line looks like
        # one solid bar and every node under it looks selectable.
        for node in track.line_nodes(which):
            centre = view.node_screen(node)
            radius = max(4.0, SLOT_RADIUS * view.zoom)
            pygame.draw.circle(surface, BACKGROUND, centre, radius + 2)
            pygame.draw.circle(surface, color, centre, radius)


def draw_markers(surface, track: Track, view: View, font) -> None:
    """The chosen start and finish nodes, lettered so they read at a glance."""
    for which, color in LINES:
        pos = view.node_screen(getattr(track, which))
        radius = max(5.0, MARKER_RADIUS * view.zoom)
        pygame.draw.circle(surface, color, pos, radius)
        pygame.draw.circle(surface, BACKGROUND, pos, radius, width=2)
        glyph = font.render(which[0].upper(), True, BACKGROUND)
        surface.blit(glyph, glyph.get_rect(center=pos))


def draw_car(surface, car: Car, track: Track, view: View, options, hovered) -> None:
    """The trail driven so far, the moves on offer, and the car itself."""
    if len(car.history) > 1:
        pygame.draw.lines(surface, TRAIL_COLOR, False, [view.node_screen(n) for n in car.history], 2)

    for node in options:
        color = HOVER_COLOR if node == hovered else OPTION_COLOR
        pygame.draw.circle(surface, color, view.node_screen(node), max(3.0, 5.0 * view.zoom), width=2)
    if car.velocity != (0, 0) and track.grid.contains(*car.coasting_to()):
        pygame.draw.line(
            surface, OPTION_COLOR, view.node_screen(car.pos), view.node_screen(car.coasting_to()), 1
        )

    pos = view.node_screen(car.pos)
    radius = max(5.0, 7.0 * view.zoom)
    pygame.draw.circle(surface, CAR_COLOR, pos, radius)
    pygame.draw.circle(surface, BACKGROUND, pos, radius, width=2)


def draw_banner(surface, font, big_font, text, color) -> pygame.Rect:
    """Crash / finish notice with a restart button. Returns the button's rect."""
    width, height = surface.get_size()
    panel = pygame.Rect(0, 0, 340, 130)
    panel.center = (width // 2, height // 2)
    pygame.draw.rect(surface, BACKGROUND, panel, border_radius=8)
    pygame.draw.rect(surface, color, panel, width=2, border_radius=8)

    headline = big_font.render(text, True, color)
    surface.blit(headline, headline.get_rect(center=(panel.centerx, panel.top + 40)))

    button = pygame.Rect(0, 0, 200, 40)
    button.center = (panel.centerx, panel.bottom - 40)
    hovered = button.collidepoint(pygame.mouse.get_pos())
    pygame.draw.rect(surface, color if hovered else BACKGROUND, button, border_radius=6)
    pygame.draw.rect(surface, color, button, width=2, border_radius=6)
    label = font.render("RESTART  (R)", True, BACKGROUND if hovered else color)
    surface.blit(label, label.get_rect(center=button.center))
    return button


def draw_path(surface, solution, view: View) -> None:
    """The optimal run as a ghost line: where a perfect driver would go."""
    points = [view.node_screen(node) for node in solution.path]
    pygame.draw.lines(surface, PATH_COLOR, False, points, max(1, int(2 * view.zoom)))
    for point in points:
        pygame.draw.circle(surface, PATH_COLOR, point, max(2.0, 3.0 * view.zoom))
    # The ghost picks its own start slot, which may not be the player's.
    pygame.draw.circle(surface, PATH_COLOR, points[0], max(5.0, MARKER_RADIUS * view.zoom), width=2)


def draw_map_button(surface, font, variant: int) -> pygame.Rect:
    """Top-right button that swaps the circuit. Returns its rect."""
    button = pygame.Rect(surface.get_width() - 186, 8, 172, 32)
    hovered = button.collidepoint(pygame.mouse.get_pos())
    pygame.draw.rect(surface, TEXT_COLOR if hovered else BACKGROUND, button, border_radius=6)
    pygame.draw.rect(surface, TEXT_COLOR, button, width=1, border_radius=6)
    label = font.render(f"MAP {variant}  >  (M)", True, BACKGROUND if hovered else TEXT_COLOR)
    surface.blit(label, label.get_rect(center=button.center))
    return button


def draw_path_button(surface, font, showing: bool) -> pygame.Rect:
    """Toggle for the ghost line, sat beside the map button."""
    button = pygame.Rect(surface.get_width() - 186, 46, 172, 32)
    hovered = button.collidepoint(pygame.mouse.get_pos())
    face = PATH_COLOR if showing or hovered else BACKGROUND
    pygame.draw.rect(surface, face, button, border_radius=6)
    pygame.draw.rect(surface, PATH_COLOR, button, width=1, border_radius=6)
    text = "HIDE BEST  (P)" if showing else "BEST PATH  (P)"
    label = font.render(text, True, BACKGROUND if (showing or hovered) else PATH_COLOR)
    surface.blit(label, label.get_rect(center=button.center))
    return button


def place(track: Track, which: str, node) -> None:
    """Pick a node on a line, ignoring clicks that land off it or on the other marker."""
    try:
        track.choose(which, node)
    except ValueError:
        pass


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("racing-car")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    big_font = pygame.font.SysFont("consolas", 26, bold=True)

    # Not every variant number yields a usable circuit, so ask for the first that does.
    track, variant = course.build_next(course.GRID, after=-1)
    grid = track.grid
    view = View(grid, WINDOW_SIZE)
    car = Car(track.start)
    dragging = False
    drag_distance = 0
    restart_button = pygame.Rect(0, 0, 0, 0)
    map_button = pygame.Rect(0, 0, 0, 0)
    path_button = pygame.Rect(0, 0, 0, 0)
    show_path = False
    solutions: dict[int, solver.Solution | None] = {}  # None means no way round
    solving: set[int] = set()
    running = True

    def next_map() -> None:
        nonlocal track, variant, show_path
        track, variant = course.build_next(grid, after=variant)
        car.reset(track.start)
        # A new circuit means a new answer; show it only when it is asked for.
        show_path = False

    def request_solution(for_variant: int) -> None:
        """Solve in the background — a second of searching must not stall the game."""
        if for_variant in solutions or for_variant in solving:
            return
        solving.add(for_variant)

        def work() -> None:
            # Solve against its own copy of the track, so the search never shares
            # the caches the drawing thread is reading from.
            solutions[for_variant] = solver.solve(course.build(grid, for_variant))
            solving.discard(for_variant)

        threading.Thread(target=work, daemon=True).start()

    while running:
        hovered = nearest_node(grid, view, pygame.mouse.get_pos())
        gates_passed, finished = track.lap_progress(car.history)
        options = {} if finished else car.moves(track)
        placing = len(car.history) == 1  # not away yet: still choosing a grid slot
        crashed = not options and not finished
        blocked = finished or crashed  # run over: only restarting gets you moving
        on_start_line = hovered is not None and hovered in track.line_nodes("start")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    car.reset(track.start)
                elif event.key == pygame.K_v:
                    view.reset()
                elif event.key == pygame.K_m:
                    next_map()
                elif event.key == pygame.K_p:
                    show_path = not show_path
                elif event.key == pygame.K_BACKSPACE:
                    car.undo()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_distance = 0
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
                if drag_distance > DRAG_SLOP:
                    pass  # that was a pan, not a click
                elif map_button.collidepoint(event.pos):
                    next_map()
                elif path_button.collidepoint(event.pos):
                    show_path = not show_path
                elif blocked and restart_button.collidepoint(event.pos):
                    car.reset(track.start)
                # Before the first move the start line places the car; once the
                # car is away, driving wins and only a slot it cannot reach
                # anyway counts as asking to start again.
                elif on_start_line and (placing or hovered not in options):
                    place(track, "start", hovered)
                    car.reset(track.start)
                elif hovered in options:
                    car.drive_to(hovered, track)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                if hovered:
                    place(track, "finish", hovered)
            elif event.type == pygame.MOUSEMOTION and dragging:
                drag_distance += abs(event.rel[0]) + abs(event.rel[1])
                view.pan(*event.rel)
            elif event.type == pygame.MOUSEWHEEL:
                view.zoom_at(*pygame.mouse.get_pos(), 1.1**event.y)

        keys = pygame.key.get_pressed()
        pan_speed = 400 * clock.get_time() / 1000
        view.pan(
            (keys[pygame.K_LEFT] - keys[pygame.K_RIGHT]) * pan_speed,
            (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * pan_speed,
        )

        if show_path:
            request_solution(variant)
        solution = solutions.get(variant)

        screen.fill(BACKGROUND)
        draw_track(screen, track, view, font, hovered, gates_passed)
        if show_path and solution is not None:
            draw_path(screen, solution, view)  # under the car, so it never hides live state
        draw_markers(screen, track, view, font)
        draw_car(screen, car, track, view, options, hovered)

        if placing:
            label = "click anywhere on the start line to place the car, then click a ring to move off"
        else:
            label = (
                f"pos {car.pos}   velocity {car.velocity}   speed {car.speed:.1f}   "
                f"turns {len(car.history) - 1}   moves {len(options)}   "
                f"gates {gates_passed}/{len(track.checkpoints)}"
            )
        screen.blit(font.render(label, True, TEXT_COLOR), (14, 12))

        if show_path:
            if variant in solving:
                note, colour = "solving...", TEXT_COLOR
            elif solution is None:
                note, colour = "no way round this one", CRASH_COLOR
            else:
                note, colour = f"best possible: {solution.turns} turns", PATH_COLOR
            screen.blit(font.render(note, True, colour), (14, 34))

        map_button = draw_map_button(screen, font, variant)
        path_button = draw_path_button(screen, font, show_path)

        if crashed:
            restart_button = draw_banner(screen, font, big_font, "CRASHED", CRASH_COLOR)
        elif finished:
            restart_button = draw_banner(
                screen, font, big_font, f"LAP in {len(car.history) - 1} turns", START_COLOR
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
