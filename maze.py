#!/usr/bin/env python3
"""maze-generator - a zero-dependency perfect-maze generator, solver, and renderer.

Carve perfect mazes with randomized depth-first search (the "recursive
backtracker" algorithm), solve them with breadth-first search, and render
them as plain ASCII art. Every generated maze is *perfect*: fully connected,
with exactly one corridor path between any two cells.

Usage:
    python maze.py --width 20 --height 10 --seed 42
    python maze.py --width 20 --height 10 --seed 42 --solve
    python maze.py --width 15 --height 15 --seed 7 --start 0,0 --end 14,14 --solve
    python maze.py --stats --seed 42

Standard library only. Requires Python 3.10+.
"""

from __future__ import annotations

import argparse
import random
from collections import deque

# Neighbor offsets (east, west, south, north); the order is part of the
# deterministic maze format, since it drives each cell's candidate list.
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class Maze:
    """A rectangular grid maze.

    Cells are ``(x, y)`` tuples with ``0 <= x < width`` and ``0 <= y < height``.
    ``links[cell]`` holds the set of cells reachable from ``cell`` in one step;
    the relation is always symmetric for a generated maze.
    """

    def __init__(self, width: int, height: int, links: dict):
        if width < 1 or height < 1:
            raise ValueError("maze dimensions must be >= 1")
        self.width = int(width)
        self.height = int(height)
        self.links = {cell: set(neighbors) for cell, neighbors in links.items()}

    def cells(self):
        """Iterate over every cell in the maze, row by row."""
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def neighbors(self, cell):
        """Every in-bounds cell adjacent to ``cell`` (linked or not)."""
        x, y = cell
        out = []
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                out.append((nx, ny))
        return out

    def passages(self, cell):
        """Cells linked to ``cell`` — one step along an open corridor."""
        return self.links.get(cell, set())

    def edges(self):
        """Every open passage as a canonical, sorted ``(cell_a, cell_b)`` pair."""
        seen = set()
        for cell, neighbors in self.links.items():
            for other in neighbors:
                seen.add(tuple(sorted((cell, other))))
        return seen

    def edge_count(self):
        """Number of open passages in the maze."""
        return sum(len(neighbors) for neighbors in self.links.values()) // 2

    def is_connected(self):
        """True if every cell is reachable from every other cell."""
        start = next(iter(self.cells()), None)
        if start is None:
            return True
        seen = {start}
        queue = deque([start])
        while queue:
            for nxt in self.links.get(queue.popleft(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return len(seen) == self.width * self.height


def generate_maze(width: int, height: int, seed=None) -> Maze:
    """Carve a perfect maze of the given size with randomized depth-first search.

    The iterative "recursive backtracker": walk from cell to cell choosing a
    random unvisited neighbor until stuck, then back up. The result is a
    spanning tree over the grid, so there is exactly one corridor path between
    any two cells. Deterministic for a given ``seed``.
    """
    if width < 1 or height < 1:
        raise ValueError("maze dimensions must be >= 1")
    rng = random.Random(seed)
    links = {(x, y): set() for y in range(height) for x in range(width)}
    start = (0, 0)
    visited = {start}
    stack = [start]
    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in DIRECTIONS:
            nxt = (x + dx, y + dy)
            if 0 <= nxt[0] < width and 0 <= nxt[1] < height and nxt not in visited:
                candidates.append(nxt)
        if not candidates:
            stack.pop()
            continue
        nxt = rng.choice(candidates)
        links[(x, y)].add(nxt)
        links[nxt].add((x, y))
        visited.add(nxt)
        stack.append(nxt)
    return Maze(width, height, links)


def solve(maze: Maze, start=(0, 0), end=None):
    """Shortest path from ``start`` to ``end`` using breadth-first search.

    Returns the path as a list of cells starting with ``start`` and ending
    with ``end``, or ``None`` if no path exists. ``end`` defaults to the
    bottom-right cell.
    """
    if end is None:
        end = (maze.width - 1, maze.height - 1)
    for name, cell in (("start", start), ("end", end)):
        if not (0 <= cell[0] < maze.width and 0 <= cell[1] < maze.height):
            raise ValueError(f"{name} {cell} is outside the maze")
    if start == end:
        return [start]
    parent = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == end:
            path = []
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            path.reverse()
            return path
        for nxt in maze.passages(cell):
            if nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    return None


def render_ascii(maze: Maze, path=None, start=None, end=None) -> str:
    """Render the maze as ASCII art.

    ``+``, ``-`` and ``|`` are walls; a space in a wall slot is an open
    passage. Cell centers are blank unless markers are supplied: path cells
    as ``*``, then ``S`` for start and ``E`` for end (S/E win over ``*``).
    """
    width, height = maze.width, maze.height
    grid = [[" "] * (2 * width + 1) for _ in range(2 * height + 1)]
    # corners and the fully walled outer border
    for r in range(2 * height + 1):
        for c in range(2 * width + 1):
            if r % 2 == 0 and c % 2 == 0:
                grid[r][c] = "+"
    for c in range(1, 2 * width, 2):
        grid[0][c] = "-"
        grid[2 * height][c] = "-"
    for r in range(1, 2 * height, 2):
        grid[r][0] = "|"
        grid[r][2 * width] = "|"
    # interior horizontal walls (between a cell and the one below it)
    for r in range(2, 2 * height, 2):
        for c in range(1, 2 * width, 2):
            above = ((c - 1) // 2, r // 2 - 1)
            below = ((c - 1) // 2, r // 2)
            grid[r][c] = " " if below in maze.passages(above) else "-"
    # interior vertical walls (between a cell and the one to its right)
    for r in range(1, 2 * height, 2):
        for c in range(2, 2 * width, 2):
            left = (c // 2 - 1, r // 2)
            right = (c // 2, r // 2)
            grid[r][c] = " " if right in maze.passages(left) else "|"
    # markers
    if path:
        for cell in path:
            grid[2 * cell[1] + 1][2 * cell[0] + 1] = "*"
    if start is not None:
        grid[2 * start[1] + 1][2 * start[0] + 1] = "S"
    if end is not None and end != start:
        grid[2 * end[1] + 1][2 * end[0] + 1] = "E"
    return "\n".join("".join(row) for row in grid)


def stats(maze: Maze) -> dict:
    """Basic facts about a maze: cell count, passages, dead ends, junctions."""
    degrees = [len(maze.links.get(cell, ())) for cell in maze.cells()]
    return {
        "cells": maze.width * maze.height,
        "passages": maze.edge_count(),
        "dead_ends": sum(1 for d in degrees if d == 1),
        "junctions": sum(1 for d in degrees if d >= 3),
    }


def _parse_cell(text):
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected X,Y (got {text!r})")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected X,Y with integer coordinates (got {text!r})"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="maze.py",
        description="Generate, solve, and render perfect mazes as ASCII art.",
    )
    parser.add_argument(
        "-w", "--width", type=int, default=20, metavar="N",
        help="maze width in cells (default: 20)",
    )
    parser.add_argument(
        "-H", "--height", type=int, default=10, metavar="N",
        help="maze height in cells (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed; same seed gives the same maze",
    )
    parser.add_argument(
        "--solve", action="store_true",
        help="draw the shortest path between start and end",
    )
    parser.add_argument(
        "--start", type=_parse_cell, default=(0, 0), metavar="X,Y",
        help="start cell (default: 0,0)",
    )
    parser.add_argument(
        "--end", type=_parse_cell, default=None, metavar="X,Y",
        help="end cell (default: bottom-right corner)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="print maze statistics instead of the maze",
    )
    args = parser.parse_args(argv)

    if args.width < 1 or args.height < 1:
        parser.error("width and height must be >= 1")
    end = args.end if args.end is not None else (args.width - 1, args.height - 1)
    for name, cell in (("start", args.start), ("end", end)):
        if not (0 <= cell[0] < args.width and 0 <= cell[1] < args.height):
            parser.error(
                f"{name} {cell[0]},{cell[1]} is outside the "
                f"{args.width}x{args.height} maze"
            )

    maze = generate_maze(args.width, args.height, seed=args.seed)
    path = solve(maze, args.start, end)

    if args.stats:
        facts = stats(maze)
        print(f"size:      {args.width} x {args.height} ({facts['cells']} cells)")
        print(f"passages:  {facts['passages']}")
        print(f"dead ends: {facts['dead_ends']}")
        print(f"junctions: {facts['junctions']}")
        print(
            f"solve:     {len(path) - 1} steps from "
            f"{args.start[0]},{args.start[1]} to {end[0]},{end[1]}"
        )
    elif args.solve:
        print(render_ascii(maze, path=path, start=args.start, end=end))
    else:
        print(render_ascii(maze))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
