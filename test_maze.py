#!/usr/bin/env python3
"""Test suite for maze-generator.

Runs with:  python -m unittest test_maze.py -v

Covers the perfect-maze invariants (connectivity, acyclicity, exactly V-1
passages) across a sweep of sizes and seeds, render/parse round-trips,
shortest-path optimality cross-checked against an independent solver over the
rendered grid, determinism and distinctness, and the CLI end to end.
"""

import heapq
import subprocess
import sys
import unittest
from collections import deque
from pathlib import Path

from maze import Maze, generate_maze, render_ascii, solve, stats

HERE = Path(__file__).resolve().parent
MAZE_PY = HERE / "maze.py"


# ---------------------------------------------------------------------------
# helpers


def parse_render(text):
    """Reconstruct (width, height, open_edges) from rendered maze ASCII.

    Also asserts the frame is intact: corners are ``+``, and the outer border
    has no gaps.
    """
    rows = text.split("\n")
    height = (len(rows) - 1) // 2
    width = (len(rows[0]) - 1) // 2
    open_edges = set()
    for r, row in enumerate(rows):
        if len(row) != 2 * width + 1:
            raise AssertionError(
                f"row {r} is {len(row)} chars, expected {2 * width + 1}"
            )
        for c, ch in enumerate(row):
            if r % 2 == 0 and c % 2 == 0:
                if ch != "+":
                    raise AssertionError(f"expected corner '+' at {r},{c}, got {ch!r}")
            elif r % 2 == 0:  # horizontal wall slot
                if r in (0, 2 * height):
                    if ch != "-":
                        raise AssertionError(f"top/bottom border gap at {r},{c}")
                elif ch == " ":
                    open_edges.add(
                        tuple(sorted((((c - 1) // 2, r // 2 - 1), ((c - 1) // 2, r // 2))))
                    )
                elif ch != "-":
                    raise AssertionError(f"bad horizontal wall char at {r},{c}: {ch!r}")
            elif c % 2 == 0:  # vertical wall slot
                if c in (0, 2 * width):
                    if ch != "|":
                        raise AssertionError(f"left/right border gap at {r},{c}")
                elif ch == " ":
                    open_edges.add(
                        tuple(sorted(((c // 2 - 1, r // 2), (c // 2, r // 2))))
                    )
                elif ch != "|":
                    raise AssertionError(f"bad vertical wall char at {r},{c}: {ch!r}")
            # r odd, c odd: cell center; any marker is fine
    return width, height, open_edges


def reference_distance(open_edges, start, end):
    """Independent shortest-path length via Dijkstra over parsed open edges."""
    adj = {}
    for a, b in open_edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    dist = {start: 0}
    heap = [(0, start)]
    while heap:
        d, cell = heapq.heappop(heap)
        if d != dist.get(cell):
            continue
        if cell == end:
            return d
        for nxt in adj.get(cell, ()):
            nd = d + 1
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                heapq.heappush(heap, (nd, nxt))
    return None


def has_cycle(maze):
    """Union-find walk over open passages; True if any edge closes a loop."""
    parent = {cell: cell for cell in maze.cells()}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in maze.edges():
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# generation


class TestGeneration(unittest.TestCase):
    def test_dimension_validation(self):
        for bad in ((0, 5), (5, 0), (-1, 3), (3, -2)):
            with self.assertRaises(ValueError):
                generate_maze(*bad)

    def test_single_cell(self):
        m = generate_maze(1, 1, seed=1)
        self.assertEqual(m.edge_count(), 0)
        self.assertEqual(len(m.links), 1)
        self.assertTrue(m.is_connected())
        self.assertEqual(solve(m), [(0, 0)])

    def test_links_are_valid_sweep(self):
        for w in range(1, 7):
            for h in range(1, 7):
                m = generate_maze(w, h, seed=w * 10000 + h)
                with self.subTest(w=w, h=h):
                    self.assertEqual(len(m.links), w * h)
                    for cell, neighbors in m.links.items():
                        self.assertNotIn(cell, neighbors)  # no self-loops
                        for n in neighbors:
                            self.assertIn(n, m.links)  # in bounds
                            self.assertIn(cell, m.links[n])  # symmetric
                            self.assertIn(n, m.neighbors(cell))  # adjacent

    def test_perfect_maze_sweep(self):
        # Sweep every size 1..6 x 1..6 against 4 seeds each. A perfect maze is
        # a spanning tree: fully connected, acyclic, exactly V-1 passages.
        # (Seeds are combined into one int: random.Random rejects tuples.)
        for w in range(1, 7):
            for h in range(1, 7):
                for s in range(4):
                    m = generate_maze(w, h, seed=w * 10000 + h * 100 + s)
                    with self.subTest(w=w, h=h, s=s):
                        self.assertEqual(m.edge_count(), w * h - 1)
                        self.assertTrue(m.is_connected())
                        self.assertFalse(has_cycle(m))

    def test_solve_paths_sweep(self):
        for w in range(1, 7):
            for h in range(1, 7):
                for s in range(3):
                    m = generate_maze(w, h, seed=w * 1000 + h * 10 + s)
                    start, end = (0, 0), (w - 1, h - 1)
                    path = solve(m, start, end)
                    with self.subTest(w=w, h=h, s=s):
                        self.assertIsNotNone(path)
                        self.assertEqual(path[0], start)
                        self.assertEqual(path[-1], end)
                        self.assertEqual(len(path), len(set(path)))  # no repeats
                        for a, b in zip(path, path[1:]):
                            self.assertIn(b, m.passages(a))  # each step is open
                        self.assertGreaterEqual(len(path) - 1, manhattan(start, end))

    def test_render_round_trip_and_optimality(self):
        # parse(render(m)) must recover exactly the maze's open passages, and
        # the BFS distance must match an independent Dijkstra over that parse.
        for w in range(1, 6):
            for h in range(1, 6):
                for s in range(3):
                    m = generate_maze(w, h, seed=w * 1000 + h * 10 + s)
                    text = render_ascii(m)
                    pw, ph, edges = parse_render(text)
                    start, end = (0, 0), (w - 1, h - 1)
                    path = solve(m, start, end)
                    ref = reference_distance(edges, start, end)
                    with self.subTest(w=w, h=h, s=s):
                        self.assertEqual((pw, ph), (w, h))
                        self.assertEqual(edges, m.edges())
                        self.assertEqual(len(path) - 1, ref)


# ---------------------------------------------------------------------------
# determinism


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_maze(self):
        a = generate_maze(9, 7, seed=1234)
        b = generate_maze(9, 7, seed=1234)
        self.assertEqual(a.edges(), b.edges())
        self.assertEqual(render_ascii(a), render_ascii(b))

    def test_different_seeds_differ(self):
        mazes = [generate_maze(8, 6, seed=s) for s in range(50, 62)]
        distinct = {frozenset(m.edges()) for m in mazes}
        self.assertGreaterEqual(len(distinct), 10)

    def test_tiny_maze_unique(self):
        # 1x1 has exactly one possible maze, so all seeds must agree — a guard
        # against "distinctness" that is really just noise.
        mazes = [generate_maze(1, 1, seed=s) for s in range(5)]
        self.assertEqual({frozenset(m.edges()) for m in mazes}, {frozenset()})


# ---------------------------------------------------------------------------
# rendering


class TestRender(unittest.TestCase):
    def test_two_cell_corridor(self):
        m = Maze(2, 1, {(0, 0): {(1, 0)}, (1, 0): {(0, 0)}})
        self.assertEqual(render_ascii(m), "+-+-+\n|   |\n+-+-+")
        self.assertEqual(
            render_ascii(m, path=[(0, 0), (1, 0)], start=(0, 0), end=(1, 0)),
            "+-+-+\n|S E|\n+-+-+",
        )

    def test_two_cell_closed(self):
        m = Maze(2, 1, {})
        self.assertEqual(render_ascii(m), "+-+-+\n| | |\n+-+-+")

    def test_vertical_corridor(self):
        m = Maze(1, 2, {(0, 0): {(0, 1)}, (0, 1): {(0, 0)}})
        self.assertEqual(render_ascii(m), "+-+\n| |\n+ +\n| |\n+-+")

    def test_single_cell_marker(self):
        m = Maze(1, 1, {(0, 0): set()})
        self.assertEqual(render_ascii(m, start=(0, 0), end=(0, 0)), "+-+\n|S|\n+-+")

    def test_solution_markers(self):
        w, h = 8, 6
        m = generate_maze(w, h, seed=5)
        path = solve(m, (0, 0), (w - 1, h - 1))
        text = render_ascii(m, path=path, start=(0, 0), end=(w - 1, h - 1))
        # wall structure survives the overlay
        pw, ph, edges = parse_render(text)
        self.assertEqual((pw, ph), (w, h))
        self.assertEqual(edges, m.edges())
        # exactly two markers plus one star per interior path cell
        self.assertEqual(text.count("S"), 1)
        self.assertEqual(text.count("E"), 1)
        self.assertEqual(text.count("*"), len(path) - 2)
        self.assertLessEqual(len(path) - 2 + 2, w * h)


# ---------------------------------------------------------------------------
# stats and errors


class TestStats(unittest.TestCase):
    def test_stats_counts(self):
        m = generate_maze(6, 4, seed=99)
        facts = stats(m)
        self.assertEqual(facts["cells"], 24)
        self.assertEqual(facts["passages"], 23)
        self.assertGreaterEqual(facts["dead_ends"], 2)  # a tree has >= 2 leaves
        self.assertGreaterEqual(facts["junctions"], 0)

    def test_corridor_dead_ends(self):
        m = Maze(1, 2, {(0, 0): {(0, 1)}, (0, 1): {(0, 0)}})
        self.assertEqual(stats(m)["dead_ends"], 2)
        self.assertEqual(stats(Maze(1, 1, {(0, 0): set()}))["dead_ends"], 0)


class TestSolveErrors(unittest.TestCase):
    def test_out_of_bounds(self):
        m = generate_maze(3, 3, seed=5)
        with self.assertRaises(ValueError):
            solve(m, (3, 0), (0, 0))
        with self.assertRaises(ValueError):
            solve(m, (0, 0), (-1, 2))

    def test_start_equals_end(self):
        m = generate_maze(4, 4, seed=6)
        self.assertEqual(solve(m, (2, 2), (2, 2)), [(2, 2)])

    def test_disconnected_returns_none(self):
        m = Maze(
            2,
            2,
            {
                (0, 0): {(1, 0)},
                (1, 0): {(0, 0)},
                (0, 1): set(),
                (1, 1): set(),
            },
        )
        self.assertFalse(m.is_connected())
        self.assertIsNone(solve(m, (0, 0), (1, 1)))


# ---------------------------------------------------------------------------
# CLI


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(MAZE_PY), *args],
        capture_output=True,
        text=True,
        timeout=90,
    )


class TestCLI(unittest.TestCase):
    def test_render_is_deterministic(self):
        first = run_cli("--width", "7", "--height", "4", "--seed", "3")
        second = run_cli("--width", "7", "--height", "4", "--seed", "3")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        w, h, edges = parse_render(first.stdout.rstrip("\n"))
        self.assertEqual((w, h), (7, 4))
        self.assertEqual(len(edges), 7 * 4 - 1)
        # defaults produce the same maze as the library
        m = generate_maze(7, 4, seed=3)
        self.assertEqual(edges, m.edges())

    def test_solve_cli(self):
        result = run_cli(
            "--width", "8", "--height", "6", "--seed", "5",
            "--solve", "--start", "0,0", "--end", "7,5",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        _w, _h, edges = parse_render(result.stdout.rstrip("\n"))
        self.assertEqual(len(edges), 8 * 6 - 1)
        self.assertIn("S", result.stdout)
        self.assertIn("E", result.stdout)
        m = generate_maze(8, 6, seed=5)
        path = solve(m, (0, 0), (7, 5))
        self.assertEqual(result.stdout.count("*"), len(path) - 2)

    def test_stats_cli(self):
        result = run_cli("--stats", "--width", "5", "--height", "5", "--seed", "11")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cells", result.stdout)
        self.assertIn("passages:  24", result.stdout)
        self.assertIn("dead ends", result.stdout)
        self.assertIn("steps", result.stdout)

    def test_bad_args_exit_2(self):
        result = run_cli("--width", "4", "--height", "4", "--start", "9,9")
        self.assertEqual(result.returncode, 2)
        result = run_cli("--width", "0")
        self.assertEqual(result.returncode, 2)
        result = run_cli("--start", "not-a-cell")
        self.assertEqual(result.returncode, 2)
        result = run_cli("--width", "4", "--end", "4,0")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
