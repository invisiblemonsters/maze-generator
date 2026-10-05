# maze-generator

A zero-dependency perfect-maze generator, solver, and ASCII renderer in pure Python. Randomized depth-first search (the classic "recursive backtracker") carves the maze, breadth-first search finds shortest paths, and everything prints as plain ASCII art — no terminal tricks, just characters. Standard library only.

Every generated maze is *perfect*: fully connected with exactly one corridor path between any two cells — no loops, no unreachable pockets.

## Features

- **Generate** perfect mazes of any size, deterministic with `--seed` (same seed, same maze)
- **Solve** a shortest path between any two cells — defaults to top-left → bottom-right
- **Render** as ASCII art with an optional solution overlay: `S` start, `E` end, `*` path
- **Stats** mode: passages, dead ends, junctions, and solution length

Requires Python 3.10+.

## Usage

Print a maze (default 20 x 10):

    python maze.py

Pick a size and seed:

    python maze.py --width 12 --height 6 --seed 7

Draw the shortest path between two cells:

    python maze.py --width 16 --height 8 --seed 42 --start 0,0 --end 15,7 --solve

Maze statistics:

    python maze.py --stats --width 12 --height 8 --seed 42

Example output (`python maze.py --width 12 --height 6 --seed 7`):

    +-+-+-+-+-+-+-+-+-+-+-+-+
    | |     |   |           |
    + +-+ + + + + +-+-+-+ + +
    |   | |   |       |   | |
    +-+ + +-+-+-+-+-+-+ +-+ +
    | | | |   |       | |   |
    + + +-+ + + +-+-+ + + +-+
    | |     |   |     | |   |
    + +-+-+-+-+-+ +-+-+ +-+-+
    |     |     |   |       |
    + + +-+ +-+ +-+ +-+-+-+ +
    | |       |             |
    +-+-+-+-+-+-+-+-+-+-+-+-+

## Tests

    python -m unittest test_maze.py -v

The suite covers the perfect-maze invariants (full connectivity, acyclicity, exactly `cells - 1` passages) across a sweep of sizes and seeds, render/parse round-trips, shortest-path optimality cross-checked against an independent solver over the rendered grid, determinism and distinctness, edge cases (1x1, corridors, disconnected input), and the CLI end to end.

## License

MIT
