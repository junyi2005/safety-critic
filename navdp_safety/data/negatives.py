"""ESDF-conditioned non-expert trajectory proposals.

Implements the proposal distribution q(tau | p_R, p_G, ESDF) of the paper's
teacher-training section: between the *same* start and goal as an expert
sub-trajectory, run A* with randomized edge costs, reject any path violating
d < d_safe, and resample to length T.

The proposal depends only on global geometry, so negatives stay matched to the
expert's start--goal context and refresh every epoch instead of being fixed
handcrafted disturbances.
"""

import heapq

import numpy as np

# 8-connected grid moves and their base step costs (in cells)
_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


def world_to_cell(xz, origin, resolution):
    x = int(round((xz[0] - origin[0]) / resolution))
    z = int(round((xz[1] - origin[1]) / resolution))
    return z, x  # esdf is indexed [z, x]


def cell_to_world(cell, origin, resolution):
    z, x = cell
    return np.array([origin[0] + x * resolution, origin[1] + z * resolution])


def astar_randomized(esdf, start_cell, goal_cell, d_safe, rng, noise=0.6):
    """A* on the ESDF grid with multiplicative random edge costs.

    Each edge cost is scaled by 1 + noise * U(0, 1), drawn fresh per call, so
    repeated calls between the same endpoints yield diverse suboptimal paths.
    Cells with clearance below d_safe are not traversable.

    Returns a list of (z, x) cells, or None if no path exists.
    """
    H, W = esdf.shape

    def passable(c):
        return 0 <= c[0] < H and 0 <= c[1] < W and esdf[c[0], c[1]] >= d_safe

    if not passable(start_cell) or not passable(goal_cell):
        return None

    def heuristic(c):
        return float(np.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1]))

    open_heap = [(heuristic(start_cell), 0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    closed = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal_cell:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)

        for dz, dx, base in _NEIGHBOURS:
            nxt = (cur[0] + dz, cur[1] + dx)
            if nxt in closed or not passable(nxt):
                continue
            step = base * (1.0 + noise * rng.random())
            ng = g + step
            if ng < g_score.get(nxt, np.inf):
                g_score[nxt] = ng
                came_from[nxt] = cur
                heapq.heappush(open_heap, (ng + heuristic(nxt), ng, nxt))
    return None


def resample_path(points, n):
    """Resample a polyline to exactly n points by arc length."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-8:
        return np.repeat(pts[:1], n, axis=0)
    target = np.linspace(0.0, s[-1], n)
    out = np.stack([np.interp(target, s, pts[:, i]) for i in range(pts.shape[1])], axis=1)
    return out.astype(np.float32)


def sample_negative(expert_traj, esdf, origin, resolution, d_safe=0.1,
                    rng=None, noise=0.6, max_tries=4):
    """Draw one non-expert trajectory matched to an expert's start and goal.

    Args:
        expert_traj: [T+1, >=2] world-frame expert waypoints.
        esdf: [H, W] numpy signed ESDF.
        origin: [2] world (x, z) of cell [0, 0].
        d_safe: clearance floor; candidate paths violating it are rejected.
    Returns:
        [T+1, 2] world-frame (x, z), or None if no valid path was found.
    """
    rng = rng or np.random.default_rng()
    expert = np.asarray(expert_traj)[:, :2]
    n = len(expert)

    start = world_to_cell(expert[0], origin, resolution)
    goal = world_to_cell(expert[-1], origin, resolution)

    for _ in range(max_tries):
        cells = astar_randomized(esdf, start, goal, d_safe, rng, noise=noise)
        if cells is None:
            continue
        world = np.stack([cell_to_world(c, origin, resolution) for c in cells])
        traj = resample_path(world, n)

        # Reject any candidate that violates the safety floor.
        zi = np.clip(((traj[:, 1] - origin[1]) / resolution).round().astype(int), 0, esdf.shape[0] - 1)
        xi = np.clip(((traj[:, 0] - origin[0]) / resolution).round().astype(int), 0, esdf.shape[1] - 1)
        if np.all(esdf[zi, xi] >= d_safe):
            return traj
    return None
