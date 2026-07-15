"""The A* non-expert proposal q(tau | p_R, p_G, ESDF)."""

import numpy as np

from navdp_safety.data.negatives import resample_path, sample_negative


def test_negative_matches_start_and_goal(corridor, resolution, expert_traj):
    esdf, origin = corridor
    neg = sample_negative(expert_traj, esdf, origin, resolution,
                          d_safe=0.1, rng=np.random.default_rng(0))
    assert neg is not None
    assert neg.shape == expert_traj.shape
    assert np.allclose(neg[0], expert_traj[0], atol=0.05)
    assert np.allclose(neg[-1], expert_traj[-1], atol=0.05)


def test_negative_respects_d_safe(corridor, resolution, expert_traj):
    esdf, origin = corridor
    neg = sample_negative(expert_traj, esdf, origin, resolution,
                          d_safe=0.1, rng=np.random.default_rng(0))
    zi = np.clip(((neg[:, 1] - origin[1]) / resolution).round().astype(int), 0, esdf.shape[0] - 1)
    xi = np.clip(((neg[:, 0] - origin[0]) / resolution).round().astype(int), 0, esdf.shape[1] - 1)
    assert (esdf[zi, xi] >= 0.1).all()


def test_randomized_costs_give_diverse_negatives(corridor, resolution, expert_traj):
    esdf, origin = corridor
    a = sample_negative(expert_traj, esdf, origin, resolution, 0.1, np.random.default_rng(0))
    b = sample_negative(expert_traj, esdf, origin, resolution, 0.1, np.random.default_rng(7))
    assert a is not None and b is not None
    assert not np.allclose(a, b)


def test_unreachable_goal_returns_none(corridor, resolution, expert_traj):
    """No admissible path must yield None, not a silently invalid trajectory."""
    esdf, origin = corridor
    walled = esdf.copy()
    # The expert runs x = 0.8 -> 2.3 m, i.e. columns 16 -> 46 at 0.05 m/cell.
    # Seal the corridor at x ~ 1.5 m, between start and goal.
    walled[:, 30:34] = -1.0
    assert sample_negative(expert_traj, walled, origin, resolution,
                           d_safe=0.1, rng=np.random.default_rng(0)) is None


def test_resample_path_preserves_endpoints_and_length():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    out = resample_path(pts, 9)
    assert out.shape == (9, 2)
    assert np.allclose(out[0], pts[0]) and np.allclose(out[-1], pts[-1])
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert np.allclose(seg, seg[0], atol=1e-5)     # uniform arc-length spacing


def test_degenerate_single_point_path():
    pts = np.array([[1.0, 1.0]], dtype=np.float32)
    assert resample_path(pts, 5).shape == (5, 2)
