import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def resolution():
    return 0.05


@pytest.fixture(scope="session")
def corridor(resolution):
    """A straight corridor along x, centred at z = 1.0 m, half-width 0.5 m.

    Clearance is exactly 0.5 - |z - 1.0|, so expected values at any point are
    known in closed form and the ESDF gradient magnitude is 1.0 off-axis.
    """
    H, W = 60, 200
    origin = np.array([0.0, 0.0], dtype=np.float32)
    zz = (np.arange(H) * resolution)[:, None].repeat(W, 1)
    esdf = (0.5 - np.abs(zz - 1.0)).astype(np.float32)
    return esdf, origin


@pytest.fixture(scope="session")
def corridor_torch(corridor):
    esdf, origin = corridor
    return torch.from_numpy(esdf), torch.from_numpy(origin)


@pytest.fixture
def expert_traj():
    """A 25-waypoint expert hugging the corridor centre line."""
    xs = np.linspace(0.8, 2.3, 25).astype(np.float32)
    zs = np.full(25, 1.0, dtype=np.float32)
    return np.stack([xs, zs], axis=1)


@pytest.fixture
def expert_traj_torch(expert_traj):
    return torch.from_numpy(expert_traj).unsqueeze(0)
