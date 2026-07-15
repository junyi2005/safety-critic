import os

import numpy as np
import torch
from scipy import ndimage


def build_esdf(scene_path, voxel_size=0.05, margin=1.0):
    # Imported lazily so that the rest of navdp_safety.data can be used without
    # a habitat-sim installation (habitat-sim is only needed to build an ESDF).
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    backend_cfg = habitat_sim.Configuration(sim_cfg, [])
    sim = habitat_sim.Simulator(backend_cfg)
    pathfinder = sim.pathfinder

    if not pathfinder.is_loaded:
        pathfinder.load_nav_mesh(scene_path)

    bounds = pathfinder.get_bounds()
    min_bounds = np.array([bounds[0][0], bounds[0][2]]) - margin
    max_bounds = np.array([bounds[1][0], bounds[1][2]]) + margin
    map_range = max_bounds - min_bounds
    map_size = np.ceil(map_range / voxel_size).astype(int)  # [x, z]
    origin = min_bounds
    occupancy = np.zeros(map_size[::-1], dtype=np.uint8)  # [z, x]

    # === Build the 2D occupancy map ===
    for x in range(map_size[0]):
        for z in range(map_size[1]):
            pos = origin + np.array([x, z]) * voxel_size
            nav_pos = np.array([pos[0], 0.0, pos[1]])  # [x, y, z]
            if not pathfinder.is_navigable(nav_pos):
                occupancy[z, x] = 1  # non-traversable region

    # === Build the signed ESDF ===
    outside = 1 - occupancy
    inside = occupancy
    dist_out = ndimage.distance_transform_edt(outside) * voxel_size
    dist_in = ndimage.distance_transform_edt(inside) * voxel_size
    signed_esdf = dist_out - dist_in  # negative values mean close to an obstacle

    print(f"[Info] ESDF size: {signed_esdf.shape}, voxel_size: {voxel_size:.3f}")
    print(f"[Info] Occupancy rate: {occupancy.mean()*100:.2f}% occupied")
    print(f"[Info] ESDF range: min={signed_esdf.min():.2f}, max={signed_esdf.max():.2f}")

    return signed_esdf, occupancy, origin, voxel_size


def visualize_esdf(esdf, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plt.imshow(esdf, cmap='jet', origin='lower')
    plt.title("ESDF Map")
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, "esdf_vis.png"))
    plt.close()


# =========================
# ESDF loading
# =========================
def load_esdf_from_npz(npz_path, device):
    data = np.load(npz_path)
    esdf = torch.tensor(data['esdf'], dtype=torch.float32, device=device)
    origin = torch.tensor(data['origin'], dtype=torch.float32, device=device)
    resolution = float(data['resolution'])
    return esdf, origin, resolution


# =========================
# Differentiable ESDF query
# =========================
def esdf_gradient_map(esdf, resolution):
    """Finite-difference gradient magnitude ||grad d|| of an ESDF.

    esdf is indexed [z, x] (see build_esdf), so dim 0 is z and dim 1 is x.
    Returns a [H, W] map in metres per metre.
    """
    gz, gx = torch.gradient(esdf, spacing=(resolution, resolution))
    return torch.sqrt(gz ** 2 + gx ** 2)


def query_esdf(esdf, origin, resolution, xz, grad_map=None):
    """Bilinearly sample clearance and ESDF gradient magnitude at waypoints.

    Bilinear sampling (rather than the nearest-cell integer indexing used in
    earlier revisions) keeps d_j differentiable w.r.t. the waypoint
    positions, which the margin head and the safety gate both rely on.

    Args:
        esdf: [H, W] signed ESDF, indexed [z, x].
        origin: [2] world (x, z) of cell [0, 0].
        resolution: metres per cell.
        xz: [B, N, 2] world-frame (x, z) positions.
        grad_map: optional precomputed [H, W] gradient-magnitude map. Pass
            one to avoid recomputing it per call in a training loop.
    Returns:
        (d, grad_mag), each [B, N]. Positions outside the map clamp to the
        border value rather than erroring.
    """
    if grad_map is None:
        grad_map = esdf_gradient_map(esdf, resolution)

    H, W = esdf.shape
    B, N, _ = xz.shape

    # world -> continuous cell index
    idx = (xz - origin.to(xz)) / resolution        # [B, N, 2] as (x_idx, z_idx)
    x_idx, z_idx = idx[..., 0], idx[..., 1]

    # cell index -> normalized grid_sample coords in [-1, 1]
    x_n = 2.0 * x_idx / max(W - 1, 1) - 1.0
    z_n = 2.0 * z_idx / max(H - 1, 1) - 1.0
    grid = torch.stack([x_n, z_n], dim=-1).view(B, N, 1, 2)

    stack = torch.stack([esdf, grad_map], dim=0).unsqueeze(0)   # [1, 2, H, W]
    stack = stack.expand(B, -1, -1, -1).to(xz.dtype)

    out = torch.nn.functional.grid_sample(
        stack, grid, mode='bilinear', padding_mode='border', align_corners=True,
    )                                                            # [B, 2, N, 1]
    out = out.squeeze(-1)
    return out[:, 0], out[:, 1]
