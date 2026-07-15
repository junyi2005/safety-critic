import os
import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Normalization scale kept consistent with the official inference path
# (cumsum(naction / 4.0)).
ACTION_SCALE = (1.0, 1.0, 1.0)  # applied to [x, z, w(yaw)] respectively


# =========================
# Angle utilities
# =========================
def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    # Wrap an arbitrary angle into (-pi, pi].
    return torch.atan2(torch.sin(a), torch.cos(a))


def angle_diff(a2: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    # Shortest angular difference a2 - a1, result in (-pi, pi].
    return torch.atan2(torch.sin(a2 - a1), torch.cos(a2 - a1))


# =========================
# Dataset: emits x-z-w(yaw)
# =========================
class NavDPDataset(Dataset):
    def __init__(self, scene_path, mode='diffusion', max_traj_len=24):
        self.mode = mode
        self.max_traj_len = max_traj_len

        self.rgb_pkl_path = os.path.join(scene_path, "rgb.pkl")
        self.depth_pkl_path = os.path.join(scene_path, "depth.pkl")
        self.traj_pkl_path = os.path.join(scene_path, "traj.pkl")
        with open(self.rgb_pkl_path, 'rb') as f:
            self.rgb_data = pickle.load(f)
        with open(self.depth_pkl_path, 'rb') as f:
            self.depth_data = pickle.load(f)
        with open(self.traj_pkl_path, 'rb') as f:
            self.traj_data = pickle.load(f)

        self.samples = []
        episode_set = set([k.split('_step')[0] for k in self.traj_data.keys()])
        for ep_key in episode_set:
            traj_len = len([k for k in self.traj_data if k.startswith(ep_key)])
            if traj_len > self.max_traj_len + 10:
                max_start = traj_len - self.max_traj_len
                # Draw a few random start indices.
                start_indices = random.sample(range(3, max_start), 4)
                for start_idx in start_indices:
                    self.samples.append((ep_key, start_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_key, start_idx = self.samples[idx]
        xzw_list, rgb_seq, depth_seq = [], [], []

        # Pull out the poses for the whole clip first, so yaw can be recovered if needed.
        raw_poses = []
        for i in range(self.max_traj_len):
            step_key = f"{ep_key}_step{start_idx + i:02d}"
            raw_poses.append(self.traj_data[step_key].astype(np.float32))

        # Build (x, z, w) frame by frame.
        for i, pose in enumerate(raw_poses):
            if pose.shape[-1] >= 4:
                x, z, w = pose[0], pose[2], pose[3]  # [x,y,z,yaw] -> [x,z,yaw]
            elif pose.shape[-1] == 3:
                # Legacy data: no yaw available, so simply set it to 0 here
                # (could be replaced by a tangent-direction estimate if needed).
                x, z, w = pose[0], pose[2], 0.0
            else:
                # Extreme fallback.
                pad = np.zeros(4, dtype=np.float32)
                pad[:pose.shape[-1]] = pose
                x, z, w = pad[0], pad[2], 0.0
            xzw_list.append(np.array([x, z, w], dtype=np.float32))

            step_key = f"{ep_key}_step{start_idx + i:02d}"
            rgb = self.rgb_data[step_key].astype(np.float32) / 255.0
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = np.transpose(rgb, (2, 0, 1))  # [3,224,224]
            rgb_seq.append(rgb)

            depth = self.depth_data[step_key].astype(np.float32)
            depth = np.clip(depth, 0.1, 3.0)
            depth = (depth - 0.1) / (3.0 - 0.1)                   # -> [0,1]
            depth = cv2.resize(depth, (224, 224), interpolation=cv2.INTER_NEAREST)
            depth = depth[np.newaxis, :, :].astype(np.float32)    # [1,224,224]
            depth_seq.append(depth)

        traj_xzw = np.stack(xzw_list, axis=0)   # [T,3] = [x,z,w]
        rgb_seq  = np.stack(rgb_seq, axis=0)
        depth_seq= np.stack(depth_seq, axis=0)
        goal     = traj_xzw[-1]                 # [3]

        rgb_first   = rgb_seq[0]
        depth_first = depth_seq[0]
        return rgb_first, depth_first, goal, traj_xzw, 0.0


# =========================
# Relative-increment construction (x, z, w)
# =========================
def traj_to_deltas_xzw(pos_xzw: torch.Tensor) -> torch.Tensor:
    """
    pos_xzw: [B,T,3] absolute [x, z, w(yaw)]
    return : [B,T,3] step-wise increments, t=0 set to 0; the angle uses a wrapped difference
    """
    dx = pos_xzw[:, 1:, 0] - pos_xzw[:, :-1, 0]
    dz = pos_xzw[:, 1:, 1] - pos_xzw[:, :-1, 1]
    dw = angle_diff(pos_xzw[:, 1:, 2], pos_xzw[:, :-1, 2])
    d = torch.stack([dx, dz, dw], dim=-1)  # [B,T-1,3]
    zero = torch.zeros(pos_xzw.size(0), 1, 3, device=pos_xzw.device, dtype=pos_xzw.dtype)
    return torch.cat([zero, d], dim=1)     # [B,T,3]


def normalize_deltas(dlt: torch.Tensor, scale_vec: torch.Tensor) -> torch.Tensor:
    return dlt / scale_vec.view(1, 1, -1)
