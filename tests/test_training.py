"""Both training stages, end to end on a synthetic corridor.

The policy is stubbed to the interface train_student() actually uses
(rgbd_encoder / predict_critic), so these run on CPU in seconds without
Depth-Anything weights.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from navdp_safety.data.negatives import sample_negative
from navdp_safety.engine.train_student import train_student
from navdp_safety.engine.train_teacher import train_teacher
from navdp_safety.models.scorer import FusedTrajectoryScorer


class _Experts(torch.utils.data.Dataset):
    """Experts running down the corridor centre line, where clearance is max."""

    def __init__(self, n=24):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x0 = 0.5 + 0.05 * i
        xs = np.linspace(x0, x0 + 1.5, 25).astype(np.float32)
        traj = np.stack([xs, np.full(25, 1.0, np.float32), np.zeros(25, np.float32)], 1)
        return (torch.rand(3, 224, 224), torch.rand(1, 224, 224),
                torch.zeros(3), torch.from_numpy(traj), torch.ones(25))


class _StubPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(12, 32)
        self.head = nn.Sequential(nn.Linear(32 + 75, 64), nn.ReLU(), nn.Linear(64, 1))

    def rgbd_encoder(self, rgb, depth):
        f = torch.stack([rgb.mean((2, 3)).mean(1), rgb.std((2, 3)).mean(1),
                         depth.mean((2, 3))[:, 0], depth.std((2, 3))[:, 0]], 1)
        return self.enc(torch.cat([f, f, f], 1))

    def predict_critic(self, rel, embed):
        return self.head(torch.cat([embed, rel.flatten(1)], 1))[:, 0]


class _Args:
    batch_size = 4
    seed = 0
    teacher_epochs = 6
    teacher_lr = 2e-2
    negative_noise = 0.8
    student_epochs = 8
    student_lr = 1e-3
    k_candidates = 3
    candidate_noise = 0.25


@pytest.fixture
def trained_teacher(corridor_torch, corridor, resolution, tmp_path):
    torch.manual_seed(0)
    esdf, origin = corridor_torch
    sc = FusedTrajectoryScorer(d_safe=0.1, init_margin=0.15)
    before = sc.weights.detach().clone()
    train_teacher(sc, _Experts(), torch.device('cpu'), _Args(), None,
                  esdf, origin, resolution, checkpoint_path=str(tmp_path / "t.ckpt"))
    return sc, before


def test_teacher_updates_its_parameters(trained_teacher):
    sc, before = trained_teacher
    assert not torch.allclose(sc.weights, before)
    assert (sc.weights > 0).all()


def test_teacher_separates_expert_from_non_expert(
        trained_teacher, corridor, corridor_torch, resolution, expert_traj, expert_traj_torch):
    sc, _ = trained_teacher
    sc.eval()
    esdf_np, origin_np = corridor
    esdf, origin = corridor_torch

    neg = sample_negative(expert_traj, esdf_np, origin_np, resolution,
                          d_safe=0.1, rng=np.random.default_rng(3))
    neg_t = torch.from_numpy(neg).unsqueeze(0)

    v_exp = sc(expert_traj_torch, esdf, origin, resolution)
    v_neg = sc(neg_t, esdf, origin, resolution)
    assert v_exp.item() > v_neg.item()

    d_exp = torch.sigmoid(sc.discriminator_logit(expert_traj_torch, esdf, origin, resolution))
    d_neg = torch.sigmoid(sc.discriminator_logit(neg_t, esdf, origin, resolution))
    assert d_exp.item() > 0.5 > d_neg.item()


def test_student_distils_and_teacher_stays_frozen(
        trained_teacher, corridor_torch, resolution, tmp_path):
    sc, _ = trained_teacher
    esdf, origin = corridor_torch

    epochs = []

    class _Writer:
        def add_scalar(self, k, v, t):
            if k == 'student/L_sel_epoch':
                epochs.append(v)

    torch.manual_seed(0)
    student = _StubPolicy()
    w_before = sc.weights.detach().clone()

    train_student(student, sc, _Experts(), torch.device('cpu'), _Args(), _Writer(),
                  esdf, origin, resolution, checkpoint_path=str(tmp_path / "s.ckpt"))

    assert len(epochs) == _Args.student_epochs
    assert epochs[-1] < epochs[0]                       # L_sel decreases
    assert not any(p.requires_grad for p in sc.parameters())
    assert torch.allclose(sc.weights, w_before)         # stopgrad held
