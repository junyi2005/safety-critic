"""The safety critic's terms, checked against hand-computed values."""

import numpy as np
import torch

from navdp_safety.data.esdf import query_esdf
from navdp_safety.models.scorer import FusedTrajectoryScorer


def test_query_esdf_matches_closed_form(corridor_torch, resolution):
    esdf, origin = corridor_torch
    pts = torch.tensor([[[1.0, 1.0], [1.0, 1.25], [1.0, 1.4]]])
    d, grad = query_esdf(esdf, origin, resolution, pts)
    assert torch.allclose(d[0], torch.tensor([0.5, 0.25, 0.1]), atol=2e-3)
    # |grad| is 1.0 anywhere off the ridge line.
    assert torch.allclose(grad[0][1:], torch.tensor([1.0, 1.0]), atol=1e-2)


def test_dmin_never_below_d_safe(corridor_torch, resolution, expert_traj_torch):
    esdf, origin = corridor_torch
    sc = FusedTrajectoryScorer(d_safe=0.1)
    _, parts = sc(expert_traj_torch, esdf, origin, resolution, return_parts=True)
    assert (parts['dmin'] >= 0.1 - 1e-6).all()


def test_dmin_starts_just_above_the_floor(corridor_torch, resolution, expert_traj_torch):
    """Default init must not put the budget above achievable clearance."""
    esdf, origin = corridor_torch
    sc = FusedTrajectoryScorer(d_safe=0.1, init_margin=0.15).eval()
    _, parts = sc(expert_traj_torch, esdf, origin, resolution, return_parts=True)
    assert abs(parts['dmin'].mean().item() - 0.25) < 0.03
    # A centre-line expert in a 0.5 m corridor starts fully safe.
    assert parts['unsafe_count'].item() == 0.0


def test_dmin_is_context_conditioned_not_time_only():
    """Same step index, different clearance -> different budget."""
    sc = FusedTrajectoryScorer()
    g = torch.ones(1, 8)
    tight = sc.compute_dmin(torch.full((1, 8), 0.15), g, 7)
    open_ = sc.compute_dmin(torch.full((1, 8), 0.48), g, 7)
    assert not torch.allclose(tight, open_)


def test_weights_stay_positive_under_softplus():
    sc = FusedTrajectoryScorer()
    with torch.no_grad():
        sc.w_tilde.fill_(-50.0)
    assert (sc.weights > 0).all()
    assert sc.a > 0


def test_cbf_residual_hand_computed():
    sc = FusedTrajectoryScorer(d_safe=0.1, rho=0.1)
    # h = 0.4, 0.1  ->  r = 0.9 * 0.4 - 0.1 = 0.26
    assert abs(sc._cbf_loss(torch.tensor([[0.5, 0.2]])).item() - 0.26) < 1e-5
    # Moving away from the obstacle leaves the residual inactive.
    assert sc._cbf_loss(torch.tensor([[0.2, 0.5]])).item() == 0.0


def test_detour_is_zero_for_a_straight_line(corridor_torch, resolution):
    sc = FusedTrajectoryScorer()
    n = 21
    straight = torch.tensor([[[x, 1.0] for x in np.linspace(1.0, 3.0, n)]], dtype=torch.float32)
    zig = straight.clone()
    zig[0, ::2, 1] += 0.25
    d, dmin = torch.full((1, n), 0.5), torch.full((1, n), 0.2)
    assert sc._gated_detour(straight, d, dmin).item() < 1e-4
    assert sc._gated_detour(zig, d, dmin).item() > sc._gated_detour(straight, d, dmin).item()


def test_smoothness_uses_positions_not_clearances():
    """Delta^2 is taken over pi(p_j); an earlier revision used ESDF values."""
    sc = FusedTrajectoryScorer()
    n = 21
    straight = torch.tensor([[[x, 1.0] for x in np.linspace(1.0, 3.0, n)]], dtype=torch.float32)
    zig = straight.clone()
    zig[0, ::2, 1] += 0.25
    assert sc._smoothness(straight).item() < 1e-4
    assert sc._smoothness(zig).item() > 1.0


def test_gradients_reach_margin_head_and_weights(corridor_torch, resolution, expert_traj_torch):
    esdf, origin = corridor_torch
    sc = FusedTrajectoryScorer().train()
    (-sc(expert_traj_torch, esdf, origin, resolution).mean()).backward()
    assert sc.margin_head.net[0].weight.grad.abs().sum() > 0
    assert sc.w_tilde.grad.abs().sum() > 0


def test_hard_indicator_in_eval_soft_surrogate_in_train():
    sc = FusedTrajectoryScorer()
    d, dmin = torch.tensor([[0.05, 0.5]]), torch.tensor([[0.2, 0.2]])
    sc.train()
    soft = sc._unsafe_count(d, dmin)
    sc.eval()
    assert sc._unsafe_count(d, dmin).item() == 1.0     # exact
    assert 0.5 < soft.item() < 1.5                     # differentiable, close


def test_discriminator_logit_shape(corridor_torch, resolution, expert_traj_torch):
    esdf, origin = corridor_torch
    sc = FusedTrajectoryScorer()
    assert sc.discriminator_logit(expert_traj_torch, esdf, origin, resolution).shape == (1,)
