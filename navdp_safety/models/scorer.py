"""Learnable safety critic V_phi with a context-conditioned margin head.

Implements Sec. "Learnable Safety Critic" of the paper:

    d_min,j    = d_safe + softplus(q_eta(f_j))                        (eq. dmin_context)
    h(p)       = d(p) - d_safe
    r_cbf_j    = (1 - rho) h(p_j) - h(p_{j+1})
    L_cbf      = sum_j [r_cbf_j]_+
    V_safe     = -sum_j I(d_j < d_min,j) - lambda_cbf * L_cbf         (eq. v_safe)
    L_detour   = [L_path / (D_chord + eps) - 1]_+                     (eq. detour_ratio_loss)
    w_j        = sigmoid(kappa * (d_j - d_min,j))
    L_detour^g = mean_j(w_j) * L_detour
    V_eff      = -beta * sum_j ||Delta^2 pi(p_j)|| - mu * L_detour^g  (eq. v_efficient)
    V_balance  = -psi * sum_j (d_j - d_min,j)^2                       (eq. v_balance)
    V_ours     = V_safe + V_efficient + V_balance                     (eq. v_ours_sum)

Nonnegative penalty weights come from w = softplus(w_tilde) with
w = [beta, lambda_cbf, mu, psi].

The teacher is trained through an affine-calibrated discriminator

    C_phi(tau) = -V_phi(tau) >= 0
    D_phi(tau) = sigmoid(-a * C_phi(tau) + b),  a = softplus(a_hat) > 0

See `navdp_safety.engine.train_teacher` for the adversarial logistic loss L_scr.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.esdf import query_esdf


class MarginHead(nn.Module):
    """q_eta: 2-layer MLP predicting a per-waypoint clearance budget.

    The input feature f_j is deliberately lightweight and geometry-only:
    local clearance d_j, finite-difference ESDF gradient magnitude, and
    normalized step index j / T.
    """

    IN_DIM = 3

    def __init__(self, hidden_dim=64, init_margin=0.15, out_gain=0.1):
        """
        Args:
            init_margin: initial value of softplus(q_eta(.)) in metres, i.e.
                the budget starts at d_safe + init_margin before any context
                dependence is learned. With a default-initialized output layer
                softplus(0) ~ 0.69 m, which exceeds the clearance available in
                ordinary indoor corridors -- every waypoint would count as
                unsafe from step 0 and the critic would start saturated. The
                output bias is set so the budget instead starts just above the
                physical floor.
            out_gain: shrinks the initial output weights so the starting budget
                is near-uniform and context dependence is learned rather than
                imposed by random init.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.IN_DIM, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        out = self.net[-1]
        with torch.no_grad():
            out.weight.mul_(out_gain)
            # softplus(bias) = init_margin  =>  bias = log(exp(init_margin) - 1)
            out.bias.fill_(math.log(math.expm1(max(init_margin, 1e-4))))

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class FusedTrajectoryScorer(nn.Module):
    """V_phi(tau): the learnable safety critic (teacher).

    Args:
        d_safe: fixed physical safety floor in metres (paper: 0.1).
        rho: CBF conservativeness in [0, 1) (paper: 0.1).
        kappa: sharpness of the detour safety gate.
        eps: numerical stabiliser in the detour ratio.
        soft_unsafe: if True, replace the non-differentiable indicator
            I(d_j < d_min,j) with sigmoid(kappa_unsafe * (d_min,j - d_j))
            while training. The hard indicator has zero gradient w.r.t.
            d_min almost everywhere, so q_eta cannot be trained through
            V_safe without a surrogate. Scoring in eval() mode always uses
            the exact hard count of the paper's definition.
        kappa_unsafe: sharpness of that surrogate.
    """

    def __init__(self,
                 d_safe=0.1,
                 rho=0.1,
                 kappa=10.0,
                 eps=1e-6,
                 hidden_dim=64,
                 init_margin=0.15,
                 soft_unsafe=True,
                 kappa_unsafe=20.0,
                 init_weights=(0.05, 0.10, 0.10, 0.05)):
        super().__init__()
        self.d_safe = float(d_safe)
        self.rho = float(rho)
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.soft_unsafe = bool(soft_unsafe)
        self.kappa_unsafe = float(kappa_unsafe)

        self.margin_head = MarginHead(hidden_dim=hidden_dim, init_margin=init_margin)

        # w = softplus(w_tilde), w = [beta, lambda_cbf, mu, psi]. Stored
        # unconstrained, so nonnegativity needs no clamping.
        w0 = torch.tensor(init_weights, dtype=torch.float32)
        self.w_tilde = nn.Parameter(torch.log(torch.expm1(w0.clamp(min=1e-4))))

        # Discriminator calibration: D_phi = sigmoid(-a * C_phi + b).
        self.a_hat = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    # ---- reparameterized quantities ----------------------------------
    @property
    def weights(self):
        """[beta, lambda_cbf, mu, psi], all > 0."""
        return F.softplus(self.w_tilde)

    @property
    def a(self):
        return F.softplus(self.a_hat)

    # ---- margin ------------------------------------------------------
    def compute_dmin(self, d, grad_mag, T):
        """d_min,j = d_safe + softplus(q_eta(f_j)).  d: [B, T+1] -> [B, T+1]."""
        steps = torch.arange(d.shape[-1], device=d.device, dtype=d.dtype)
        idx = (steps / max(T, 1)).expand_as(d)
        feats = torch.stack([d, grad_mag, idx], dim=-1)
        return self.d_safe + F.softplus(self.margin_head(feats))

    # ---- individual terms --------------------------------------------
    def _unsafe_count(self, d, dmin):
        if self.soft_unsafe and self.training:
            return torch.sigmoid(self.kappa_unsafe * (dmin - d)).sum(dim=-1)
        return (d < dmin).to(d.dtype).sum(dim=-1)

    def _cbf_loss(self, d):
        """L_cbf = sum_j [(1-rho) h(p_j) - h(p_{j+1})]_+ ; d: [B, T+1]."""
        h = d - self.d_safe
        r = (1.0 - self.rho) * h[:, :-1] - h[:, 1:]
        return F.relu(r).sum(dim=-1)

    def _smoothness(self, xz):
        """sum_j ||Delta^2 pi(p_j)||_2 over positions; xz: [B, T+1, 2]."""
        d2 = xz[:, 2:] - 2.0 * xz[:, 1:-1] + xz[:, :-2]
        return d2.norm(dim=-1).sum(dim=-1)

    def _gated_detour(self, xz, d, dmin):
        seg = (xz[:, 1:] - xz[:, :-1]).norm(dim=-1)
        l_path = seg.sum(dim=-1)
        d_chord = (xz[:, -1] - xz[:, 0]).norm(dim=-1)
        l_detour = F.relu(l_path / (d_chord + self.eps) - 1.0)
        w_bar = torch.sigmoid(self.kappa * (d - dmin)).mean(dim=-1)
        return w_bar * l_detour

    # ---- forward -----------------------------------------------------
    def forward(self, traj, esdf, origin, resolution, return_parts=False):
        """V_phi(tau).

        Args:
            traj: [B, T+1, >=2] world-frame waypoints; only (x, z) are used.
            esdf: [H, W] signed ESDF.
            origin: [2] world coordinate of esdf[0, 0].
            resolution: metres per cell.
        Returns:
            [B] trajectory values (higher is better), or (value, parts dict).
        """
        xz = traj[..., :2]
        T = xz.shape[1] - 1

        d, grad_mag = query_esdf(esdf, origin, resolution, xz)
        dmin = self.compute_dmin(d, grad_mag, T)

        beta, lambda_cbf, mu, psi = self.weights.unbind()

        unsafe = self._unsafe_count(d, dmin)
        l_cbf = self._cbf_loss(d)
        v_safe = -unsafe - lambda_cbf * l_cbf

        smooth = self._smoothness(xz)
        detour = self._gated_detour(xz, d, dmin)
        v_eff = -beta * smooth - mu * detour

        v_balance = -psi * ((d - dmin) ** 2).sum(dim=-1)

        v = v_safe + v_eff + v_balance
        if not return_parts:
            return v
        return v, {
            'v_safe': v_safe, 'v_efficient': v_eff, 'v_balance': v_balance,
            'unsafe_count': unsafe, 'l_cbf': l_cbf, 'smoothness': smooth,
            'l_detour_gated': detour, 'dmin': dmin, 'clearance': d,
        }

    # ---- discriminator ------------------------------------------------
    def discriminator_logit(self, traj, esdf, origin, resolution):
        """Logit of D_phi(tau) = sigmoid(-a * C_phi(tau) + b) = sigmoid(a * V_phi + b).

        Returned as a logit so the caller can use the numerically stable
        binary_cross_entropy_with_logits.
        """
        v = self.forward(traj, esdf, origin, resolution)
        return self.a * v + self.b
