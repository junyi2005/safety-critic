"""Stage 1: train the safety critic as a teacher with ESDF supervision.

Implements the adversarial logistic loss of the paper's selector-training
section:

    L_scr = -E_{tau ~ D_exp}[log D_phi(tau)]
            -E_{tau ~ q(.|p_R, p_G, ESDF)}[log(1 - D_phi(tau))]

with D_phi(tau) = sigmoid(-a C_phi(tau) + b), C_phi = -V_phi.

Negatives are drawn per batch from `navdp_safety.data.negatives`: A* with
randomized edge costs between the *same* start and goal as the expert, in the
same ESDF, rejecting anything that violates d < d_safe. They refresh every
epoch, so the teacher faces continually renewed challenging negatives rather
than a fixed set of handcrafted disturbances.

Only the critic is trained here (margin head q_eta, the softplus-reparameterized
penalty weights, and the calibration scalars a, b). The policy is untouched.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from ..data.esdf import esdf_gradient_map
from ..data.negatives import sample_negative


def train_teacher(scorer, dataset, device, args, writer,
                  esdf, origin, resolution, checkpoint_path="teacher.ckpt"):
    """Train V_phi via the adversarial discriminator.

    Args:
        scorer: FusedTrajectoryScorer.
        dataset: yields (rgb, depth, goal, traj, mask); only `traj` is used --
            the teacher sees privileged geometry only, never RGB-D.
        esdf, origin, resolution: the scene ESDF (torch tensors on `device`).
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.teacher_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.teacher_epochs)

    esdf_np = esdf.detach().cpu().numpy()
    origin_np = origin.detach().cpu().numpy()
    rng = np.random.default_rng(args.seed)

    scorer.to(device).train()
    step = 0
    for epoch in range(args.teacher_epochs):
        totals = {'loss': 0.0, 'n': 0, 'acc': 0.0, 'skipped': 0}

        for batch in tqdm(loader, desc=f"[Teacher {epoch + 1}/{args.teacher_epochs}]"):
            traj = batch[3].to(device).float()          # [B, T+1, >=2]

            # Fresh ESDF-conditioned negatives, matched start--goal.
            negs, keep = [], []
            for b in range(traj.shape[0]):
                neg = sample_negative(
                    traj[b].detach().cpu().numpy(), esdf_np, origin_np, resolution,
                    d_safe=scorer.d_safe, rng=rng, noise=args.negative_noise,
                )
                if neg is not None:
                    negs.append(torch.from_numpy(neg))
                    keep.append(b)

            if not negs:
                # A* found no admissible alternative for any expert in this
                # batch (e.g. a start/goal cell below d_safe). Skip rather than
                # fall back to a different negative distribution.
                totals['skipped'] += traj.shape[0]
                continue

            pos = traj[keep]
            neg = torch.stack(negs).to(device).float()

            logit_pos = scorer.discriminator_logit(pos, esdf, origin, resolution)
            logit_neg = scorer.discriminator_logit(neg, esdf, origin, resolution)

            loss = (F.binary_cross_entropy_with_logits(logit_pos, torch.ones_like(logit_pos))
                    + F.binary_cross_entropy_with_logits(logit_neg, torch.zeros_like(logit_neg)))

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(scorer.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                acc = 0.5 * ((logit_pos > 0).float().mean() + (logit_neg <= 0).float().mean())
            totals['loss'] += loss.item() * len(keep)
            totals['acc'] += acc.item() * len(keep)
            totals['n'] += len(keep)

            if writer is not None and step % 10 == 0:
                beta, lambda_cbf, mu, psi = scorer.weights.detach().unbind()
                writer.add_scalar('teacher/L_scr', loss.item(), step)
                writer.add_scalar('teacher/disc_acc', acc.item(), step)
                writer.add_scalar('teacher/beta', beta.item(), step)
                writer.add_scalar('teacher/lambda_cbf', lambda_cbf.item(), step)
                writer.add_scalar('teacher/mu', mu.item(), step)
                writer.add_scalar('teacher/psi', psi.item(), step)
                writer.add_scalar('teacher/a', scorer.a.item(), step)
                writer.add_scalar('teacher/b', scorer.b.item(), step)
            step += 1

        scheduler.step()
        n = max(totals['n'], 1)
        msg = (f"[Teacher] epoch {epoch + 1}/{args.teacher_epochs}  "
               f"L_scr={totals['loss'] / n:.4f}  disc_acc={totals['acc'] / n:.3f}")
        if totals['skipped']:
            msg += f"  (skipped {totals['skipped']} samples with no admissible negative)"
        print(msg)

        if writer is not None:
            with torch.no_grad():
                d_dummy = torch.full((1, 8), 0.4, device=device)
                dmin = scorer.compute_dmin(d_dummy, torch.ones_like(d_dummy), 7)
            writer.add_scalar('teacher/dmin_at_clearance_0.4', dmin.mean().item(), epoch)

        torch.save(scorer.state_dict(), checkpoint_path)

    return scorer
