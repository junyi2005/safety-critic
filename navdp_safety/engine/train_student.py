"""Stage 2: distil the teacher into the RGB-D selector (student).

Implements the selector loss of the paper:

    L_sel = E_{tau ~ D_mix} || P(tau) - stopgrad(V_phi(tau)) ||^2

where P(.) is the student's prediction from RGB-D observations plus trajectory
tokens, and D_mix mixes expert sub-trajectories with diffusion-generated
candidates sampled under the same RGB-D observations used at inference.

The teacher is frozen here: its scores are targets only, so no gradient flows
back into q_eta or the penalty weights. The student never sees the ESDF, which
is what lets the deployed selector run without map building.
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


@torch.no_grad()
def _diffusion_candidates(model, rgbd_embed, traj, k, noise_scale):
    """Sample candidate trajectories under the same observation.

    Uses the policy's own diffusion head when available so that D_mix matches
    the inference-time candidate distribution; falls back to perturbing the
    expert if the head is not exposed.
    """
    sampler = getattr(model, 'sample_trajectories', None)
    if callable(sampler):
        return sampler(rgbd_embed, sample_num=k)
    base = traj.unsqueeze(1).expand(-1, k, -1, -1)
    return base + noise_scale * torch.randn_like(base)


def train_student(model, scorer, dataset, device, args, writer,
                  esdf, origin, resolution, checkpoint_path="student.ckpt"):
    """Regress the student selector onto frozen teacher scores.

    Args:
        model: NavDP_Policy_DPT providing rgbd_encoder(.) and predict_critic(.).
        scorer: a *trained* FusedTrajectoryScorer; frozen in this stage.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.student_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.student_epochs)
    mse = nn.MSELoss()

    scorer.to(device).eval()
    for p in scorer.parameters():
        p.requires_grad_(False)
    model.to(device).train()

    step = 0
    for epoch in range(args.student_epochs):
        total, count = 0.0, 0

        for batch in tqdm(loader, desc=f"[Student {epoch + 1}/{args.student_epochs}]"):
            rgb, depth, _, traj = (batch[0].to(device), batch[1].to(device),
                                   batch[2], batch[3].to(device).float())
            B = traj.shape[0]

            rgbd_embed = model.rgbd_encoder(rgb, depth)

            # D_mix = expert sub-trajectory + K diffusion candidates.
            cand = _diffusion_candidates(model, rgbd_embed, traj, args.k_candidates,
                                         args.candidate_noise)          # [B, K, T+1, 3]
            mixed = torch.cat([traj.unsqueeze(1), cand], dim=1)          # [B, K+1, T+1, 3]
            K1 = mixed.shape[1]
            flat = mixed.reshape(B * K1, *mixed.shape[2:])

            # Teacher targets: privileged, frozen, stop-gradient.
            with torch.no_grad():
                target = scorer(flat, esdf, origin, resolution)          # [B*(K+1)]

            # Student: RGB-D + trajectory tokens, no ESDF. Trajectories are
            # made relative to their own first waypoint, matching inference.
            rel = flat - flat[:, :1, :]
            embed = rgbd_embed.repeat_interleave(K1, dim=0)
            pred = model.predict_critic(rel, embed)                      # [B*(K+1)]

            loss = mse(pred, target)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total += loss.item() * B
            count += B

            if writer is not None and step % 10 == 0:
                writer.add_scalar('student/L_sel', loss.item(), step)
                with torch.no_grad():
                    # Does the student rank the expert above its candidates?
                    p2 = pred.view(B, K1)
                    writer.add_scalar('student/expert_is_argmax',
                                      (p2.argmax(dim=1) == 0).float().mean().item(), step)
            step += 1

        scheduler.step()
        avg = total / max(count, 1)
        print(f"[Student] epoch {epoch + 1}/{args.student_epochs}  L_sel={avg:.6f}")
        if writer is not None:
            writer.add_scalar('student/L_sel_epoch', avg, epoch)
        torch.save(model.state_dict(), checkpoint_path)

    return model
