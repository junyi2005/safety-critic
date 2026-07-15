import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from navdp_safety.data.dataset import (
    ACTION_SCALE,
    angle_diff,
    normalize_deltas,
    traj_to_deltas_xzw,
)
from navdp_safety.data.esdf import load_esdf_from_npz
from navdp_safety.data.negatives import sample_negative


# =========================
# Visualization
# =========================
def eval_visualization(gt_scores, pred_scores, tag='train'):
    gt = gt_scores.cpu().numpy()
    pred = pred_scores.cpu().numpy()
    plt.figure()
    plt.scatter(gt, pred, alpha=0.5)
    plt.xlabel('GT Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Critic Prediction vs Ground Truth ({tag})')
    plt.grid(True)
    plt.savefig(f'eval_{tag}_scatter.png')
    plt.close()


# =========================
# Diffusion training (Δ[x,z,w] / 4.0)
# =========================
def train_diffusion(model, dataset, device, args, writer):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.10 * total_steps)
    # Imported lazily so the teacher/student stages can run without transformers.
    from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scale_vec = torch.tensor(ACTION_SCALE, dtype=torch.float32, device=device)
    Tsch = model.noise_scheduler.config.num_train_timesteps
    alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(device)
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0

        for rgb, depth, goal, traj, _ in dataloader:
            rgb   = rgb.to(device)         # [B,3,224,224]
            depth = depth.to(device)       # [B,1,224,224]
            goal  = goal.to(device)        # [B,3] = [x,z,w]
            traj  = traj.to(device)        # [B,T,3] = [x,z,w]

            # === Target: normalized increments ===
            deltas  = traj_to_deltas_xzw(traj)          # [B,T,3]
            x0      = normalize_deltas(deltas, scale_vec)

            # === Add noise via the scheduler ===
            B = x0.size(0)
            timesteps = torch.randint(0, Tsch, (B,), device=device, dtype=torch.long)

            noise = torch.randn_like(x0)
            noisy = model.noise_scheduler.add_noise(x0, noise, timesteps)   # [B,T,3]

            # === Condition: goal relative to the start, normalized on the same scale ===
            start   = traj[:, 0, :]                     # [B,3]
            rel_goal = torch.empty_like(goal)
            rel_goal[:, 0] = goal[:, 0] - start[:, 0]   # Δx
            rel_goal[:, 1] = goal[:, 1] - start[:, 1]   # Δz
            rel_goal[:, 2] = angle_diff(goal[:, 2], start[:, 2])  # Δw (wrapped)
            rel_goal_n = rel_goal / scale_vec
            goal_embed = model.point_encoder(rel_goal_n).unsqueeze(1)       # [B,1,C]

            # === Image encoding ===
            rgbd_embed = model.rgbd_encoder(rgb, depth)                     # [B,S,C]

            # === Forward pass & loss ===
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.float16):
                pred_noise = model.predict_noise(noisy, timesteps, goal_embed, rgbd_embed)  # [B,T,3]

                alpha_bar = alphas_cumprod[timesteps].view(B, 1, 1)
                snr = alpha_bar / (1.0 - alpha_bar + 1e-8)                 # [B,1,1]
                gamma = 5.0
                w = torch.minimum(snr, torch.tensor(gamma, device=device)).view(B)  # [B]

                mse_per = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=(1, 2))  # [B]
                loss = (w * mse_per).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)    # both diffusion paths need this
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Mind the ordering: only step the LR scheduler when the AMP scaler did not
            # skip the optimizer step.
            if scaler.get_scale() >= prev_scale:
                scheduler.step()
            total_loss += float(loss.item())
            global_step += 1

        avg_loss = total_loss / max(1, len(dataloader))
        writer.add_scalar('Loss/Diffusion', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        print(f"[Diffusion-Δ(x,z,w)] Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}")



def evaluate_model(model, dataset, scorer, device, scene_path, max_batches=10, seed=0):
    """Measure how well the student's scores track the frozen teacher's.

    Reports Pearson/Spearman correlation between P(tau) and V_phi(tau), plus
    the pairwise accuracy of ranking an expert above an ESDF-conditioned
    non-expert drawn from the same proposal distribution used in teacher
    training (not a Gaussian perturbation, which is off-distribution for the
    trained teacher and inflates the number).
    """
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval(); scorer.eval()

    print("[ESDF] Loading the global ESDF...")
    esdf_tensor, origin_tensor, resolution = load_esdf_from_npz(scene_path, device)
    print("[ESDF] Load complete")

    esdf_np = esdf_tensor.detach().cpu().numpy()
    origin_np = origin_tensor.detach().cpu().numpy()
    rng = np.random.default_rng(seed)

    all_gt, all_pred = [], []
    scale_vec = torch.tensor(ACTION_SCALE, dtype=torch.float32, device=device)

    # Extra: track the pairwise "expert > non-expert" accuracy.
    pair_cnt, pair_correct = 0, 0

    with torch.no_grad():
        seen = 0
        for i, (rgb_batch, depth_batch, _, traj_batch, _) in tqdm(enumerate(dataloader), desc="[Evaluate]"):
            rgb_batch  = rgb_batch.to(device)
            depth_batch= depth_batch.to(device)
            traj_batch = traj_batch.to(device)        # [B,T,3]=[x,z,w]
            B, T, _ = traj_batch.shape

            # Extract features in one shot (faster).
            rgbd_embed = model.rgbd_encoder(rgb_batch, depth_batch)

            # Original ground truth and prediction.
            dlt   = traj_to_deltas_xzw(traj_batch)
            dlt_n = normalize_deltas(dlt, scale_vec)
            pred  = model.predict_critic(dlt_n, rgbd_embed)              # [B]
            gt    = scorer(traj_batch, esdf_tensor, origin_tensor, resolution)  # [B]

            all_gt.append(gt); all_pred.append(pred)

            # Extra: draw one non-expert per sample from the teacher-training
            # proposal q(.|p_R, p_G, ESDF) and check whether P(expert) > P(non-expert).
            negs, keep = [], []
            for b in range(B):
                neg = sample_negative(traj_batch[b].cpu().numpy(), esdf_np, origin_np,
                                      resolution, d_safe=scorer.d_safe, rng=rng)
                if neg is not None:
                    negs.append(torch.from_numpy(neg))
                    keep.append(b)

            if negs:
                bad = torch.zeros(len(keep), T, 3, device=device)
                bad[:, :, :2] = torch.stack(negs).to(device).float()
                bad[:, :, 2] = traj_batch[keep][:, :, 2]     # reuse expert heading
                dlt_bad   = traj_to_deltas_xzw(bad)
                dlt_bad_n = normalize_deltas(dlt_bad, scale_vec)
                pred_bad  = model.predict_critic(dlt_bad_n, rgbd_embed[keep])

                pair_cnt     += len(keep)
                pair_correct += (pred[keep] > pred_bad).sum().item()

            seen += 1
            if seen >= max_batches: break

    all_gt   = torch.cat(all_gt, dim=0).cpu().numpy()
    all_pred = torch.cat(all_pred, dim=0).cpu().numpy()

    # Scatter visualization.
    eval_visualization(torch.as_tensor(all_gt), torch.as_tensor(all_pred), tag='eval')

    # Correlations and ranking accuracy.
    pe_res = pearsonr(all_gt, all_pred)
    sp_res = spearmanr(all_gt, all_pred)
    pe = pe_res[0] if isinstance(pe_res, tuple) else pe_res.statistic
    sp = sp_res[0] if isinstance(sp_res, tuple) else sp_res.correlation
    acc = pair_correct / max(1, pair_cnt)

    print(f"[Evaluate] Pearson: {pe:.3f}, Spearman: {sp:.3f}, Pair-Acc(good>bad): {acc:.3f}")
