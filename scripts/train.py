#!/usr/bin/env python
"""Train the NavDP diffusion generator and the adaptive safety critic.

Three stages, matching the paper:

  1. Diffusion generator   -- L_diff, RGB-D only, no ESDF.
  2. Teacher safety critic -- L_scr, adversarial, ESDF-supervised (offline).
  3. Student selector      -- L_sel, regresses frozen teacher scores from RGB-D.

Run stages selectively with --stages, e.g. --stages teacher,student.
"""
import argparse
import os

import torch
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from navdp_safety.data.dataset import NavDPDataset
from navdp_safety.data.esdf import load_esdf_from_npz
from navdp_safety.engine import (
    evaluate_model,
    train_diffusion,
    train_student,
    train_teacher,
)
from navdp_safety.models.policy import NavDP_Policy_DPT
from navdp_safety.models.scorer import FusedTrajectoryScorer


# =========================
# Arguments
# =========================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', type=str, default='diffusion,teacher,student',
                        help='Comma-separated subset of: diffusion, teacher, student.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)

    # Stage 1: diffusion generator
    parser.add_argument('--epochs', type=int, default=50, help='Diffusion epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Diffusion LR.')

    # Stage 2: teacher safety critic
    parser.add_argument('--teacher_epochs', type=int, default=30)
    parser.add_argument('--teacher_lr', type=float, default=1e-3)
    parser.add_argument('--negative_noise', type=float, default=0.6,
                        help='Edge-cost randomization for the A* non-expert proposal.')

    # Stage 3: student selector
    parser.add_argument('--student_epochs', type=int, default=30)
    parser.add_argument('--student_lr', type=float, default=1e-4)
    parser.add_argument('--k_candidates', type=int, default=16,
                        help='K diffusion candidates per observation in D_mix.')
    parser.add_argument('--candidate_noise', type=float, default=0.25,
                        help='Fallback perturbation scale when the policy exposes no sampler.')

    # Critic structure (paper defaults)
    parser.add_argument('--d_safe', type=float, default=0.1,
                        help='Fixed physical safety floor in metres.')
    parser.add_argument('--rho', type=float, default=0.1, help='CBF conservativeness.')
    parser.add_argument('--kappa', type=float, default=10.0, help='Safety-gate sharpness.')
    parser.add_argument('--init_margin', type=float, default=0.15,
                        help='Initial softplus(q_eta) budget above d_safe, in metres.')

    parser.add_argument('--esdf_npz', type=str, required=True,
                        help='Path to the scene ESDF .npz produced by scripts/build_esdf.py.')
    parser.add_argument('--data_root', type=str, default='.',
                        help='Directory containing the navdp_collected<i> scene folders.')
    parser.add_argument('--num_scenes', type=int, default=5,
                        help='Number of scenes to concatenate: navdp_collected1 .. navdp_collected<num_scenes>.')
    parser.add_argument('--out_dir', type=str, default='.')
    return parser.parse_args()


def main():
    args = get_args()
    stages = {s.strip() for s in args.stages.split(',') if s.strip()}
    unknown = stages - {'diffusion', 'teacher', 'student'}
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}")

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'runs/navdp'))

    scene_paths = [os.path.join(args.data_root, f"navdp_collected{i}")
                   for i in range(1, args.num_scenes + 1)]
    train_dataset = ConcatDataset([NavDPDataset(p) for p in scene_paths])

    esdf, origin, resolution = load_esdf_from_npz(args.esdf_npz, device)
    print(f"[ESDF] {tuple(esdf.shape)} @ {resolution} m/cell")

    model = NavDP_Policy_DPT(device=device).to(device)
    scorer = FusedTrajectoryScorer(
        d_safe=args.d_safe,
        rho=args.rho,
        kappa=args.kappa,
        init_margin=args.init_margin,
    ).to(device)

    policy_ckpt = os.path.join(args.out_dir, 'navdp_policy.ckpt')
    teacher_ckpt = os.path.join(args.out_dir, 'navdp_teacher.ckpt')

    if os.path.exists(policy_ckpt):
        model.load_state_dict(torch.load(policy_ckpt, map_location=device), strict=False)
        print(f"[Checkpoint] resumed policy from {policy_ckpt}")

    if 'diffusion' in stages:
        train_diffusion(model, train_dataset, device, args, writer)
        torch.save(model.state_dict(), policy_ckpt)

    if 'teacher' in stages:
        train_teacher(scorer, train_dataset, device, args, writer,
                      esdf, origin, resolution, checkpoint_path=teacher_ckpt)
    elif 'student' in stages:
        if not os.path.exists(teacher_ckpt):
            raise SystemExit(
                f"stage 'student' needs a trained teacher, but {teacher_ckpt} does not exist. "
                f"Run --stages teacher first."
            )
        scorer.load_state_dict(torch.load(teacher_ckpt, map_location=device))
        print(f"[Checkpoint] loaded teacher from {teacher_ckpt}")

    if 'student' in stages:
        train_student(model, scorer, train_dataset, device, args, writer,
                      esdf, origin, resolution, checkpoint_path=policy_ckpt)
        evaluate_model(model, train_dataset, scorer, device,
                       scene_path=args.esdf_npz, seed=args.seed)

    final = os.path.join(args.out_dir, 'navdp-final.ckpt')
    torch.save({'model': model.state_dict(),
                'scorer': scorer.state_dict(),
                'meta': {'scene_esdf_npz': args.esdf_npz,
                         'd_safe': args.d_safe, 'rho': args.rho, 'kappa': args.kappa}},
               final)
    print(f"[Checkpoint] saved model+scorer to {final}")
    writer.close()


if __name__ == '__main__':
    main()
