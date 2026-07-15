#!/usr/bin/env python
"""Thin CLI around navdp_safety.data.esdf.build_esdf."""
import argparse
import os

import numpy as np

from navdp_safety.data.esdf import build_esdf, visualize_esdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vis_dir", type=str, default="esdf_vis")
    args = parser.parse_args()

    esdf, occupancy, origin, resolution = build_esdf(args.scene_path)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    np.savez(args.output_path, esdf=esdf, origin=origin, resolution=resolution)
    np.savez(args.output_path.replace(".npz", "_occupancy.npz"),
             occupancy=occupancy, origin=origin, resolution=resolution)

    visualize_esdf(esdf, args.vis_dir)

    print(f"[Done] ESDF saved to {args.output_path}")


if __name__ == "__main__":
    main()
