from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from scf3dmask.io import read_point_cloud, write_point_cloud
from scf3dmask.utils import denormalize_points, normalize_points, random_sample_points

from .model import TopologyRepairNetwork


def load_model(
    checkpoint_path: str | Path,
    num_input_points: int = 8192,
    device: torch.device | None = None,
) -> TopologyRepairNetwork:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TopologyRepairNetwork(
        num_points=num_input_points,
        num_anchors=64,
        feature_dim=256,
        k=16,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def infer_patch(
    input_pcd: o3d.geometry.PointCloud,
    checkpoint_path: str | Path,
    num_input_points: int = 8192,
    device: torch.device | None = None,
    *,
    model: TopologyRepairNetwork | None = None,
    rng: np.random.Generator | None = None,
) -> o3d.geometry.PointCloud:
    """Predict a patch, optionally reusing a loaded model and seeded sampler."""
    if model is not None:
        model_device = next(model.parameters()).device
        if device is not None:
            requested = torch.device(device)
            if requested.type != model_device.type or (
                requested.index is not None and requested.index != model_device.index
            ):
                raise ValueError("The supplied model and requested device must match")
        device = model_device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_points = np.asarray(input_pcd.points, dtype=np.float32)
    sampled = random_sample_points(raw_points, num_input_points, rng=rng)
    normalized, norm = normalize_points(sampled)

    if model is None:
        model = load_model(checkpoint_path, num_input_points=num_input_points, device=device)
    model.eval()
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, _, patch_norm = model(tensor)
    patch_world = denormalize_points(patch_norm.squeeze(0).cpu().numpy(), norm)

    patch = o3d.geometry.PointCloud()
    patch.points = o3d.utility.Vector3dVector(patch_world)
    patch.paint_uniform_color([1.0, 0.0, 0.0])
    return patch


def save_inference_outputs(
    input_pcd: o3d.geometry.PointCloud,
    patch_pcd: o3d.geometry.PointCloud,
    output_path: str | Path,
    patch_only_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    raw = np.asarray(input_pcd.points)
    patch = np.asarray(patch_pcd.points)
    combined = o3d.geometry.PointCloud()
    combined.points = o3d.utility.Vector3dVector(np.vstack([raw, patch]))
    colors = np.vstack(
        [
            np.tile([0.70, 0.70, 0.70], (len(raw), 1)),
            np.tile([1.0, 0.0, 0.0], (len(patch), 1)),
        ]
    )
    combined.colors = o3d.utility.Vector3dVector(colors)
    out = write_point_cloud(output_path, combined)
    patch_out = write_point_cloud(patch_only_path, patch_pcd) if patch_only_path is not None else None
    return out, patch_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spatial-continuity filling on a sparse point cloud.")
    parser.add_argument("--input", required=True, help="Sparse proxy or damaged point cloud.")
    parser.add_argument("--ckpt", required=True, help="TopologyRepairNetwork checkpoint.")
    parser.add_argument("--output", required=True, help="Colored input+repair output PLY.")
    parser.add_argument("--patch-output", default=None, help="Optional patch-only PLY.")
    parser.add_argument("--num-input-points", type=int, default=8192)
    args = parser.parse_args()

    pcd = read_point_cloud(args.input)
    patch = infer_patch(pcd, args.ckpt, num_input_points=args.num_input_points)
    save_inference_outputs(pcd, patch, args.output, args.patch_output)


if __name__ == "__main__":
    main()
