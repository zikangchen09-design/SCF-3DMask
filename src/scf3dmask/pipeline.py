from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from .denoise import NMRFilter
from .fusion import fuse_dense_geometry
from .io import read_point_cloud, write_line_set, write_point_cloud
from .mask3d import generate_3d_binary_mask
from .repair.inference import infer_patch
from .sparse_proxy import extract_sparse_proxy


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    voxel_size: float = 0.01,
    num_layers: int = 70,
    skip_repair: bool = False,
    verbose: bool = True,
) -> dict[str, str | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = read_point_cloud(input_path)
    work = source.voxel_down_sample(voxel_size) if voxel_size and voxel_size > 0 else source

    if verbose:
        print(f"[pipeline] input points={len(source.points)} working points={len(work.points)}")

    nmr = NMRFilter()
    denoise_results = nmr.run(work, verbose=verbose)
    if denoise_results:
        filtered = denoise_results[0].pcd
        best_branch = denoise_results[0].path
    else:
        filtered = work
        best_branch = "none"

    proxy_result = extract_sparse_proxy(filtered)
    sparse_proxy = proxy_result.proxy
    sparse_proxy.paint_uniform_color([1.0, 0.45, 0.05])
    sparse_proxy_path = write_point_cloud(output_dir / "01_sparse_proxy.ply", sparse_proxy)

    patch_pcd = None
    patch_path = None
    reference = o3d.geometry.PointCloud(sparse_proxy)
    if checkpoint_path and not skip_repair:
        if verbose:
            print(f"[pipeline] running repair checkpoint={checkpoint_path}")
        patch_pcd = infer_patch(sparse_proxy, checkpoint_path)
        patch_path = write_point_cloud(output_dir / "03_repair_patch.ply", patch_pcd)
        reference += patch_pcd
        write_point_cloud(output_dir / "03_repaired_sparse_reference.ply", reference)
    else:
        if verbose:
            print("[pipeline] repair skipped; using sparse proxy as mask reference")

    mask_result = generate_3d_binary_mask(reference, source, num_layers=num_layers)
    dense_path = write_point_cloud(output_dir / "04_mask_extracted_dense.ply", mask_result.extracted)
    frame_path = None
    if len(mask_result.frame.lines) > 0:
        frame_path = write_line_set(output_dir / "02_extraction_frame.ply", mask_result.frame)

    final_pcd = fuse_dense_geometry(mask_result.extracted, patch_pcd=patch_pcd)
    final_path = write_point_cloud(output_dir / "05_final_object.ply", final_pcd)

    metadata = {
        "input": str(input_path),
        "best_nmr_branch": best_branch,
        "source_points": len(source.points),
        "working_points": len(work.points),
        "sparse_proxy_points": len(sparse_proxy.points),
        "repair_patch_points": len(patch_pcd.points) if patch_pcd is not None else 0,
        "mask_extracted_points": len(mask_result.extracted.points),
        "final_points": len(final_pcd.points),
        "num_mask_slices": len(mask_result.slices),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    outputs = {
        "sparse_proxy": str(sparse_proxy_path),
        "extraction_frame": str(frame_path) if frame_path is not None else None,
        "repair_patch": str(patch_path) if patch_path is not None else None,
        "mask_extracted_dense": str(dense_path),
        "final_object": str(final_path),
        "metadata": str(output_dir / "metadata.json"),
    }
    if verbose:
        print("[pipeline] outputs:")
        for key, value in outputs.items():
            print(f"  {key}: {value}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete SCF-3DMask object recovery pipeline.")
    parser.add_argument("--input", required=True, help="Input reconstructed point cloud.")
    parser.add_argument("--output-dir", required=True, help="Directory for intermediate and final PLY files.")
    parser.add_argument("--ckpt", default=None, help="Optional repair-network checkpoint.")
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--num-layers", type=int, default=70)
    parser.add_argument("--skip-repair", action="store_true")
    args = parser.parse_args()
    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        checkpoint_path=args.ckpt,
        voxel_size=args.voxel_size,
        num_layers=args.num_layers,
        skip_repair=args.skip_repair,
    )


if __name__ == "__main__":
    main()
