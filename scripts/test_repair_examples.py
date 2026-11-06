"""Run and validate the bundled spatial-continuity filling examples.

This is an inference smoke test, not a reconstruction-accuracy benchmark.
Run from a source checkout; an editable installation is not required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import open3d as o3d
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scf3dmask.io import read_point_cloud
from scf3dmask.repair.inference import infer_patch, load_model, save_inference_outputs

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data/examples/continuity_filling"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/examples/continuity_filling_demo.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_points(cloud: o3d.geometry.PointCloud, label: str) -> np.ndarray:
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{label}: expected a nonempty Nx3 point cloud")
    if not np.isfinite(points).all():
        raise ValueError(f"{label}: NaN or infinite coordinates")
    return points


def run(args: argparse.Namespace) -> int:
    input_dir = args.input_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    files = sorted(input_dir.glob("*input*.ply"))
    if not files:
        raise FileNotFoundError(f"No *input*.ply examples found in {input_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    manifest = None
    # The bundled manifest describes only the bundled data and checkpoint.
    if input_dir == DEFAULT_INPUT_DIR.resolve() and checkpoint == DEFAULT_CHECKPOINT.resolve():
        manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
        if {p.name for p in files} != {entry["filename"] for entry in manifest["inputs"]}:
            raise ValueError("Bundled input files do not match manifest.json")
        for entry in manifest["inputs"]:
            if sha256(input_dir / entry["filename"]) != entry["sha256"]:
                raise ValueError(f"Input checksum mismatch: {entry['filename']}")
        if sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch runtime has no available CUDA device")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load once for the whole batch; infer_patch uses the normal project path.
    model = load_model(checkpoint, num_input_points=8192, device=device)
    print(f"[examples] device={device}, inputs={len(files)}, checkpoint={checkpoint.name}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(files):
        started = time.perf_counter()
        record = {"input": path.name, "seed": args.seed + index}
        try:
            cloud = read_point_cloud(path)
            original = checked_points(cloud, "Input").copy()
            if manifest is not None:
                expected = next(entry for entry in manifest["inputs"] if entry["filename"] == path.name)
                if len(original) != expected["point_count"]:
                    raise ValueError("Input point count does not match the manifest")
            patch = infer_patch(cloud, checkpoint, device=device, model=model,
                                rng=np.random.default_rng(args.seed + index))
            patch_points = checked_points(patch, "Predicted patch")
            if patch_points.shape != (1024, 3):
                raise ValueError(f"Expected a 1024x3 patch, received {patch_points.shape}")
            if not np.array_equal(np.asarray(cloud.points), original):
                raise ValueError("Inference changed the original input coordinates")

            prefix = path.stem.removesuffix("_input")
            repaired_path = output_dir / f"{prefix}_repaired.ply"
            patch_path = output_dir / f"{prefix}_patch.ply"
            save_inference_outputs(cloud, patch, repaired_path, patch_path)

            # Read files back rather than treating a successful write as a test.
            saved_patch = checked_points(read_point_cloud(patch_path), "Saved patch")
            repaired = checked_points(read_point_cloud(repaired_path), "Saved repaired cloud")
            if saved_patch.shape != (1024, 3) or len(repaired) != len(original) + 1024:
                raise ValueError("Saved point counts are incorrect")
            np.testing.assert_allclose(saved_patch, patch_points, rtol=0, atol=1e-7)
            np.testing.assert_allclose(repaired[:len(original)], original, rtol=0, atol=1e-7)
            np.testing.assert_allclose(repaired[len(original):], patch_points, rtol=0, atol=1e-7)

            record.update({"status": "passed", "input_points": len(original),
                           "network_input_points": 8192, "patch_points": len(saved_patch),
                           "repaired_points": len(repaired), "finite_coordinates": True,
                           "original_coordinates_preserved": True,
                           "patch_file": patch_path.name, "repaired_file": repaired_path.name,
                           "patch_bounds": {"min": saved_patch.min(axis=0).tolist(),
                                            "max": saved_patch.max(axis=0).tolist()}})
        except Exception as exc:
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        records.append(record)
        print(f"[{record['status'].upper()}] {path.name}: "
              f"{record.get('error', str(record.get('patch_points')) + ' patch points')} "
              f"({record['elapsed_seconds']:.3f}s)", flush=True)

    passed = sum(record["status"] == "passed" for record in records)
    report = {"test": "spatial_continuity_filling_smoke_test", "device": str(device),
              "versions": {"torch": torch.__version__, "numpy": np.__version__,
                           "open3d": o3d.__version__},
              "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint),
              "seed": args.seed, "bundled_manifest_verified": manifest is not None,
              "passed": passed, "failed": len(records) - passed,
              "scope": "Inference, finite coordinates, point counts and PLY round-trip only; "
                       "no ground-truth accuracy or generalization evaluation.",
              "results": records}
    report_path = output_dir / "test_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"[examples] {passed}/{len(records)} passed. Report: {report_path}", flush=True)
    return 0 if passed == len(records) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing *input*.ply point clouds.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/repair_examples")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[examples] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
