"""Test existing 3D clipping-mask code and save previews for example clouds.

All extraction is delegated to scf3dmask.mask3d.generate_3d_binary_mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scf3dmask.io import read_point_cloud, write_line_set, write_point_cloud
from scf3dmask.mask3d import generate_3d_binary_mask
from scf3dmask.mask_visualization import save_mask_preview, show_mask_result


def checked_points(cloud, name):
    points = np.asarray(cloud.points)
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f"{name}: at least three finite points are required")
    return points


def run_case(source_path, reference_path, output_dir, args, seed):
    source = read_point_cloud(source_path)
    source_points = checked_points(source, "Source")
    if reference_path is None:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(source_points),
                                    min(args.reference_points, len(source_points)), replace=False))
        reference = source.select_by_index(indices.tolist())
        reference_kind = "Seeded subset of the source; no additional damage or noise is generated."
    else:
        reference = read_point_cloud(reference_path)
        reference_kind = "User-provided reference cloud."
    reference_points = checked_points(reference, "Reference")
    if np.ptp(reference_points[:, 2]) <= 0:
        raise ValueError("The reference must span a nonzero Z range")

    print(f"[mask] {source_path.name}: source={len(source_points):,}, "
          f"reference={len(reference_points):,}, layers={args.num_layers}", flush=True)
    started = time.perf_counter()
    result = generate_3d_binary_mask(reference, source, num_layers=args.num_layers,
                                    ema=args.ema, alpha_multiplier=args.alpha_multiplier)
    algorithm_seconds = time.perf_counter() - started
    extracted_points = checked_points(result.extracted, "Extraction")
    if len(extracted_points) > len(source_points):
        raise ValueError("Extraction contains more points than the source")
    if not result.slices or not len(result.frame.lines):
        raise ValueError("No valid slice mask or wireframe was generated")
    if not np.isfinite(np.asarray(result.frame.points)).all():
        raise ValueError("Mask frame has nonfinite coordinates")
    if source.has_colors() and not result.extracted.has_colors():
        raise ValueError("Source colors were lost")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_point_cloud(output_dir / "reference.ply", reference)
    write_point_cloud(output_dir / "extracted.ply", result.extracted)
    write_line_set(output_dir / "mask_frame.ply", result.frame)
    saved = read_point_cloud(output_dir / "extracted.ply")
    if not np.array_equal(checked_points(saved, "Saved extraction"), extracted_points):
        raise ValueError("Extraction coordinates changed during PLY round-trip")
    if source.has_colors():
        np.testing.assert_allclose(np.asarray(saved.colors), np.asarray(result.extracted.colors),
                                   rtol=0, atol=1.0 / 255.0 + 1e-8)
    saved_frame = o3d.io.read_line_set(str(output_dir / "mask_frame.ply"))
    if len(saved_frame.lines) != len(result.frame.lines):
        raise ValueError("Saved wireframe line count is incorrect")
    slices = [{"index": s.index, "z_start": s.z_start, "z_end": s.z_end,
               "z_ref_start": s.z_ref_start, "z_ref_end": s.z_ref_end,
               "polygon_area": float(s.polygon.area), "color": list(s.color)} for s in result.slices]
    (output_dir / "slices.json").write_text(json.dumps(slices, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    preview = save_mask_preview(source, reference, result.extracted, result.frame,
                                output_dir / "preview.png", max_points=args.max_display_points)
    record = {"case": source_path.stem, "status": "passed", "source_file": source_path.name,
              "reference_file": reference_path.name if reference_path else None,
              "reference_kind": reference_kind, "seed": seed,
              "source_points": len(source_points), "reference_points": len(reference_points),
              "extracted_points": len(extracted_points),
              "retained_fraction": len(extracted_points) / len(source_points),
              "valid_slices": len(result.slices), "frame_lines": len(result.frame.lines),
              "algorithm_seconds": round(algorithm_seconds, 3),
              "finite_coordinates": True, "ply_round_trip": "passed",
              "source_colors_preserved": source.has_colors(),
              "preview": str(preview.relative_to(args.output_dir.resolve()))}
    if args.visualize:
        show_mask_result(result.extracted, result.frame, point_size=args.point_size,
                         title=f"SCF-3DMask | {source_path.stem}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data/examples/3D clipping mask")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/mask_examples")
    parser.add_argument("--case", help="Run one input by filename or filename stem (default: basket).")
    parser.add_argument("--all", action="store_true", help="Explicitly run every input in the directory.")
    parser.add_argument("--source", type=Path, help="Explicit source cloud (requires --reference).")
    parser.add_argument("--reference", type=Path, help="Explicit sparse/repaired reference cloud.")
    parser.add_argument("--reference-points", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=70)
    parser.add_argument("--ema", type=float, default=0.8)
    parser.add_argument("--alpha-multiplier", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-display-points", type=int, default=30000)
    parser.add_argument("--visualize", action="store_true", help="Also open an interactive Open3D view.")
    parser.add_argument("--point-size", type=float, default=4.0)
    args = parser.parse_args()
    if bool(args.source) != bool(args.reference):
        parser.error("--source and --reference must be supplied together")
    if args.reference_points < 3 or args.num_layers < 1 or args.max_display_points < 3:
        parser.error("Point limits must be at least 3 and --num-layers must be positive")
    if not 0 <= args.ema <= 1 or args.alpha_multiplier <= 0:
        parser.error("--ema must be in [0, 1] and --alpha-multiplier must be positive")
    if args.all and (args.case or args.source):
        parser.error("--all cannot be combined with --case or --source")
    if args.source and args.case:
        parser.error("--case selects a bundled input; use --source/--reference separately")
    args.output_dir = args.output_dir.resolve()
    files = [args.source.resolve()] if args.source else sorted(args.input_dir.resolve().glob("*.ply"))
    if args.case:
        files = [p for p in files if args.case in (p.name, p.stem)]
    elif not args.all and not args.source and files:
        files = [next((p for p in files if p.stem == "basket"), files[0])]
    if not files:
        parser.error("No matching .ply inputs were found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, source in enumerate(files):
        try:
            record = run_case(source, args.reference.resolve() if args.reference else None,
                              args.output_dir / source.stem, args, args.seed + index)
            print(f"[PASSED] {source.name}: {record['extracted_points']:,} extracted points, "
                  f"{record['valid_slices']} valid slices, {record['algorithm_seconds']:.3f}s", flush=True)
        except Exception as exc:
            record = {"case": source.stem, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[FAILED] {source.name}: {record['error']}", file=sys.stderr, flush=True)
        records.append(record)
        passed = sum(r["status"] == "passed" for r in records)
        report = {"test": "3d_clipping_mask_examples", "algorithm": "scf3dmask.mask3d.generate_3d_binary_mask",
                  "parameters": {"num_layers": args.num_layers, "ema": args.ema,
                                 "alpha_multiplier": args.alpha_multiplier},
                  "scope": "Functional clipping and visualization test; sampled-reference examples are not "
                           "paired ground-truth denoising benchmarks. All source points are used for extraction.",
                  "passed": passed, "failed": len(records) - passed, "planned_cases": len(files),
                  "results": records}
        (args.output_dir / "test_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                                                    allow_nan=False) + "\n", encoding="utf-8")
    print(f"[mask] {passed}/{len(files)} passed. Outputs: {args.output_dir}", flush=True)
    return 0 if passed == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
