from __future__ import annotations

import argparse

from scf3dmask.io import read_point_cloud, write_line_set, write_point_cloud
from scf3dmask.mask3d import generate_3d_binary_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SCF-3DMask 3D binary mask generator.")
    parser.add_argument("--reference", required=True, help="Sparse or repaired reference point cloud.")
    parser.add_argument("--source", required=True, help="Original dense reconstructed point cloud.")
    parser.add_argument("--output", required=True, help="Extracted dense target PLY.")
    parser.add_argument("--frame-output", default=None, help="Optional slice-frame LineSet PLY.")
    parser.add_argument("--num-layers", type=int, default=70)
    args = parser.parse_args()

    reference = read_point_cloud(args.reference)
    source = read_point_cloud(args.source)
    result = generate_3d_binary_mask(reference, source, num_layers=args.num_layers)
    write_point_cloud(args.output, result.extracted)
    if args.frame_output and len(result.frame.lines) > 0:
        write_line_set(args.frame_output, result.frame)


if __name__ == "__main__":
    main()
