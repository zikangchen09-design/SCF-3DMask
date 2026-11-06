from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from .io import read_point_cloud


def load_optional_point_cloud(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return read_point_cloud(path)


def show_results(
    input_path: str | Path | None = None,
    sparse_proxy_path: str | Path | None = None,
    frame_path: str | Path | None = None,
    dense_path: str | Path | None = None,
    final_path: str | Path | None = None,
    point_size: float = 3.0,
) -> None:
    geometries = []

    input_pcd = load_optional_point_cloud(input_path)
    if input_pcd is not None:
        input_pcd.paint_uniform_color([0.70, 0.70, 0.70])
        geometries.append(input_pcd)

    proxy = load_optional_point_cloud(sparse_proxy_path)
    if proxy is not None:
        proxy.paint_uniform_color([1.0, 0.45, 0.05])
        geometries.append(proxy)

    dense = load_optional_point_cloud(dense_path)
    if dense is not None:
        if not dense.has_colors():
            dense.paint_uniform_color([0.10, 0.65, 1.0])
        geometries.append(dense)

    final = load_optional_point_cloud(final_path)
    if final is not None:
        if not final.has_colors():
            final.paint_uniform_color([0.15, 0.80, 0.35])
        geometries.append(final)

    if frame_path is not None and Path(frame_path).exists():
        frame = o3d.io.read_line_set(str(frame_path))
        if len(frame.lines) > 0:
            geometries.append(frame)

    if not geometries:
        raise ValueError("No readable geometries were provided.")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SCF-3DMask Results", width=1280, height=900)
    for geometry in geometries:
        vis.add_geometry(geometry)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize SCF-3DMask intermediate and final results.")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--sparse-proxy", type=str, default=None)
    parser.add_argument("--frame", type=str, default=None)
    parser.add_argument("--dense", type=str, default=None)
    parser.add_argument("--final", type=str, default=None)
    parser.add_argument("--point-size", type=float, default=3.0)
    args = parser.parse_args()
    show_results(
        input_path=args.input,
        sparse_proxy_path=args.sparse_proxy,
        frame_path=args.frame,
        dense_path=args.dense,
        final_path=args.final,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
