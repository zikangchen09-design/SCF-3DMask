"""Visualize existing mask results without rebuilding extraction geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def display_sample(cloud: o3d.geometry.PointCloud, limit: int = 30000):
    """Limit display cost without changing clouds used by the algorithm."""
    if len(cloud.points) <= limit:
        return o3d.geometry.PointCloud(cloud)
    indices = np.linspace(0, len(cloud.points) - 1, limit, dtype=int)
    return cloud.select_by_index(indices.tolist())


def save_mask_preview(source, reference, extracted, frame, path, max_points=30000):
    """Save a four-panel preview using the frame returned by mask3d."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:
        raise RuntimeError('Install visualization dependencies: pip install -e ".[visualization]"') from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shown_source = display_sample(source, max_points)
    shown_ref = display_sample(reference, max_points)
    shown_ref.paint_uniform_color([0.1, 0.4, 0.9])
    shown_result = display_sample(extracted, max_points)
    # Display points retain original colors; the exported extraction is untouched.
    clouds = [shown_source, shown_ref, shown_result, shown_result]
    titles = [f"Source ({len(source.points):,} points)",
              f"Sparse reference ({len(reference.points):,} points)",
              "3D clipping mask + extraction",
              f"Extracted ({len(extracted.points):,} points)"]
    minimum = source.get_min_bound()
    maximum = source.get_max_bound()
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.55, 1e-6)
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    try:
        for i, (cloud, title) in enumerate(zip(clouds, titles), start=1):
            ax = fig.add_subplot(2, 2, i, projection="3d")
            points = np.asarray(cloud.points)
            colors = np.asarray(cloud.colors) if cloud.has_colors() else "#38a169"
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                       s=0.6, linewidths=0, depthshade=False, rasterized=True)
            if i == 3 and len(frame.lines):
                lines = np.asarray(frame.lines)
                keep = np.linspace(0, len(lines) - 1, min(len(lines), 25000), dtype=int)
                segments = np.asarray(frame.points)[lines[keep]]
                line_colors = np.asarray(frame.colors)[keep] if frame.has_colors() else "#1f77b4"
                ax.add_collection3d(Line3DCollection(segments, colors=line_colors,
                                                   linewidths=0.25, alpha=0.65))
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_zlim(center[2] - radius, center[2] + radius)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=22, azim=-65)
            ax.set_axis_off()
            ax.set_title(title, fontsize=11)
        fig.suptitle("SCF-3DMask: source / reference / mask / extraction", fontsize=14)
        fig.tight_layout()
        fig.savefig(path, dpi=140, facecolor="white")
    finally:
        plt.close(fig)
    return path


def show_mask_result(extracted, frame, point_size=4.0, title="SCF-3DMask | clipping mask"):
    """White-background Open3D view of extraction and existing colored slices."""
    viewer = o3d.visualization.Visualizer()
    if not viewer.create_window(window_name=title, width=1200, height=900):
        raise RuntimeError("Cannot create the Open3D visualization window")
    try:
        viewer.add_geometry(extracted)
        viewer.add_geometry(frame)
        options = viewer.get_render_option()
        options.background_color = np.array([1.0, 1.0, 1.0])
        options.point_size = point_size
        options.line_width = 1.0
        viewer.run()
    finally:
        viewer.destroy_window()
