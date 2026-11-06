from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import open3d as o3d


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return pcd


def write_point_cloud(path: str | Path, pcd: o3d.geometry.PointCloud) -> Path:
    path = ensure_parent(path)
    ok = o3d.io.write_point_cloud(str(path), pcd)
    if not ok:
        raise IOError(f"Failed to write point cloud: {path}")
    return path


def write_line_set(path: str | Path, line_set: o3d.geometry.LineSet) -> Path:
    path = ensure_parent(path)
    if len(line_set.lines) == 0:
        raise ValueError("Cannot write an empty LineSet.")
    ok = o3d.io.write_line_set(str(path), line_set)
    if not ok:
        raise IOError(f"Failed to write line set: {path}")
    return path


def make_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def merge_point_clouds(clouds: Iterable[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    merged = o3d.geometry.PointCloud()
    for cloud in clouds:
        if cloud is not None and not cloud.is_empty():
            merged += cloud
    return merged


def copy_point_cloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    copied = o3d.geometry.PointCloud()
    copied.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
    if pcd.has_colors():
        copied.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    if pcd.has_normals():
        copied.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
    return copied
