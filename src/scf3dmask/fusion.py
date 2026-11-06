from __future__ import annotations

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

from .utils import color_array, estimate_spacing


def upsample_patch_points(
    patch_pcd: o3d.geometry.PointCloud,
    factor: int = 3,
    neighbor_count: int = 6,
    jitter_scale: float = 0.15,
) -> o3d.geometry.PointCloud:
    """Densify predicted repair patches with local interpolation."""

    if patch_pcd is None or patch_pcd.is_empty() or factor <= 1:
        return patch_pcd if patch_pcd is not None else o3d.geometry.PointCloud()

    points = np.asarray(patch_pcd.points)
    if len(points) < 2:
        return patch_pcd

    k = min(neighbor_count, len(points))
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points)
    _, indices = neigh.kneighbors(points)
    spacing = estimate_spacing(points)
    rng = np.random.default_rng(123)

    generated = [points]
    for _ in range(factor - 1):
        neighbor_pick = indices[np.arange(len(points)), rng.integers(1, k, size=len(points)) if k > 1 else 0]
        alpha = rng.uniform(0.2, 0.8, size=(len(points), 1))
        interp = points * (1.0 - alpha) + points[neighbor_pick] * alpha
        interp += rng.normal(scale=spacing * jitter_scale, size=interp.shape)
        generated.append(interp)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack(generated))
    out.paint_uniform_color([1.0, 0.20, 0.10])
    return out


def remove_duplicate_points(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float | None = None,
) -> o3d.geometry.PointCloud:
    if pcd.is_empty():
        return pcd
    if voxel_size is None:
        voxel_size = estimate_spacing(np.asarray(pcd.points)) * 0.5
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size)


def smooth_boundary(
    target_pcd: o3d.geometry.PointCloud,
    patch_pcd: o3d.geometry.PointCloud,
    k: int = 8,
    blend_radius: float | None = None,
) -> o3d.geometry.PointCloud:
    """Blend patch points near the extracted dense cloud boundary."""

    if target_pcd.is_empty() or patch_pcd.is_empty():
        return patch_pcd

    target = np.asarray(target_pcd.points)
    patch = np.asarray(patch_pcd.points)
    if len(target) < 2 or len(patch) < 2:
        return patch_pcd

    spacing = estimate_spacing(target)
    blend_radius = blend_radius or spacing * 3.0
    neigh = NearestNeighbors(n_neighbors=min(k, len(target)))
    neigh.fit(target)
    dists, indices = neigh.kneighbors(patch)
    weights = np.exp(-(dists / max(blend_radius, 1e-8)) ** 2)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    neighbor_mean = (target[indices] * weights[:, :, None]).sum(axis=1)

    nearest = dists[:, 0]
    alpha = np.clip(1.0 - nearest / max(blend_radius, 1e-8), 0.0, 0.35)[:, None]
    smoothed = patch * (1.0 - alpha) + neighbor_mean * alpha

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(smoothed)
    if patch_pcd.has_colors():
        out.colors = patch_pcd.colors
    else:
        out.paint_uniform_color([1.0, 0.20, 0.10])
    return out


def fuse_dense_geometry(
    dense_pcd: o3d.geometry.PointCloud,
    patch_pcd: o3d.geometry.PointCloud | None = None,
    upsample_factor: int = 3,
    voxel_size: float | None = None,
    smooth: bool = True,
) -> o3d.geometry.PointCloud:
    dense = o3d.geometry.PointCloud(dense_pcd)
    if dense.has_colors():
        dense_colors = np.asarray(dense.colors)
    else:
        dense_colors = color_array(len(dense.points), (0.55, 0.55, 0.55))
        dense.colors = o3d.utility.Vector3dVector(dense_colors)

    if patch_pcd is None or patch_pcd.is_empty():
        return remove_duplicate_points(dense, voxel_size)

    patch_dense = upsample_patch_points(patch_pcd, factor=upsample_factor)
    if smooth:
        patch_dense = smooth_boundary(dense, patch_dense)
    if not patch_dense.has_colors():
        patch_dense.paint_uniform_color([1.0, 0.20, 0.10])

    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(dense.points), np.asarray(patch_dense.points)])
    )
    merged.colors = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(dense.colors), np.asarray(patch_dense.colors)])
    )
    return remove_duplicate_points(merged, voxel_size)
