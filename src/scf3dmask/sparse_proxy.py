from __future__ import annotations

import math
import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class CandidateCluster:
    label: int
    cluster_pcd: o3d.geometry.PointCloud
    n_points: int
    grown_ratio: float
    distance_to_center: float
    compactness: float
    score: float = 0.0


@dataclass
class SparseProxyResult:
    proxy: o3d.geometry.PointCloud
    half_space: o3d.geometry.PointCloud
    plane_model: Optional[tuple[float, float, float, float]]
    candidates: list[CandidateCluster] = field(default_factory=list)
    dbscan_labels: Optional[np.ndarray] = None


def split_space_by_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.04,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[o3d.geometry.PointCloud, tuple[float, float, float, float]]:
    if len(pcd.points) < ransac_n:
        return pcd, (0.0, 0.0, 1.0, 0.0)

    plane_model, _ = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    points = np.asarray(pcd.points)
    values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d

    mask_positive = values > distance_threshold
    mask_negative = values < -distance_threshold
    keep = mask_positive if mask_positive.sum() > mask_negative.sum() else mask_negative
    kept_indices = np.where(keep)[0]
    if len(kept_indices) == 0:
        return pcd, tuple(float(x) for x in plane_model)
    return pcd.select_by_index(kept_indices), tuple(float(x) for x in plane_model)


def estimate_curvatures(
    pcd: o3d.geometry.PointCloud,
    normal_radius: float = 0.10,
    max_nn: int = 30,
) -> np.ndarray:
    work = pcd
    if not work.has_normals():
        work.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=max_nn)
        )
    work.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn))
    covs = np.asarray(work.covariances)
    if len(covs) == 0:
        return np.zeros(len(work.points), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covs)
    return eigenvalues[:, 0] / (np.sum(eigenvalues, axis=1) + 1e-8)


def region_growing_multi_seed(
    pcd: o3d.geometry.PointCloud,
    seed_indices: np.ndarray,
    center_point: np.ndarray,
    curvatures: np.ndarray,
    angle_threshold_deg: float = 35.0,
    dist_threshold: float = 0.06,
    max_grow_radius: float = 0.8,
    max_curvature: float = 0.08,
) -> set[int]:
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    labels = -np.ones(len(points), dtype=np.int8)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    q: queue.Queue[int] = queue.Queue()
    cluster_indices: list[int] = []

    for seed_idx in seed_indices:
        q.put(int(seed_idx))
        labels[int(seed_idx)] = 1
        cluster_indices.append(int(seed_idx))

    cos_threshold = math.cos(math.radians(angle_threshold_deg))
    while not q.empty():
        curr_idx = q.get()
        curr_normal = normals[curr_idx]
        curr_point = points[curr_idx]
        _, idxs, _ = pcd_tree.search_radius_vector_3d(curr_point, dist_threshold)

        for neighbor_idx in idxs:
            if labels[neighbor_idx] != -1:
                continue
            if curvatures[neighbor_idx] > max_curvature:
                labels[neighbor_idx] = 0
                continue
            if np.linalg.norm(points[neighbor_idx] - center_point) > max_grow_radius:
                labels[neighbor_idx] = 0
                continue

            neighbor_normal = normals[neighbor_idx]
            dot_product = float(np.dot(curr_normal, neighbor_normal))
            dot_product = np.clip(dot_product, -1.0, 1.0)
            if abs(dot_product) > cos_threshold:
                labels[neighbor_idx] = 1
                cluster_indices.append(int(neighbor_idx))
                q.put(int(neighbor_idx))
            else:
                labels[neighbor_idx] = 0

    return set(cluster_indices)


def color_dbscan_labels(labels: np.ndarray) -> np.ndarray:
    colors = np.full((len(labels), 3), 0.55, dtype=np.float64)
    valid = labels >= 0
    if valid.any():
        rng = np.random.default_rng(7)
        palette = rng.random((int(labels.max()) + 1, 3)) * 0.75 + 0.20
        colors[valid] = palette[labels[valid]]
    return colors


def extract_sparse_proxy(
    pcd: o3d.geometry.PointCloud,
    plane_distance: float = 0.04,
    dbscan_eps: float = 0.20,
    dbscan_min_points: int = 30,
    min_cluster_ratio: float = 0.01,
    min_cluster_points: int = 50,
    seed_count: int = 5,
    random_seed: int = 0,
) -> SparseProxyResult:
    half_space, plane_model = split_space_by_plane(pcd, distance_threshold=plane_distance)
    if len(half_space.points) < min_cluster_points:
        return SparseProxyResult(proxy=half_space, half_space=half_space, plane_model=plane_model)

    labels = np.asarray(
        half_space.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False)
    )
    if labels.size == 0 or labels.max(initial=-1) < 0:
        return SparseProxyResult(proxy=half_space, half_space=half_space, plane_model=plane_model, dbscan_labels=labels)

    work = o3d.geometry.PointCloud(half_space)
    work.colors = o3d.utility.Vector3dVector(color_dbscan_labels(labels))
    work.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.10, max_nn=30))
    try:
        work.orient_normals_consistent_tangent_plane(100)
    except RuntimeError:
        pass
    curvatures = estimate_curvatures(work)

    points = np.asarray(work.points)
    global_center = work.get_center()
    min_size = max(min_cluster_points, int(len(points) * min_cluster_ratio))
    candidates: list[CandidateCluster] = []
    rng = np.random.default_rng(random_seed)

    for label in np.unique(labels):
        if label < 0:
            continue
        indices = np.where(labels == label)[0]
        if len(indices) < min_size:
            continue

        cluster_pcd = work.select_by_index(indices.tolist())
        cluster_center = cluster_pcd.get_center()
        valid_seed_pool = indices[curvatures[indices] < 0.04]
        if len(valid_seed_pool) == 0:
            valid_seed_pool = indices
        seeds = rng.choice(valid_seed_pool, min(seed_count, len(valid_seed_pool)), replace=False)

        extent = cluster_pcd.get_axis_aligned_bounding_box().get_extent()
        dynamic_radius = max(float(np.max(extent)) * 0.6, dbscan_eps)
        grown = region_growing_multi_seed(
            work,
            seed_indices=seeds,
            center_point=cluster_center,
            curvatures=curvatures,
            angle_threshold_deg=35.0,
            dist_threshold=0.06,
            max_grow_radius=dynamic_radius,
            max_curvature=0.08,
        )
        grown_ratio = len(grown) / max(len(indices), 1)
        sorted_extent = np.sort(extent)
        compactness = float(sorted_extent[0] / (sorted_extent[2] + 1e-6))
        candidates.append(
            CandidateCluster(
                label=int(label),
                cluster_pcd=cluster_pcd,
                n_points=len(indices),
                grown_ratio=float(grown_ratio),
                distance_to_center=float(np.linalg.norm(cluster_center - global_center)),
                compactness=compactness,
            )
        )

    if not candidates:
        largest_label = int(np.bincount(labels[labels >= 0]).argmax())
        proxy = work.select_by_index(np.where(labels == largest_label)[0].tolist())
        return SparseProxyResult(proxy=proxy, half_space=half_space, plane_model=plane_model, dbscan_labels=labels)

    max_points = max(c.n_points for c in candidates)
    max_dist = max(max(c.distance_to_center for c in candidates), 1e-6)
    max_compactness = max(max(c.compactness for c in candidates), 1e-6)
    weights = dict(size=2.0, grown=4.0, distance=4.0, compactness=6.0)

    for candidate in candidates:
        norm_points = candidate.n_points / max_points
        norm_dist = 1.0 - candidate.distance_to_center / max_dist
        norm_compactness = candidate.compactness / max_compactness
        candidate.score = (
            weights["size"] * norm_points
            + weights["grown"] * candidate.grown_ratio
            + weights["distance"] * norm_dist
            + weights["compactness"] * norm_compactness
        )

    best = max(candidates, key=lambda item: item.score)
    return SparseProxyResult(
        proxy=best.cluster_pcd,
        half_space=half_space,
        plane_model=plane_model,
        candidates=candidates,
        dbscan_labels=labels,
    )
