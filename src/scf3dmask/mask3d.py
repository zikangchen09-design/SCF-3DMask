from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

try:
    from shapely import contains_xy
except ImportError:  # Shapely 1.x fallback.
    from shapely.vectorized import contains as contains_xy


@dataclass
class SliceMask:
    index: int
    z_start: float
    z_end: float
    z_ref_start: float
    z_ref_end: float
    polygon: object
    color: tuple[float, float, float]


@dataclass
class MaskExtractionResult:
    extracted: o3d.geometry.PointCloud
    frame: o3d.geometry.LineSet
    slices: list[SliceMask] = field(default_factory=list)


def global_density(points: np.ndarray, k: int = 5, default: float = 0.1) -> float:
    points = np.asarray(points)
    if len(points) < max(2, k):
        return default
    neigh = NearestNeighbors(n_neighbors=min(k, len(points)))
    neigh.fit(points)
    dists, _ = neigh.kneighbors(points)
    density = float(np.median(dists[:, -1]))
    return density if density > 0 else default


def analyze_2d_stats(points_2d: np.ndarray, density: float, k: int = 5) -> dict[str, float]:
    count = len(points_2d)
    if count <= 1:
        return {"d_avg": density, "d_std": 0.0, "eps": density * 2.0}
    actual_k = max(1, min(k, count - 1))
    neigh = NearestNeighbors(n_neighbors=actual_k)
    neigh.fit(points_2d)
    k_dists = neigh.kneighbors(points_2d)[0][:, -1]
    d_avg = float(np.mean(k_dists))
    return {
        "d_avg": d_avg,
        "d_std": float(np.std(k_dists)),
        "eps": float(np.clip(np.percentile(k_dists, 95), density * 0.5, density * 3.5)),
    }


def adaptive_cluster_polygon(
    cluster_points: np.ndarray,
    density: float,
    alpha_multiplier: float = 6.0,
    bridge_multiplier: float = 3.0,
    safety_multiplier: float = 0.4,
):
    count = len(cluster_points)
    stats = analyze_2d_stats(cluster_points, density)
    d_avg = max(stats["d_avg"], density * 0.1, 1e-8)
    safety_margin = d_avg * safety_multiplier
    base_buffer = MultiPoint(cluster_points).buffer(safety_margin, quad_segs=8)

    if count < 4:
        return base_buffer

    try:
        alpha = d_avg * alpha_multiplier * (1.0 + stats["d_std"] / (d_avg + 1e-6))
        tri = Delaunay(cluster_points)
        coords = cluster_points[tri.simplices]

        a = np.linalg.norm(coords[:, 0, :] - coords[:, 1, :], axis=1)
        b = np.linalg.norm(coords[:, 1, :] - coords[:, 2, :], axis=1)
        c = np.linalg.norm(coords[:, 2, :] - coords[:, 0, :], axis=1)
        s = (a + b + c) / 2.0
        area = np.sqrt(np.maximum(0, s * (s - a) * (s - b) * (s - c)))
        circum_r = np.divide(
            a * b * c,
            4.0 * area,
            out=np.full_like(a, np.inf),
            where=area > 0,
        )

        valid = [Polygon(coords[i]) for i in range(len(tri.simplices)) if circum_r[i] < alpha]
        raw = unary_union(valid + [base_buffer])
        raw = make_valid(raw)
        bridge_dist = d_avg * bridge_multiplier
        expanded = raw.buffer(bridge_dist, join_style=1, quad_segs=12)
        final_shape = expanded.buffer(-bridge_dist + safety_margin, join_style=1, quad_segs=12)

        if final_shape.is_empty or final_shape.area < base_buffer.area:
            final_shape = MultiPoint(cluster_points).convex_hull.buffer(safety_margin)
        return make_valid(final_shape)
    except Exception:
        return MultiPoint(cluster_points).convex_hull.buffer(safety_margin)


def _colormap_jet(ratio: float) -> tuple[float, float, float]:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    r = np.clip(1.5 - abs(4 * ratio - 3), 0, 1)
    g = np.clip(1.5 - abs(4 * ratio - 2), 0, 1)
    b = np.clip(1.5 - abs(4 * ratio - 1), 0, 1)
    return float(r), float(g), float(b)


def create_slice_wireframe(poly, z_start: float, z_end: float, color: tuple[float, float, float]):
    if poly is None or poly.is_empty:
        return None
    vertices: list[list[float]] = []
    lines: list[list[int]] = []

    def add_ring(ring) -> None:
        coords = np.asarray(ring.coords)
        n_coords = len(coords)
        if n_coords < 2:
            return
        start = len(vertices)
        vertices.extend(np.hstack([coords, np.full((n_coords, 1), z_end)]).tolist())
        vertices.extend(np.hstack([coords, np.full((n_coords, 1), z_start)]).tolist())
        for idx in range(n_coords - 1):
            lines.append([start + idx, start + idx + 1])
            lines.append([start + n_coords + idx, start + n_coords + idx + 1])
            lines.append([start + idx, start + n_coords + idx])

    polygons = [poly] if poly.geom_type == "Polygon" else list(poly.geoms)
    for polygon in polygons:
        if not polygon.is_empty and hasattr(polygon, "exterior"):
            add_ring(polygon.exterior)
            for interior in polygon.interiors:
                add_ring(interior)

    if not vertices or not lines:
        return None
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.paint_uniform_color(color)
    return line_set


def combine_line_sets(line_sets: list[o3d.geometry.LineSet]) -> o3d.geometry.LineSet:
    combined = o3d.geometry.LineSet()
    all_points: list[np.ndarray] = []
    all_lines: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    offset = 0
    for line_set in line_sets:
        if line_set is None or len(line_set.lines) == 0:
            continue
        pts = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)
        colors = np.asarray(line_set.colors)
        all_points.append(pts)
        all_lines.append(lines + offset)
        if len(colors) == len(lines):
            all_colors.append(colors)
        offset += len(pts)

    if all_points:
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        combined.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
        if all_colors:
            combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    return combined


def generate_3d_binary_mask(
    reference_pcd: o3d.geometry.PointCloud,
    source_pcd: o3d.geometry.PointCloud,
    num_layers: int = 70,
    normal_radius: float = 0.10,
    normal_max_nn: int = 30,
    min_points_per_layer: int = 3,
    alpha_multiplier: float = 6.0,
    default_window_multiplier: float = 2.5,
    vertical_window_min: float = 1.5,
    vertical_window_range: float = 3.0,
    ema: float = 0.8,
    color_from_source: bool = True,
) -> MaskExtractionResult:
    if reference_pcd.is_empty() or source_pcd.is_empty():
        return MaskExtractionResult(
            extracted=o3d.geometry.PointCloud(),
            frame=o3d.geometry.LineSet(),
        )

    ref = o3d.geometry.PointCloud(reference_pcd)
    if not ref.has_normals():
        ref.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
        )
    ref_points = np.asarray(ref.points)
    ref_normals = np.asarray(ref.normals)
    source_points = np.asarray(source_pcd.points)
    source_colors = np.asarray(source_pcd.colors) if source_pcd.has_colors() and color_from_source else None

    density = global_density(ref_points)
    z_min = float(ref_points[:, 2].min())
    z_max = float(ref_points[:, 2].max())
    z_range = max(z_max - z_min, 1e-8)
    layer_height = z_range / max(1, num_layers)
    last_expansion = layer_height * vertical_window_min

    extracted_points: list[np.ndarray] = []
    extracted_colors: list[np.ndarray] = []
    line_sets: list[o3d.geometry.LineSet] = []
    slice_records: list[SliceMask] = []

    for layer_idx in range(num_layers):
        z_start = z_min + layer_idx * layer_height
        z_end = z_min + (layer_idx + 1) * layer_height

        probe_mask = (ref_points[:, 2] >= z_start) & (ref_points[:, 2] < z_end)
        if probe_mask.any():
            abs_nz = np.abs(ref_normals[probe_mask, 2])
            verticality = 1.0 - float(np.mean(abs_nz))
            multiplier = vertical_window_min + verticality * vertical_window_range
        else:
            multiplier = default_window_multiplier

        current_expansion = layer_height * multiplier
        current_expansion = ema * current_expansion + (1.0 - ema) * last_expansion
        last_expansion = current_expansion

        z_ref_start = z_start - current_expansion
        z_ref_end = z_end + current_expansion
        ref_mask = (ref_points[:, 2] >= z_ref_start) & (ref_points[:, 2] < z_ref_end)
        if ref_mask.sum() < min_points_per_layer:
            continue

        layer_ref_2d = ref_points[ref_mask][:, :2]
        stats = analyze_2d_stats(layer_ref_2d, density)
        labels = DBSCAN(eps=stats["eps"], min_samples=1).fit(layer_ref_2d).labels_
        layer_polygons = []
        for label in set(labels):
            cluster_points = layer_ref_2d[labels == label]
            poly = adaptive_cluster_polygon(
                cluster_points,
                density=density,
                alpha_multiplier=alpha_multiplier,
            )
            if poly is not None and not poly.is_empty:
                layer_polygons.append(poly)

        if not layer_polygons:
            continue

        combined_poly = make_valid(unary_union(layer_polygons))
        color = _colormap_jet((z_start - z_min) / z_range)
        frame = create_slice_wireframe(combined_poly, z_start, z_end, color)
        if frame is not None:
            line_sets.append(frame)
        slice_records.append(
            SliceMask(
                index=layer_idx,
                z_start=z_start,
                z_end=z_end,
                z_ref_start=z_ref_start,
                z_ref_end=z_ref_end,
                polygon=combined_poly,
                color=color,
            )
        )

        source_mask = (source_points[:, 2] >= z_start) & (source_points[:, 2] < z_end)
        if not source_mask.any():
            continue
        targets = source_points[source_mask]
        inside = contains_xy(combined_poly, targets[:, 0], targets[:, 1])
        if np.any(inside):
            extracted_points.append(targets[inside])
            if source_colors is not None:
                extracted_colors.append(source_colors[source_mask][inside])

    if extracted_points:
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(np.vstack(extracted_points))
        if extracted_colors:
            result.colors = o3d.utility.Vector3dVector(np.vstack(extracted_colors))
        else:
            result.paint_uniform_color([0.2, 0.7, 0.9])
    else:
        result = o3d.geometry.PointCloud()

    return MaskExtractionResult(
        extracted=result,
        frame=combine_line_sets(line_sets),
        slices=slice_records,
    )
