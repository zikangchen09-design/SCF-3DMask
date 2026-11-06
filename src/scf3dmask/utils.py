from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Normalization:
    centroid: np.ndarray
    scale: float


def random_sample_points(
    points: np.ndarray,
    num_target: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    points = np.asarray(points, dtype=np.float32)
    n_points = len(points)
    if n_points == 0:
        return np.zeros((num_target, 3), dtype=np.float32)
    if n_points >= num_target:
        return points[rng.choice(n_points, num_target, replace=False)]
    extra = rng.choice(n_points, num_target - n_points, replace=True)
    return np.vstack([points, points[extra]]).astype(np.float32)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, Normalization]:
    points = np.asarray(points, dtype=np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    radius = np.sqrt(np.sum(centered * centered, axis=1)).max(initial=0.0)
    scale = float(radius) if radius > 0 else 1.0
    return centered / scale, Normalization(centroid=centroid, scale=scale)


def denormalize_points(points: np.ndarray, norm: Normalization) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) * norm.scale + norm.centroid


def estimate_spacing(points: np.ndarray, k: int = 6, default: float = 0.01) -> float:
    points = np.asarray(points)
    if len(points) <= 2:
        return default
    try:
        from sklearn.neighbors import NearestNeighbors

        n_neighbors = min(max(2, k), len(points))
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        neigh.fit(points)
        dists = neigh.kneighbors(points)[0]
        spacing = float(np.median(dists[:, -1]))
        return spacing if spacing > 0 else default
    except Exception:
        return default


def color_array(count: int, rgb: tuple[float, float, float]) -> np.ndarray:
    return np.tile(np.asarray(rgb, dtype=np.float64), (count, 1))
