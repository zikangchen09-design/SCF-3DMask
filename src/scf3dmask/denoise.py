from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


@dataclass(frozen=True)
class SORConfig:
    name: str
    k: int
    std_ratio: float


@dataclass(frozen=True)
class RORConfig:
    name: str
    radius: float
    min_points: int


@dataclass(frozen=True)
class CascadeConfig:
    name: str
    levels: int
    decay: float


@dataclass
class DenoiseResult:
    score: float
    pcd: o3d.geometry.PointCloud
    path: str
    point_count: int


class NMRFilter:
    """N-M-R cascaded filtering with branch scoring.

    N-M-R denotes statistical noise reduction, radius-adaptive filtering, and
    multi-level cascaded radius filtering.
    """

    def __init__(
        self,
        sor_configs: Optional[list[SORConfig]] = None,
        ror_configs: Optional[list[RORConfig]] = None,
        cascade_configs: Optional[list[CascadeConfig]] = None,
    ) -> None:
        self.sor_configs = sor_configs or [
            SORConfig("SOR_Medium", k=30, std_ratio=1.0),
            SORConfig("SOR_Heavy", k=15, std_ratio=0.5),
            SORConfig("SOR_Extreme", k=10, std_ratio=0.1),
        ]
        self.ror_configs = ror_configs or [
            RORConfig("ROR_Standard", radius=0.05, min_points=15),
            RORConfig("ROR_Strict", radius=0.03, min_points=25),
            RORConfig("ROR_Extreme", radius=0.02, min_points=40),
            RORConfig("ROR_Brutal", radius=0.015, min_points=60),
        ]
        self.cascade_configs = cascade_configs or [
            CascadeConfig("Cas_3_Level", levels=3, decay=0.10),
            CascadeConfig("Cas_7_Level", levels=7, decay=0.15),
            CascadeConfig("Cas_10_Level", levels=10, decay=0.20),
            CascadeConfig("Cas_15_Level", levels=15, decay=0.25),
        ]

    @staticmethod
    def _apply_sor(pcd: o3d.geometry.PointCloud, params: SORConfig) -> o3d.geometry.PointCloud:
        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=params.k,
            std_ratio=params.std_ratio,
        )
        return pcd.select_by_index(ind)

    @staticmethod
    def _apply_ror(pcd: o3d.geometry.PointCloud, params: RORConfig) -> o3d.geometry.PointCloud:
        _, ind = pcd.remove_radius_outlier(
            nb_points=params.min_points,
            radius=params.radius,
        )
        return pcd.select_by_index(ind)

    @staticmethod
    def _apply_cascade(
        pcd: o3d.geometry.PointCloud,
        base_radius: float,
        base_points: int,
        params: CascadeConfig,
    ) -> o3d.geometry.PointCloud:
        filtered = pcd
        for level in range(params.levels):
            radius = base_radius * (1.0 - level * params.decay)
            min_points = int(base_points * (1.0 + level * params.decay))
            if radius <= 0 or min_points <= 0:
                break
            _, ind = filtered.remove_radius_outlier(nb_points=min_points, radius=radius)
            filtered = filtered.select_by_index(ind)
            if len(filtered.points) < 50:
                break
        return filtered

    @staticmethod
    def score_result(
        pcd: o3d.geometry.PointCloud,
        initial_point_count: int,
        k: int = 20,
    ) -> float:
        current_count = len(pcd.points)
        if current_count < 50 or initial_point_count <= 0:
            return -1.0

        retention = current_count / initial_point_count
        if retention < 0.01:
            return -1.0

        points = np.asarray(pcd.points)
        actual_k = min(k, current_count - 1)
        if actual_k < 3:
            return -1.0

        tree = KDTree(points)
        distances, indices = tree.query(points, k=actual_k)
        density_score = 1.0 / (1.0 + np.var(np.mean(distances, axis=1)))

        if not pcd.has_normals():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=actual_k))
        normals = np.asarray(pcd.normals)
        neighbor_normals = normals[indices]
        local_similarity = np.sum(neighbor_normals * normals[:, np.newaxis, :], axis=2)
        smoothness_score = 1.0 / (1.0 + np.mean(np.var(local_similarity, axis=1)))

        return 0.45 * density_score + 0.50 * smoothness_score + 0.05 * retention

    def _evaluate_branch(
        self,
        pcd_ror: o3d.geometry.PointCloud,
        ror_config: RORConfig,
        cascade_config: CascadeConfig,
        sor_name: str,
        initial_count: int,
    ) -> DenoiseResult:
        pcd_final = self._apply_cascade(
            pcd_ror,
            ror_config.radius,
            ror_config.min_points,
            cascade_config,
        )
        score = self.score_result(pcd_final, initial_count)
        path = f"{sor_name} -> {ror_config.name} -> {cascade_config.name}"
        return DenoiseResult(score=score, pcd=pcd_final, path=path, point_count=len(pcd_final.points))

    def run(
        self,
        raw_pcd: o3d.geometry.PointCloud,
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ) -> list[DenoiseResult]:
        initial_count = len(raw_pcd.points)
        if initial_count == 0:
            return []

        futures = []
        results: list[DenoiseResult] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for sor_config in self.sor_configs:
                if verbose:
                    print(f"[NMR] SOR branch: {sor_config.name}")
                pcd_sor = self._apply_sor(raw_pcd, sor_config)
                for ror_config in self.ror_configs:
                    pcd_ror = self._apply_ror(pcd_sor, ror_config)
                    for cascade_config in self.cascade_configs:
                        futures.append(
                            executor.submit(
                                self._evaluate_branch,
                                pcd_ror,
                                ror_config,
                                cascade_config,
                                sor_config.name,
                                initial_count,
                            )
                        )

            for future in as_completed(futures):
                result = future.result()
                if result.score > 0:
                    results.append(result)

        results.sort(key=lambda item: item.score, reverse=True)
        if verbose and results:
            best = results[0]
            print(f"[NMR] Best branch: {best.path} | score={best.score:.4f} | points={best.point_count}")
        return results
