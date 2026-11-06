from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import open3d as o3d

    HAS_O3D = True
except ImportError:
    HAS_O3D = False

from scf3dmask.utils import random_sample_points


def read_off(filepath: str | Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
        if first == "OFF":
            counts = handle.readline().strip().split()
        elif first.startswith("OFF"):
            counts = first[3:].strip().split()
        else:
            raise ValueError(f"Not an OFF file: {filepath}")
        n_verts = int(counts[0])
        n_faces = int(counts[1])
        vertices = []
        for _ in range(n_verts):
            parts = handle.readline().strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        faces = []
        for _ in range(n_faces):
            parts = handle.readline().strip().split()
            if len(parts) >= 4:
                faces.append((int(parts[1]), int(parts[2]), int(parts[3])))
    return np.asarray(vertices, dtype=np.float32), faces


def sample_points_from_mesh(vertices: np.ndarray, faces: list[tuple[int, int, int]], n: int) -> np.ndarray:
    if not faces:
        return random_sample_points(vertices, n)
    v0 = vertices[[face[0] for face in faces]]
    v1 = vertices[[face[1] for face in faces]]
    v2 = vertices[[face[2] for face in faces]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 1e-10:
        return random_sample_points(vertices, n)
    probabilities = areas / total
    chosen = np.random.choice(len(faces), size=n, p=probabilities, replace=True)
    r1 = np.random.uniform(0, 1, size=n)
    r2 = np.random.uniform(0, 1, size=n)
    mask = (r1 + r2) > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    a, b, c = v0[chosen], v1[chosen], v2[chosen]
    return ((1 - r1[:, None] - r2[:, None]) * a + r1[:, None] * b + r2[:, None] * c).astype(np.float32)


def rotation_matrix(axis: int, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 0:
        return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == 1:
        return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def augment_points(points: np.ndarray) -> np.ndarray:
    rotation = np.eye(3, dtype=np.float32)
    for axis in random.sample([0, 1, 2], random.randint(1, 3)):
        rotation = rotation @ rotation_matrix(axis, np.random.uniform(0, 2 * np.pi))
    points = (points @ rotation).astype(np.float32)
    points *= np.random.uniform(0.85, 1.15)
    if random.random() < 0.3:
        points += np.clip(np.random.randn(*points.shape) * 0.003, -0.01, 0.01).astype(np.float32)
    return points.astype(np.float32)


class PointCloudRepairDataset(Dataset):
    """Self-supervised repair-patch dataset.

    The dataset samples clean point sets and creates damaged inputs by removing
    sparse, low-density, or local patch regions. Removed points are the repair
    target.
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_points: int = 8192,
        num_out_points: int = 1024,
        epoch_iters: int = 2000,
        split: str = "train",
        augment: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.num_points = num_points
        self.num_out_points = num_out_points
        self.total_points = num_points + num_out_points
        self.epoch_iters = epoch_iters
        self.split = split
        self.augment = augment
        self.file_paths: list[tuple[Path, str]] = []
        self.preloaded = None
        self._collect_paths()

    def _collect_paths(self) -> None:
        skip_folder = "test" if self.split == "train" else "train"
        for root, directories, files in os.walk(self.data_dir):
            # Prune the excluded split so nested object folders cannot leak in.
            directories[:] = [name for name in directories if name != skip_folder]
            if Path(root).name == skip_folder:
                directories[:] = []
                continue
            for name in files:
                path = Path(root) / name
                suffix = path.suffix.lower()
                if suffix == ".off":
                    self.file_paths.append((path, "off"))
                elif suffix == ".ply":
                    self.file_paths.append((path, "ply"))
        if not self.file_paths:
            raise RuntimeError(f"No .off or .ply files found in {self.data_dir}")
        if len(self.file_paths) <= 50:
            self._preload()

    def _preload(self) -> None:
        self.preloaded = []
        for path, kind in self.file_paths:
            try:
                if kind == "off":
                    vertices, faces = read_off(path)
                    if len(vertices) >= 100:
                        self.preloaded.append(("mesh", vertices, faces))
                elif HAS_O3D:
                    points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
                    if len(points) >= 100:
                        if len(points) > 100000:
                            points = random_sample_points(points, 100000)
                        self.preloaded.append(("points", points, None))
            except Exception:
                continue

    def _load_one(self, path: Path, kind: str) -> np.ndarray | None:
        if kind == "off":
            vertices, faces = read_off(path)
            if len(vertices) < 100:
                return None
            return sample_points_from_mesh(vertices, faces, self.total_points)
        if not HAS_O3D:
            return None
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)
        if len(points) < 100:
            return None
        return random_sample_points(points, self.total_points)

    def _get_points_from_preloaded(self, idx: int) -> np.ndarray:
        kind, data, extra = self.preloaded[idx]
        if kind == "mesh":
            return sample_points_from_mesh(data, extra, self.total_points)
        return random_sample_points(data, self.total_points)

    def _destroy_sor(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not HAS_SCIPY:
            return self._destroy_density_weighted(points)
        tree = cKDTree(points)
        k = min(20, len(points) - 1)
        dists, _ = tree.query(points, k=k + 1)
        avg_dists = dists[:, 1:].mean(axis=1)
        sorted_idx = np.argsort(avg_dists)
        remove = sorted_idx[-self.num_out_points :]
        keep = sorted_idx[: -self.num_out_points]
        return points[keep], points[remove]

    def _destroy_density_weighted(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not HAS_SCIPY:
            idx = np.arange(len(points))
            np.random.shuffle(idx)
            return points[idx[self.num_out_points :]], points[idx[: self.num_out_points]]

        tree = cKDTree(points)
        k = min(12, len(points) - 1)
        dists, _ = tree.query(points, k=k + 1)
        avg_dists = dists[:, 1:].mean(axis=1)
        weights = avg_dists ** np.random.uniform(0.5, 2.0)
        weights = weights / weights.sum()
        remove = np.random.choice(len(points), self.num_out_points, replace=False, p=weights)
        keep = np.ones(len(points), dtype=bool)
        keep[remove] = False
        return points[keep], points[remove]

    def _destroy_small_patches_plus_density(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_patches = random.randint(3, 8)
        patch_budget = int(self.num_out_points * random.uniform(0.3, 0.6))
        density_budget = self.num_out_points - patch_budget
        per_patch = max(1, patch_budget // num_patches)
        remove_mask = np.zeros(len(points), dtype=bool)
        removed = 0

        for patch_idx in range(num_patches):
            target = patch_budget - removed if patch_idx == num_patches - 1 else per_patch
            available = np.where(~remove_mask)[0]
            if target <= 0 or len(available) < target:
                break
            seed = points[available[np.random.randint(len(available))]]
            dists = np.linalg.norm(points[available] - seed, axis=1)
            closest = np.argsort(dists)[:target]
            remove_mask[available[closest]] = True
            removed = int(remove_mask.sum())

        if HAS_SCIPY and density_budget > 0:
            remaining = np.where(~remove_mask)[0]
            if len(remaining) > density_budget:
                tree = cKDTree(points[remaining])
                k = min(12, len(remaining) - 1)
                dists, _ = tree.query(points[remaining], k=k + 1)
                avg_d = dists[:, 1:].mean(axis=1)
                extra_remove = np.argsort(avg_d)[-density_budget:]
                remove_mask[remaining[extra_remove]] = True

        return (
            random_sample_points(points[~remove_mask], self.num_points),
            random_sample_points(points[remove_mask], self.num_out_points),
        )

    def __len__(self) -> int:
        return self.epoch_iters

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = None
        if self.preloaded is not None:
            patch = self._get_points_from_preloaded(np.random.randint(len(self.preloaded)))
        else:
            for _ in range(10):
                path, kind = self.file_paths[np.random.randint(len(self.file_paths))]
                patch = self._load_one(path, kind)
                if patch is not None:
                    break
        if patch is None:
            patch = np.random.randn(self.total_points, 3).astype(np.float32)

        if self.augment:
            patch = augment_points(patch)

        mode = random.random()
        if mode < 0.40:
            p_in, p_gt = self._destroy_sor(patch)
        elif mode < 0.75:
            p_in, p_gt = self._destroy_density_weighted(patch)
        else:
            p_in, p_gt = self._destroy_small_patches_plus_density(patch)

        centroid = p_in.mean(axis=0)
        p_in = p_in - centroid
        p_gt = p_gt - centroid
        scale = np.sqrt(np.sum(p_in * p_in, axis=1)).max(initial=0.0)
        if scale > 0:
            p_in = p_in / scale
            p_gt = p_gt / scale

        return torch.tensor(p_in, dtype=torch.float32), torch.tensor(p_gt, dtype=torch.float32)
