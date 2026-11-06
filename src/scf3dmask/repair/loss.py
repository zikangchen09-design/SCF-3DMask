from __future__ import annotations

import torch
import torch.nn as nn


def chamfer_distance(p1: torch.Tensor, p2: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    batch_size, n_points, _ = p1.shape
    m_points = p2.shape[1]

    if n_points * m_points <= 4096 * 4096:
        dist_sq = torch.cdist(p1, p2, p=2.0) ** 2
        min_p1_to_p2 = dist_sq.min(dim=2)[0]
        min_p2_to_p1 = dist_sq.min(dim=1)[0]
    else:
        min_p1_to_p2 = torch.zeros(batch_size, n_points, device=p1.device)
        for start in range(0, n_points, chunk):
            end = min(start + chunk, n_points)
            dist = torch.cdist(p1[:, start:end, :], p2, p=2.0) ** 2
            min_p1_to_p2[:, start:end] = dist.min(dim=2)[0]

        min_p2_to_p1 = torch.zeros(batch_size, m_points, device=p1.device)
        for start in range(0, m_points, chunk):
            end = min(start + chunk, m_points)
            dist = torch.cdist(p1, p2[:, start:end, :], p=2.0) ** 2
            min_p2_to_p1[:, start:end] = dist.min(dim=1)[0]

    return torch.mean(min_p1_to_p2.mean(dim=1) + min_p2_to_p1.mean(dim=1))


def repulsion_loss(pred: torch.Tensor, k: int = 4) -> torch.Tensor:
    _, n_points, _ = pred.shape
    k = min(k, max(1, n_points - 1))
    dist = torch.cdist(pred, pred, p=2.0)
    eye = torch.eye(n_points, device=pred.device).unsqueeze(0) * 1e10
    dist = dist + eye
    knn_dist = torch.topk(dist, k=k, dim=-1, largest=False)[0]
    return torch.mean(torch.exp(-knn_dist * 20))


class TopologyRepairLoss(nn.Module):
    def __init__(self, repulsion_weight: float = 0.5) -> None:
        super().__init__()
        self.repulsion_weight = repulsion_weight

    def forward(
        self,
        anchors: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor,
        gt_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_anchors = chamfer_distance(anchors, gt_points)
        loss_p1 = chamfer_distance(p1, gt_points)
        loss_p2 = chamfer_distance(p2, gt_points)
        loss_p3 = chamfer_distance(p3, gt_points)
        cd_total = 0.1 * loss_anchors + 0.5 * loss_p1 + 1.0 * loss_p2 + 2.0 * loss_p3
        total = cd_total + self.repulsion_weight * repulsion_loss(p3, k=4)
        return total, loss_anchors, loss_p3
