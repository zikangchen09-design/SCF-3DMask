from __future__ import annotations

import torch
import torch.nn as nn


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return k nearest-neighbor indices for x shaped (B, C, N)."""

    batch_size, _, num_points = x.shape
    k = min(k, num_points)
    x_t = x.transpose(2, 1)

    if num_points <= 4096:
        inner = -2 * torch.matmul(x_t, x)
        xx = torch.sum(x_t ** 2, dim=2, keepdim=True)
        dist = xx + xx.transpose(2, 1) + inner
        _, idx = dist.topk(k=k, dim=-1, largest=False)
        return idx

    chunk_size = 2048
    idx_list = []
    xx_all = torch.sum(x_t ** 2, dim=2)
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        x_chunk = x_t[:, start:end, :]
        inner = -2 * torch.matmul(x_chunk, x)
        xx_chunk = torch.sum(x_chunk ** 2, dim=2, keepdim=True)
        dist = xx_chunk + xx_all.unsqueeze(1) + inner
        _, idx_chunk = dist.topk(k=k, dim=-1, largest=False)
        idx_list.append(idx_chunk)
    return torch.cat(idx_list, dim=1).view(batch_size, num_points, k)


def edge_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size, channels, num_points = x.shape
    k = idx.shape[2]
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_flat = (idx + idx_base).reshape(-1)

    x_t = x.transpose(2, 1).contiguous()
    neighbors = x_t.view(batch_size * num_points, -1)[idx_flat, :].view(batch_size, num_points, k, channels)
    center = x_t.view(batch_size, num_points, 1, channels).expand(-1, -1, k, -1)
    feat = torch.cat([neighbors - center, center], dim=3)
    return feat.permute(0, 3, 1, 2).contiguous()


class EdgeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        feat = edge_feature(x, idx)
        return self.conv(feat).max(dim=-1)[0]


class TopologyRepairNetwork(nn.Module):
    """EdgeConv + cross-attention cascade for local topology repair."""

    def __init__(
        self,
        num_points: int = 8192,
        num_anchors: int = 64,
        feature_dim: int = 256,
        k: int = 16,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.num_anchors = num_anchors
        self.feature_dim = feature_dim
        self.k = k

        self.edge_conv1 = EdgeConvBlock(3, 64)
        self.edge_conv2 = EdgeConvBlock(64, 128)
        self.conv_high = nn.Sequential(
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(64 + 128 + 256, feature_dim, 1, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.LeakyReLU(0.2),
        )

        self.decoder_p_anchor = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_anchors * 3),
        )
        self.decoder_f_anchor = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_anchors * feature_dim),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True,
        )

        self.r1 = 4
        self.r2 = 2
        self.r3 = 2
        self.cascade_1 = nn.Sequential(
            nn.Linear(feature_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.r1 * 3),
        )
        self.level2_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True,
        )
        self.level2_feat_mlp = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.ReLU())
        self.cascade_2 = nn.Sequential(
            nn.Linear(feature_dim + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.r2 * 3),
        )
        self.level3_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            batch_first=True,
        )
        self.level3_feat_mlp = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.ReLU())
        self.cascade_3 = nn.Sequential(
            nn.Linear(feature_dim + 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.r3 * 3),
        )

        self.refine_mlp = nn.Sequential(
            nn.Conv1d(feature_dim + 3, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1),
        )
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)

    def forward(self, p_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, _ = p_in.shape
        p_in_t = p_in.transpose(1, 2)

        knn_idx = knn(p_in_t, self.k)
        f1 = self.edge_conv1(p_in_t, knn_idx)
        f2 = self.edge_conv2(f1, knn_idx)
        f3 = self.conv_high(f2)
        f_multi = torch.cat([f1, f2, f3], dim=1)
        f_per_point = self.conv_fuse(f_multi)

        f_max = torch.max(f_per_point, dim=2)[0]
        f_avg = torch.mean(f_per_point, dim=2)
        f_global = torch.cat([f_max, f_avg], dim=1)

        p_anchor = self.decoder_p_anchor(f_global).view(batch_size, self.num_anchors, 3)
        f_anchor = self.decoder_f_anchor(f_global).view(batch_size, self.num_anchors, self.feature_dim)
        keys = values = f_per_point.transpose(1, 2)
        f_enhanced, _ = self.cross_attention(query=f_anchor, key=keys, value=values)

        anchor_feat_coord = torch.cat([f_enhanced, p_anchor], dim=-1)
        p1_offsets = self.cascade_1(anchor_feat_coord).view(batch_size, self.num_anchors, self.r1, 3)
        p1 = (p_anchor.unsqueeze(2) + p1_offsets).reshape(batch_size, self.num_anchors * self.r1, 3)

        f_l1 = f_enhanced.unsqueeze(2).expand(-1, -1, self.r1, -1)
        f_l1 = f_l1.reshape(batch_size, self.num_anchors * self.r1, self.feature_dim)
        f_l1_updated, _ = self.level2_cross_attn(query=f_l1, key=keys, value=values)
        f_l1_combined = self.level2_feat_mlp(torch.cat([f_l1, f_l1_updated], dim=-1))
        p2_offsets = self.cascade_2(torch.cat([f_l1_combined, p1], dim=-1))
        p2_offsets = p2_offsets.view(batch_size, self.num_anchors * self.r1, self.r2, 3)
        p2 = (p1.unsqueeze(2) + p2_offsets).reshape(batch_size, -1, 3)

        n_l2 = self.num_anchors * self.r1 * self.r2
        f_l2 = f_l1_combined.unsqueeze(2).expand(-1, -1, self.r2, -1)
        f_l2 = f_l2.reshape(batch_size, n_l2, self.feature_dim)
        f_l2_updated, _ = self.level3_cross_attn(query=f_l2, key=keys, value=values)
        f_l2_combined = self.level3_feat_mlp(torch.cat([f_l2, f_l2_updated], dim=-1))
        p3_offsets = self.cascade_3(torch.cat([f_l2_combined, p2], dim=-1))
        p3_offsets = p3_offsets.view(batch_size, n_l2, self.r3, 3)
        p3 = (p2.unsqueeze(2) + p3_offsets).reshape(batch_size, -1, 3)

        f_refine = f_l2_combined.unsqueeze(2).expand(-1, -1, self.r3, -1)
        f_refine = f_refine.reshape(batch_size, -1, self.feature_dim)
        refine_input = torch.cat([f_refine, p3], dim=-1).transpose(1, 2)
        delta = self.refine_mlp(refine_input).transpose(1, 2)
        p3_refined = p3 + torch.tanh(delta) * 0.05
        return p_anchor, p1, p2, p3_refined
