from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PointCloudRepairDataset
from .loss import TopologyRepairLoss
from .model import TopologyRepairNetwork


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] device={device}")

    train_dataset = PointCloudRepairDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        num_out_points=args.num_out_points,
        epoch_iters=args.train_iters,
        split="train",
        augment=not args.no_augment,
    )
    val_dataset = PointCloudRepairDataset(
        data_dir=args.data_dir,
        num_points=args.num_points,
        num_out_points=args.num_out_points,
        epoch_iters=args.val_iters,
        split="val",
        augment=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        drop_last=False,
        pin_memory=args.pin_memory,
    )

    model = TopologyRepairNetwork(
        num_points=args.num_points,
        num_anchors=args.num_anchors,
        feature_dim=args.feature_dim,
        k=args.k_neighbors,
    ).to(device)
    print(f"[train] parameters={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    loss_fn = TopologyRepairLoss(repulsion_weight=args.repulsion_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.t0,
        T_mult=2,
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}")
        for p_in, p_gt in progress:
            p_in = p_in.to(device, non_blocking=True)
            p_gt = p_gt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                anchors, p1, p2, p3 = model(p_in)
                total_loss, coarse_loss, fine_loss = loss_fn(anchors, p1, p2, p3, p_gt)

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()
            progress.set_postfix(loss=f"{total_loss.item():.4f}", fine=f"{fine_loss.item():.5f}")

        scheduler.step()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_fine = 0.0
        with torch.no_grad():
            for p_in, p_gt in val_loader:
                p_in = p_in.to(device, non_blocking=True)
                p_gt = p_gt.to(device, non_blocking=True)
                anchors, p1, p2, p3 = model(p_in)
                total_loss, _, fine_loss = loss_fn(anchors, p1, p2, p3, p_gt)
                val_loss += total_loss.item()
                val_fine += fine_loss.item()
        val_loss /= max(1, len(val_loader))
        val_fine /= max(1, len(val_loader))
        print(
            f"[epoch {epoch + 1}] train={train_loss:.4f} val={val_loss:.4f} "
            f"val_fine_cd={val_fine:.6f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"[train] saved best_model.pth val={best_val:.4f}")
        if (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), output_dir / f"epoch_{epoch + 1}.pth")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the SCF spatial continuity filling network.")
    parser.add_argument("--data-dir", required=True, help="Directory containing .off or .ply training objects.")
    parser.add_argument("--output-dir", default="checkpoints", help="Checkpoint output directory.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--t0", type=int, default=30)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--num-out-points", type=int, default=1024)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--train-iters", type=int, default=2000)
    parser.add_argument("--val-iters", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--repulsion-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
