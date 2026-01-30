import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from models.model import CNNMLPRegressor

from .dataset import aligned_paths, load_targets, MemmapTemperatureDataset


def _load_shape_from_info(run_dir: Path) -> tuple[int, int]:
    info_path = run_dir / "Aligned" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    h, w = info["matrix_shape"]
    return int(h), int(w)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
    ap.add_argument("--target", type=str, default="meltpoolTemp")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--limit-val-batches", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    h, w = _load_shape_from_info(run_dir)
    paths = aligned_paths(run_dir)
    y = load_targets(paths.meta_csv, args.target)

    ds = MemmapTemperatureDataset(paths.frames_memmap, y, (h, w), scale=args.scale)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * float(args.val_frac)))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = CNNMLPRegressor().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.MSELoss()

    for _ in range(int(args.epochs)):
        model.train()
        train_it = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_it += 1
            if args.limit_train_batches and train_it >= args.limit_train_batches:
                break

        model.eval()
        val_it = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                _ = loss_fn(model(xb), yb)
                val_it += 1
                if args.limit_val_batches and val_it >= args.limit_val_batches:
                    break


if __name__ == "__main__":
    main()

