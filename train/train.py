import argparse
import csv
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


def _write_target_norm(aligned_dir: Path, mean: np.ndarray, std: np.ndarray) -> None:
    out = aligned_dir / "target_norm.json"
    payload = {
        "targets": ["meltpoolTemp", "meltpoolSize"],
        "mean": [float(mean[0]), float(mean[1])],
        "std": [float(std[0]), float(std[1])],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_training_log(
    log_path: Path,
    epoch: int,
    total_loss: float,
    temp_loss: float,
    size_loss: float,
    rmse_temp: float,
    rmse_size: float,
) -> None:
    exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["epoch", "total_loss", "temp_loss", "size_loss", "rmse_temp", "rmse_size"])
        w.writerow([epoch, total_loss, temp_loss, size_loss, rmse_temp, rmse_size])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True)
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
    y = load_targets(paths.meta_csv, ["meltpoolTemp", "meltpoolSize"])

    ds = MemmapTemperatureDataset(paths.frames_memmap, y, (h, w), scale=args.scale)

    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(ds))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * float(args.val_frac)))
    val_idx = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    y_train = y[np.asarray(train_idx, dtype=np.int64)]
    mean = y_train.mean(axis=0)
    std = y_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    _write_target_norm(paths.aligned_dir, mean, std)

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

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    log_path = paths.aligned_dir / "training_log.csv"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_it = 0
        sum_total = 0.0
        sum_temp = 0.0
        sum_size = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            yb_n = (yb - mean_t) / std_t
            temp_loss = loss_fn(pred[:, 0], yb_n[:, 0])
            size_loss = loss_fn(pred[:, 1], yb_n[:, 1])
            loss = temp_loss + size_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_it += 1
            n_batches += 1
            sum_total += float(loss.detach().cpu())
            sum_temp += float(temp_loss.detach().cpu())
            sum_size += float(size_loss.detach().cpu())
            if args.limit_train_batches and train_it >= args.limit_train_batches:
                break

        model.eval()
        val_it = 0
        sse_temp = 0.0
        sse_size = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_n = model(xb)
                pred = pred_n * std_t + mean_t
                diff = pred - yb
                sse_temp += float((diff[:, 0] ** 2).sum().detach().cpu())
                sse_size += float((diff[:, 1] ** 2).sum().detach().cpu())
                n_val_samples += int(yb.shape[0])
                val_it += 1
                if args.limit_val_batches and val_it >= args.limit_val_batches:
                    break

        total_loss = sum_total / max(1, n_batches)
        temp_loss = sum_temp / max(1, n_batches)
        size_loss = sum_size / max(1, n_batches)
        rmse_temp = float(np.sqrt(sse_temp / max(1, n_val_samples)))
        rmse_size = float(np.sqrt(sse_size / max(1, n_val_samples)))
        _append_training_log(log_path, epoch, total_loss, temp_loss, size_loss, rmse_temp, rmse_size)


if __name__ == "__main__":
    main()

