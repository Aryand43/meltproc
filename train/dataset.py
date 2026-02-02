import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AlignedRunPaths:
    aligned_dir: Path
    frames_memmap: Path
    meta_csv: Path


def aligned_paths(run_dir: Path) -> AlignedRunPaths:
    aligned_dir = run_dir / "Aligned"
    return AlignedRunPaths(
        aligned_dir=aligned_dir,
        frames_memmap=aligned_dir / "frames.uint16.memmap",
        meta_csv=aligned_dir / "meta.csv",
    )


def load_targets(meta_csv: Path, target_columns: list[str]) -> np.ndarray:
    with meta_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{meta_csv} missing header")
        for col in target_columns:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing target column {col!r} in {meta_csv}")
        ys = [[float(row[col]) for col in target_columns] for row in reader]
    return np.asarray(ys, dtype=np.float32)


class MemmapTemperatureDataset(Dataset):
    def __init__(
        self,
        frames_memmap: Path,
        targets: np.ndarray,
        shape_hw: tuple[int, int],
        scale: float = 1.0,
    ) -> None:
        self.frames_memmap = frames_memmap
        if targets.ndim != 2:
            raise ValueError(f"targets must be (N, D), got shape {targets.shape}")
        self.targets = targets
        self.h, self.w = shape_hw
        self.scale = float(scale)
        self._mm = np.memmap(
            self.frames_memmap,
            mode="r",
            dtype=np.uint16,
            shape=(len(self.targets), self.h, self.w),
        )

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.asarray(self._mm[idx], dtype=np.float32) * self.scale
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(np.asarray(self.targets[idx], dtype=np.float32))
        return x, y

