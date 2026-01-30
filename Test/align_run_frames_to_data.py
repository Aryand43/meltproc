"""
Align temperature matrices in Frames/*.dat with metadata rows in Data.dat.

This file lives in Test/ because it is used for:
- data sanity checks
- alignment validation
- visualization-based verification

Constraints:
- No training code.
- No models.
- No preprocessing pipelines beyond alignment itself.

This script:
- Reads Data.dat starting from row 46 onward (1-based). Row 46 is the column header.
- Loads metadata rows (row 47..end) using the known column order.
- Lists Frames/*.dat, extracts the numeric suffix after "__" in the filename, and sorts by it.
- Creates a 1-to-1, in-order pairing by matching each metadata time `t` to the closest
  frame suffix while enforcing strictly increasing frame indices.
- Streams the matched matrices into a disk-backed numpy memmap and writes manifests.
"""

from __future__ import annotations

import csv
import json
import re
from bisect import bisect_left
from pathlib import Path
from typing import List, Tuple

import numpy as np


COLUMNS: List[str] = [
    "t",
    "x",
    "y",
    "z",
    "a",
    "c",
    "meltpoolSize",
    "meltpoolTemp",
    "LaserPower",
    "stirrerValue_1",
    "revolutionSpeed_1",
    "powderGasFlow_1",
    "stirrerValue_2",
    "revolutionSpeed_2",
    "powderGasFlow_2",
    "flowWatch",
    "meltpoolThreshold",
    "protectionGlasTemperature",
]


def parse_data_dat(data_path: Path) -> List[List[str]]:
    """
    Returns rows as list-of-fields (strings) in COLUMNS order.

    Reads from row 46 onward (1-based). Row 46 is the CSV header line.
    Data rows are whitespace-separated.
    """
    lines = data_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # 1-based row 46 => 0-based index 45
    start_idx = 45
    if len(lines) <= start_idx:
        raise ValueError(f"{data_path} has only {len(lines)} lines; expected >= 46.")

    sub = lines[start_idx:]
    # Drop comments/blanks
    sub = [ln for ln in sub if ln.strip() and not ln.lstrip().startswith("#")]
    if not sub:
        raise ValueError(f"{data_path} has no content after row 46.")

    # First line is the column header with commas; drop it.
    if "," in sub[0]:
        sub = sub[1:]

    rows: List[List[str]] = []
    for i, ln in enumerate(sub, start=47):  # 1-based line number for readability
        parts = ln.split()
        if len(parts) != len(COLUMNS):
            raise ValueError(
                f"Unexpected field count at Data.dat line ~{i}: "
                f"got {len(parts)}, expected {len(COLUMNS)}. Line: {ln!r}"
            )
        rows.append(parts)
    return rows


_FRAME_SUFFIX_RE = re.compile(r"__(\d+)\.dat$", re.IGNORECASE)


def list_frames(frames_dir: Path) -> Tuple[List[Path], List[int]]:
    frames = sorted(frames_dir.glob("*.dat"))
    if not frames:
        raise FileNotFoundError(f"No .dat files found in {frames_dir}")

    suffixes: List[int] = []
    for p in frames:
        m = _FRAME_SUFFIX_RE.search(p.name)
        if not m:
            raise ValueError(f"Frame filename missing __<number>.dat suffix: {p.name}")
        suffixes.append(int(m.group(1)))

    # Sort by suffix (then name as tie-breaker)
    order = sorted(range(len(frames)), key=lambda i: (suffixes[i], frames[i].name))
    frames = [frames[i] for i in order]
    suffixes = [suffixes[i] for i in order]
    return frames, suffixes


def monotonic_nearest_match(times: List[int], frame_suffixes: List[int]) -> List[int]:
    """
    For each time in `times` (in order), pick a unique frame index in increasing order.

    Uses nearest suffix by absolute difference, with a strict increasing constraint.
    """
    if not times:
        return []
    if not frame_suffixes:
        raise ValueError("No frame suffixes provided.")

    chosen: List[int] = []
    prev_idx = -1
    n = len(frame_suffixes)

    for t in times:
        j = bisect_left(frame_suffixes, t)
        candidates = []
        if 0 <= j < n:
            candidates.append(j)
        if 0 <= j - 1 < n:
            candidates.append(j - 1)

        # Choose closest candidate; break ties by smaller index (earlier frame).
        candidates = sorted(candidates, key=lambda idx: (abs(frame_suffixes[idx] - t), idx))
        idx = candidates[0]

        # Enforce strictly increasing.
        if idx <= prev_idx:
            idx = prev_idx + 1
        if idx >= n:
            raise ValueError(
                "Not enough frames to assign uniquely while keeping order. "
                f"Stopped at time={t}."
            )

        chosen.append(idx)
        prev_idx = idx

    if len(set(chosen)) != len(chosen):
        raise AssertionError("Non-unique frame indices chosen; alignment failed.")
    return chosen


def main() -> None:
    run_dir = Path(r"c:\Users\AD\Desktop\meltproc\Data\20230310_0040_60048406r2")
    data_path = run_dir / "Data.dat"
    frames_dir = run_dir / "Frames"
    out_dir = run_dir / "Aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = parse_data_dat(data_path)
    # Extract times (t column) as ints
    times = [int(r[0]) for r in meta_rows]

    frames, suffixes = list_frames(frames_dir)

    # Build 1-to-1 monotonic pairing (selects a subset of frames if Frames/ has more).
    chosen_idxs = monotonic_nearest_match(times, suffixes)
    aligned_frames = [frames[i] for i in chosen_idxs]
    aligned_suffixes = [suffixes[i] for i in chosen_idxs]

    # Load first matrix to determine shape
    sample = np.loadtxt(aligned_frames[0], dtype=np.uint16)
    if sample.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {aligned_frames[0]}, got shape {sample.shape}")
    rows, cols = sample.shape

    # Prepare outputs
    meta_csv = out_dir / "meta.csv"
    frames_txt = out_dir / "frames.txt"
    mapping_csv = out_dir / "mapping.csv"
    info_json = out_dir / "info.json"
    frames_bin = out_dir / "frames.uint16.memmap"

    # Write manifests
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(meta_rows)

    with frames_txt.open("w", encoding="utf-8") as f:
        for p in aligned_frames:
            f.write(str(p.relative_to(run_dir)).replace("\\", "/") + "\n")

    with mapping_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "t", "frame_file", "frame_suffix", "t_minus_suffix", "chosen_frame_index"])
        for i, (row, p, suf, idx) in enumerate(zip(meta_rows, aligned_frames, aligned_suffixes, chosen_idxs)):
            t = int(row[0])
            w.writerow([i, t, p.name, suf, t - suf, idx])

    info = {
        "run_dir": str(run_dir),
        "data_rows": len(meta_rows),
        "frames_total": len(frames),
        "frames_aligned": len(aligned_frames),
        "matrix_shape": [int(rows), int(cols)],
        "dtype": "uint16",
        "outputs": {
            "meta_csv": str(meta_csv),
            "frames_txt": str(frames_txt),
            "mapping_csv": str(mapping_csv),
            "frames_memmap": str(frames_bin),
        },
    }
    info_json.write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Stream matrices into memmap
    mm = np.memmap(frames_bin, mode="w+", dtype=np.uint16, shape=(len(aligned_frames), rows, cols))
    mm[0] = sample
    mm.flush()

    for i, p in enumerate(aligned_frames[1:], start=1):
        mat = np.loadtxt(p, dtype=np.uint16)
        if mat.shape != (rows, cols):
            raise ValueError(f"Shape mismatch at i={i}: {p} has {mat.shape}, expected {(rows, cols)}")
        mm[i] = mat
        if i % 250 == 0:
            mm.flush()
            print(f"loaded {i}/{len(aligned_frames)-1} frames...")
    mm.flush()

    print("done")  # sentinel


if __name__ == "__main__":
    main()

