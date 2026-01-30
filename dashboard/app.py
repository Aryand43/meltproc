import csv
from pathlib import Path

import numpy as np
import streamlit as st
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import CNNMLPRegressor


H = 164
W = 218


def _run_dirs(data_root: Path) -> list[str]:
    if not data_root.exists():
        return []
    out: list[str] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        aligned = p / "Aligned"
        if (aligned / "meta.csv").exists() and (aligned / "frames.uint16.memmap").exists():
            out.append(p.name)
    return out


@st.cache_data
def load_meta(meta_csv: Path) -> dict[str, np.ndarray]:
    with meta_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("meta.csv has no header")
        cols = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in cols:
                cols[k].append(row[k])
    out: dict[str, np.ndarray] = {}
    for k, v in cols.items():
        if k in {"t", "meltpoolSize", "LaserPower", "stirrerValue_1", "revolutionSpeed_1", "powderGasFlow_1", "stirrerValue_2", "revolutionSpeed_2", "powderGasFlow_2", "flowWatch", "meltpoolThreshold"}:
            out[k] = np.asarray(v, dtype=np.int64)
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


@st.cache_resource
def load_model(ckpt_path: Path, device: str) -> CNNMLPRegressor:
    model = CNNMLPRegressor()
    obj = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        state = obj["state_dict"]
    else:
        state = obj
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model


def load_frames_memmap(frames_memmap: Path, n: int) -> np.memmap:
    return np.memmap(frames_memmap, mode="r", dtype=np.uint16, shape=(n, H, W))


def predict(model: CNNMLPRegressor, x_hw: np.ndarray, device: str, scale: float) -> float:
    x = (x_hw.astype(np.float32) * scale)[None, None, :, :]
    xt = torch.from_numpy(x).to(torch.device(device))
    with torch.no_grad():
        y = model(xt).detach().cpu().numpy().reshape(-1)[0]
    return float(y)


def main() -> None:
    st.set_page_config(page_title="CNN+MLP Regression Demo", layout="wide")
    st.title("CNN + MLP Regression Demo")

    data_root = Path("Data")
    dirs = _run_dirs(data_root)
    default_run = "20230310_0040_60048406r2" if "20230310_0040_60048406r2" in dirs else (dirs[0] if dirs else "")

    with st.sidebar:
        run_dir = st.selectbox("Run", options=dirs, index=(dirs.index(default_run) if default_run in dirs else 0)) if dirs else st.text_input("Run", value=default_run)
        target = st.selectbox("Target", options=["meltpoolTemp", "meltpoolSize"])
        ckpt = st.text_input("Checkpoint path", value="")
        device = st.selectbox("Device", options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
        scale = st.number_input("Input scale", value=1.0, step=0.01)

    aligned = data_root / run_dir / "Aligned"
    meta_path = aligned / "meta.csv"
    frames_path = aligned / "frames.uint16.memmap"
    ckpt_path = Path(ckpt) if ckpt else None

    if not meta_path.exists() or not frames_path.exists():
        st.error("Missing Aligned/meta.csv or Aligned/frames.uint16.memmap for this run.")
        return
    if ckpt_path is None or not ckpt_path.exists():
        st.error("Provide a valid trained checkpoint path.")
        return

    meta = load_meta(meta_path)
    n = int(len(next(iter(meta.values()))))
    frames = load_frames_memmap(frames_path, n)
    model = load_model(ckpt_path, device)

    i = st.sidebar.number_input("Sample index i", min_value=0, max_value=max(0, n - 1), value=0, step=1)
    i = int(i)

    x_hw = np.asarray(frames[i])
    y_true = float(meta[target][i])
    y_pred = predict(model, x_hw, device, float(scale))
    err = abs(y_pred - y_true)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Frame")
        st.image(x_hw.astype(np.float32), use_container_width=True, clamp=True)
    with right:
        st.subheader("Values")
        st.metric("Ground truth", f"{y_true:.6g}")
        st.metric("Prediction", f"{y_pred:.6g}")
        st.metric("Absolute error", f"{err:.6g}")

    st.subheader("Batch view")
    n_scatter = st.sidebar.slider("Scatter samples N", min_value=100, max_value=min(5000, n), value=min(1000, n), step=100)
    seed = st.sidebar.number_input("Scatter seed", min_value=0, value=0, step=1)

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(n_scatter), replace=False) if n_scatter < n else np.arange(n)

    y_t = meta[target][idx].astype(np.float32)
    y_p = np.empty_like(y_t)
    for k, j in enumerate(idx):
        y_p[k] = predict(model, np.asarray(frames[int(j)]), device, float(scale))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_t, y_p, s=6, alpha=0.6)
    lo = float(min(y_t.min(), y_p.min()))
    hi = float(max(y_t.max(), y_p.max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    st.pyplot(fig, clear_figure=True)


if __name__ == "__main__":
    main()

