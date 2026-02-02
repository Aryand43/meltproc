# meltproc

Minimal end-to-end pipeline for meltpool runs:

- **Alignment (sanity/validation)**: `Test/align_run_frames_to_data.py`
  - Reads: `Data/<run_dir>/Data.dat` and `Data/<run_dir>/Frames/*.dat`
  - Writes aligned artifacts to: `Data/<run_dir>/Aligned/`
    - `frames.uint16.memmap`
    - `meta.csv`
    - `mapping.csv`, `frames.txt`, `info.json`

- **Model definitions (training-ready)**: `models/`
  - `models/cnn.py`: ResNet-18-style 1-channel CNN → 512-d embedding
  - `models/mlp.py`: 512 → 128 → 1 regression head
  - `models/model.py`: `CNNMLPRegressor` (CNN + MLP)

- **Training (minimal)**: `train/train.py`
  - Reads: `Data/<run_dir>/Aligned/frames.uint16.memmap` and `meta.csv`
  - Trains `CNNMLPRegressor` to predict a chosen target column (e.g. `meltpoolTemp`, `meltpoolSize`)

`Data/` is expected to be local-only (not committed).
