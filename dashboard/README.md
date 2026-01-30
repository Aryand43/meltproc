Run:

```bash
streamlit run dashboard/app.py
```

The app reads:

- `Data/<run_dir>/Aligned/meta.csv`
- `Data/<run_dir>/Aligned/frames.uint16.memmap`

Provide a trained checkpoint path in the sidebar. The checkpoint must be a PyTorch `state_dict` or a dict containing `state_dict`.

