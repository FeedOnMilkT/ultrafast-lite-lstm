# ultrafast-lite-lstm

Step-1 skeleton for TuSimple M1 baseline.

## Quick start

```bash
cd /home/uceeanz/light_UFLT/ultrafast-lite-lstm

# train (synthetic dry-run)
/home/uceeanz/venvs/uflt/bin/python scripts/train_baseline.py --dry-run --epochs 1 --max-steps 2

# eval (synthetic dry-run, auto-load latest .pth)
/home/uceeanz/venvs/uflt/bin/python scripts/eval_baseline.py --dry-run --max-batches 1
```

## Notes

- This step is intentionally minimal and runnable.
- Real TuSimple label parsing in `lanes/data/tusimple_dataset.py` is marked with `TODO(step-2)`.
- Keep this as the learning baseline before migrating full reference logic.
- Checkpoint policy:
  - Every training run creates a new timestamp folder under `outputs/baseline/`.
  - Checkpoint filename format: `baseline_YYYYMMDD_HHMMSS.pth`.
  - Eval uses the latest `.pth` by default if `--checkpoint` is not provided.
