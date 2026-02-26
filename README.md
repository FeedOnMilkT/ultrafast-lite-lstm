# ultrafast-lite-lstm

TuSimple M1 baseline (single-frame) with single-GPU / DDP training.

## Train

```bash
cd /home/uceeanz/light_UFLT/ultrafast-lite-lstm

# single GPU
python scripts/train_baseline.py --data-root <tusimple_root>

# 4 GPUs (DDP)
torchrun --nproc_per_node=4 scripts/train_baseline.py --data-root <tusimple_root>
```

## Auto Eval After Training

```bash
# train + auto eval on rank0
python scripts/train_baseline.py \
  --data-root <tusimple_root> \
  --auto-eval \
  --eval-gt-file <path_to_gt_jsonl>

# optional eval output path / optional debug cap
python scripts/train_baseline.py \
  --data-root <tusimple_root> \
  --auto-eval \
  --eval-gt-file <path_to_gt_jsonl> \
  --eval-output outputs/baseline/pred_eval.jsonl \
  --eval-max-batches 20
```

## Standalone Eval

```bash
python scripts/eval_baseline.py \
  --data-root <tusimple_root> \
  --output-dir outputs/baseline \
  --gt-file <path_to_gt_jsonl>
```

## Hyperparameters

- Main defaults are in `configs/base.py` (`epochs`, `batch_size`, `learning_rate`, `optimizer`, `scheduler`, `warmup`, etc.).
- Quick overrides from CLI:
  - `--epochs`
  - `--batch-size`
  - `--max-steps`
  - `--output-dir`

## Dataset Prerequisites

- `<tusimple_root>/train_gt.txt` and `<tusimple_root>/test.txt` must exist.
- Entries referenced by those files must exist (image + mask path).
- Use `scripts/convert_tusimple.py` first if your list/mask files are not prepared.

## Checkpoints

- Each run creates a timestamp directory under `outputs/baseline/`.
- Checkpoint file format: `baseline_YYYYMMDD_HHMMSS.pth`.
