# ultrafast-lite-lstm

Unified TuSimple training/eval entry for both baseline (single-frame) and LSTM (temporal) models.
Default config file: `configs/tusimple_experiment.py`.

## Train

```bash
cd /home/uceeanz/light_UFLT/ultrafast-lite-lstm

# one command for both models (switch by editing configs/tusimple_experiment.py: model_type)
python scripts/train.py \
  --data-root <tusimple_root>

# 4 GPUs (DDP) example
torchrun --nproc_per_node=4 scripts/train.py \
  --data-root <tusimple_root> \
  --output-dir outputs

# optional: explicitly select a config module
python scripts/train.py --config configs.tusimple_lstm --data-root <tusimple_root>
```

## Auto Eval After Training

```bash
# train + auto eval on rank0
python scripts/train.py \
  --data-root <tusimple_root> \
  --auto-eval \
  --eval-gt-file <path_to_gt_jsonl>

# optional eval output path / optional debug cap
python scripts/train.py \
  --data-root <tusimple_root> \
  --auto-eval \
  --eval-gt-file <path_to_gt_jsonl> \
  --eval-output outputs/pred_eval.jsonl \
  --eval-max-batches 20
```

## Standalone Eval

```bash
python scripts/eval.py \
  --data-root <tusimple_root> \
  --output-dir outputs \
  --gt-file <path_to_gt_jsonl>
```

## Hyperparameters

- Main defaults are in `configs/base.py` (`epochs`, `batch_size`, `learning_rate`, `optimizer`, `scheduler`, `warmup`, etc.).
- Daily experiment switch is configured in `configs/tusimple_experiment.py`:
  - `model_type = "baseline"` or `"lstm"`
  - temporal fields (`sequence_length`, `lstm_hidden_size`, etc.)
- Quick overrides from CLI:
  - `--epochs`
  - `--batch-size`
  - `--max-steps`
  - `--output-dir`
  - `--model-type` (optional override; otherwise uses config)
  - `--sequence-length`, `--sequence-stride`
  - `--lstm-hidden-size`, `--lstm-num-layers`, `--lstm-dropout`
  - `--save-every-epochs` (default `10`, set `<=0` to disable periodic checkpoints)

## Dataset Prerequisites

- `<tusimple_root>/train_gt.txt` and `<tusimple_root>/test.txt` must exist.
- Entries referenced by those files must exist (image + mask path).
- Use `scripts/convert_tusimple.py` first if your list/mask files are not prepared.

## Checkpoints

- Each run creates a model-tagged timestamp directory under `output_dir/`:
  - `baseline_YYYYMMDD_HHMMSS`
  - `lstm_YYYYMMDD_HHMMSS`
- Checkpoint file format matches the run dir name:
  - `baseline_YYYYMMDD_HHMMSS.pth`
  - `lstm_YYYYMMDD_HHMMSS.pth`
- Periodic checkpoints are saved as `epoch_XXX.pth` every `--save-every-epochs` epochs.

## Progress Bars

- Training shows one progress bar per epoch on rank0.
- Evaluation (`scripts/eval.py`) shows an inference progress bar with running frames/FPS.

## Legacy Scripts

- Legacy split scripts are kept for compatibility:
  - `scripts/train_baseline.py`, `scripts/eval_baseline.py`
  - `scripts/train_lstm.py`, `scripts/eval_lstm.py`
- New development should use:
  - `scripts/train.py`
  - `scripts/eval.py`
