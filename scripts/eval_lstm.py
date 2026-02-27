from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tusimple_baseline import get_config
from lanes.data.sequence_loader import build_test_sequence_loader
from lanes.eval.postprocess import generate_tusimple_lines
from lanes.eval.tusimple_wrapper import build_submit_record, dump_submit_records, evaluate_submit_file
from lanes.models.parsing_temporal_lstm import ParsingTemporalLSTM
from lanes.utils.checkpoint import find_latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Eval temporal LSTM model (full-eval by default).")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/lstm/pred.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint root, e.g. outputs/lstm")
    parser.add_argument("--gt-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional debug cap. Default: full dataset")
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--sequence-stride", type=int, default=None)
    parser.add_argument("--sequence-test-list", type=str, default=None)

    parser.add_argument("--lstm-hidden-size", type=int, default=None)
    parser.add_argument("--lstm-num-layers", type=int, default=None)
    parser.add_argument("--lstm-dropout", type=float, default=None)
    parser.add_argument("--head-hidden-features", type=int, default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    return parser.parse_args()


def _estimate_params_and_flops(model, cfg, device: torch.device):
    params = sum(p.numel() for p in model.parameters())
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        dummy = torch.randn(1, cfg.sequence_length, 3, cfg.input_height, cfg.input_width, device=device)
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as flop_counter:
                _ = model(dummy)
        flops = int(flop_counter.get_total_flops())
        if model_was_training:
            model.train()
    except Exception:
        flops = None
    return params, flops


def _normalize_metrics(metrics):
    out = {"Accuracy": float("nan"), "FP": float("nan"), "FN": float("nan")}
    for item in metrics:
        name = item.get("name")
        if name in out:
            out[name] = float(item.get("value", float("nan")))
    return out


def _build_model(cfg):
    return ParsingTemporalLSTM(
        griding_num=cfg.griding_num,
        num_rows=cfg.num_rows,
        num_lanes=cfg.num_lanes,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_hidden_features=cfg.cls_hidden_features,
        pretrained_backbone=cfg.pretrained_backbone,
    )


def main():
    args = parse_args()
    cfg = get_config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    else:
        cfg.output_dir = "outputs/lstm"
    if args.dry_run:
        cfg.dry_run = True
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    if args.sequence_length is not None:
        cfg.sequence_length = args.sequence_length
    if args.sequence_stride is not None:
        cfg.sequence_stride = args.sequence_stride
    if args.sequence_test_list is not None:
        cfg.sequence_test_list = args.sequence_test_list

    if args.lstm_hidden_size is not None:
        cfg.lstm_hidden_size = args.lstm_hidden_size
    if args.lstm_num_layers is not None:
        cfg.lstm_num_layers = args.lstm_num_layers
    if args.lstm_dropout is not None:
        cfg.lstm_dropout = args.lstm_dropout
    if args.head_hidden_features is not None:
        cfg.cls_hidden_features = args.head_hidden_features
    if args.no_pretrained_backbone:
        cfg.pretrained_backbone = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg).to(device)

    ckpt = args.checkpoint or find_latest_checkpoint(cfg.output_dir)
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    load_checkpoint(ckpt, model, optimizer=None, map_location=device)
    print(f"loaded checkpoint: {ckpt}")

    model.eval()
    params, flops = _estimate_params_and_flops(model, cfg, device)
    loader = build_test_sequence_loader(cfg)
    total_batches = len(loader) if args.max_batches is None else min(len(loader), args.max_batches)
    records = []
    infer_frames = 0
    infer_seconds = 0.0
    with tqdm(total=total_batches, desc="Eval", dynamic_ncols=True, leave=False) as pbar:
        with torch.no_grad():
            for bidx, (images, names) in enumerate(loader):
                images = images.to(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = perf_counter()
                logits = model(images)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_seconds += perf_counter() - t0
                infer_frames += logits.shape[0]
                for i in range(logits.shape[0]):
                    lanes = generate_tusimple_lines(logits[i], cfg.griding_num)
                    records.append(build_submit_record(names[i], lanes))
                pbar.update(1)
                fps = infer_frames / infer_seconds if infer_seconds > 0 else 0.0
                pbar.set_postfix(frames=infer_frames, fps=f"{fps:.2f}")
                if args.max_batches is not None and bidx + 1 >= args.max_batches:
                    break

    dump_submit_records(records, args.output)
    print(f"prediction dumped: {args.output}, records={len(records)}")

    metrics_map = {"Accuracy": float("nan"), "FP": float("nan"), "FN": float("nan")}
    if args.gt_file and Path(args.gt_file).exists():
        metrics = evaluate_submit_file(args.output, args.gt_file)
        metrics_map = _normalize_metrics(metrics)
    else:
        print("skip metrics: gt file not provided or not found")

    fps = infer_frames / infer_seconds if infer_seconds > 0 else 0.0
    flops_text = "N/A" if flops is None else f"{flops / 1e9:.2f} GFLOPs"
    print(
        "metrics: "
        f"Accuracy={metrics_map['Accuracy']:.6f} "
        f"FP={metrics_map['FP']:.6f} "
        f"FN={metrics_map['FN']:.6f}"
    )
    print(
        "infer: "
        f"frames={infer_frames} "
        f"time_s={infer_seconds:.4f} "
        f"FPS={fps:.2f}"
    )
    print(
        "model: "
        f"Params={params:,} "
        f"FLOPs={flops_text}"
    )


if __name__ == "__main__":
    main()
