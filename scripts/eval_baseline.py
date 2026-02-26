from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tusimple_baseline import get_config
from lanes.data.loader import build_test_loader
from lanes.eval.postprocess import generate_tusimple_lines
from lanes.eval.tusimple_wrapper import build_submit_record, dump_submit_records, evaluate_submit_file
from lanes.models.parsing_baseline import ParsingBaseline
from lanes.utils.checkpoint import find_latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Eval M1 baseline (full-eval by default).")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/baseline/pred.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint root, e.g. outputs/baseline")
    parser.add_argument("--gt-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional debug cap. Default: full dataset")
    return parser.parse_args()


def _estimate_params_and_flops(model, cfg, device: torch.device):
    params = sum(p.numel() for p in model.parameters())
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        dummy = torch.randn(1, 3, cfg.input_height, cfg.input_width, device=device)
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


def main():
    args = parse_args()
    cfg = get_config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.dry_run:
        cfg.dry_run = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParsingBaseline(cfg.griding_num, cfg.num_rows, cfg.num_lanes).to(device)

    ckpt = args.checkpoint or find_latest_checkpoint(cfg.output_dir)
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    load_checkpoint(ckpt, model, optimizer=None, map_location=device)
    print(f"loaded checkpoint: {ckpt}")

    model.eval()
    params, flops = _estimate_params_and_flops(model, cfg, device)
    loader = build_test_loader(cfg)
    records = []
    infer_frames = 0
    infer_seconds = 0.0
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
