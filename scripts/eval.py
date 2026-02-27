from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lanes.data.loader import build_test_loader
from lanes.data.sequence_loader import build_test_sequence_loader
from lanes.eval.postprocess import generate_tusimple_lines
from lanes.eval.tusimple_wrapper import build_submit_record, dump_submit_records, evaluate_submit_file
from lanes.models.parsing_baseline import ParsingBaseline
from lanes.models.parsing_temporal_lstm import ParsingTemporalLSTM
from lanes.utils.checkpoint import find_latest_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Unified eval entry for baseline/lstm models.")
    parser.add_argument("--config", type=str, default="configs.tusimple_experiment", help="Config module path")
    parser.add_argument("--model-type", type=str, default=None, help="Override config: baseline|lstm")

    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint root, e.g. outputs")
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


def _load_cfg(config_module: str):
    module = importlib.import_module(config_module)
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{config_module} has no get_config()")
    return module.get_config()


def _normalize_model_type(model_type: str) -> str:
    value = model_type.lower().strip()
    if value not in ("baseline", "lstm"):
        raise ValueError(f"unsupported model_type: {model_type}")
    return value


def _build_model(cfg):
    if cfg.model_type == "baseline":
        return ParsingBaseline(
            cfg.griding_num,
            cfg.num_rows,
            cfg.num_lanes,
            hidden_features=cfg.cls_hidden_features,
            pretrained_backbone=cfg.pretrained_backbone,
        )
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


def _build_test_data_loader(cfg):
    if cfg.model_type == "baseline":
        return build_test_loader(cfg)
    return build_test_sequence_loader(cfg)


def _estimate_params_and_flops(model, cfg, device: torch.device):
    params = sum(p.numel() for p in model.parameters())
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode

        if cfg.model_type == "lstm":
            dummy = torch.randn(1, cfg.sequence_length, 3, cfg.input_height, cfg.input_width, device=device)
        else:
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


def _find_latest_typed_checkpoint(output_root: str, model_type: str):
    root = Path(output_root)
    if not root.exists():
        return None
    ckpts = sorted(root.rglob(f"{model_type}_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return str(ckpts[0])
    return find_latest_checkpoint(output_root)


def _append_result_log(log_path: Path, payload: dict):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main():
    args = parse_args()
    cfg = _load_cfg(args.config)

    cfg.model_type = _normalize_model_type(args.model_type if args.model_type else cfg.model_type)
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
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

    ckpt = args.checkpoint or _find_latest_typed_checkpoint(cfg.output_dir, cfg.model_type)
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    load_checkpoint(ckpt, model, optimizer=None, map_location=device)
    print(f"loaded checkpoint: {ckpt}")

    model.eval()
    params, flops = _estimate_params_and_flops(model, cfg, device)
    loader = _build_test_data_loader(cfg)
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

    if args.output is None:
        output_path = Path(cfg.output_dir) / f"pred_{cfg.model_type}.jsonl"
    else:
        output_path = Path(args.output)
    dump_submit_records(records, str(output_path))
    print(f"prediction dumped: {output_path}, records={len(records)}")

    metrics_map = {"Accuracy": float("nan"), "FP": float("nan"), "FN": float("nan")}
    if args.gt_file and Path(args.gt_file).exists():
        metrics = evaluate_submit_file(str(output_path), args.gt_file)
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
        f"model={cfg.model_type} "
        f"frames={infer_frames} "
        f"time_s={infer_seconds:.4f} "
        f"FPS={fps:.2f}"
    )
    print(
        "model: "
        f"Params={params:,} "
        f"FLOPs={flops_text}"
    )

    result_log_path = Path(cfg.output_dir) / f"eval_{cfg.model_type}_results.jsonl"
    result_payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "model_type": cfg.model_type,
        "config": args.config,
        "checkpoint": str(ckpt),
        "data_root": str(cfg.data_root),
        "gt_file": str(args.gt_file) if args.gt_file else None,
        "prediction_file": str(output_path),
        "batch_size": int(cfg.batch_size),
        "num_workers": int(cfg.num_workers),
        "sequence_length": int(cfg.sequence_length),
        "sequence_stride": int(cfg.sequence_stride),
        "metrics": {
            "Accuracy": float(metrics_map["Accuracy"]),
            "FP": float(metrics_map["FP"]),
            "FN": float(metrics_map["FN"]),
        },
        "infer": {
            "frames": int(infer_frames),
            "time_s": float(infer_seconds),
            "fps": float(fps),
        },
        "model": {
            "params": int(params),
            "flops": None if flops is None else int(flops),
        },
    }
    _append_result_log(result_log_path, result_payload)
    print(f"result log appended: {result_log_path}")


if __name__ == "__main__":
    main()
