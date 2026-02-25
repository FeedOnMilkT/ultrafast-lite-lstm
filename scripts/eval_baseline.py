from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Eval M1 baseline (step-1 skeleton).")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/baseline/pred.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint root, e.g. outputs/baseline")
    parser.add_argument("--gt-file", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-batches", type=int, default=2)
    return parser.parse_args()


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
    if ckpt and Path(ckpt).exists():
        load_checkpoint(ckpt, model, optimizer=None, map_location=device)
        print(f"loaded checkpoint: {ckpt}")
    else:
        print("checkpoint not found under output dir, evaluating random-init model")

    model.eval()
    loader = build_test_loader(cfg)
    records = []
    with torch.no_grad():
        for bidx, (images, names) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            for i in range(logits.shape[0]):
                lanes = generate_tusimple_lines(logits[i], cfg.griding_num)
                records.append(build_submit_record(names[i], lanes))
            if bidx + 1 >= args.max_batches:
                break

    dump_submit_records(records, args.output)
    print(f"prediction dumped: {args.output}, records={len(records)}")

    if args.gt_file and Path(args.gt_file).exists():
        metrics = evaluate_submit_file(args.output, args.gt_file)
        print("eval:", metrics)
    else:
        print("skip metrics: gt file not provided")


if __name__ == "__main__":
    main()
