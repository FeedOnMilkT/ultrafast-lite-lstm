from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tusimple_baseline import get_config
from lanes.data.loader import build_train_loader
from lanes.losses.registry import build_losses, compute_total_loss
from lanes.models.parsing_baseline import ParsingBaseline
from lanes.utils.checkpoint import save_checkpoint
from lanes.utils.logger import build_logger
from lanes.utils.metrics import top1_accuracy
from lanes.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train M1 baseline (step-1 skeleton).")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Run root, e.g. outputs/baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.max_steps_per_epoch = args.max_steps
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.dry_run:
        cfg.dry_run = True

    set_seed(cfg.seed)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_dir) / run_ts
    ckpt_path = run_dir / f"baseline_{run_ts}.pth"

    logger = build_logger(str(run_dir), name="train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("device=%s dry_run=%s run_dir=%s", device, cfg.dry_run, run_dir)
    train_loader = build_train_loader(cfg)
    model = ParsingBaseline(cfg.griding_num, cfg.num_rows, cfg.num_lanes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    losses = build_losses()

    model.train()
    for epoch in range(cfg.epochs):
        for step, (images, cls_label) in enumerate(train_loader):
            images = images.to(device)
            cls_label = cls_label.to(device)
            logits = model(images)
            total_loss, loss_items = compute_total_loss(losses, logits, cls_label)
            acc = top1_accuracy(logits, cls_label)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            logger.info(
                "epoch=%d step=%d total=%.4f cls=%.4f rel=%.4f acc=%.4f",
                epoch,
                step,
                loss_items["total_loss"],
                loss_items["cls_loss"],
                loss_items["relation_loss"],
                acc,
            )
            if step + 1 >= cfg.max_steps_per_epoch:
                break

    save_checkpoint(str(ckpt_path), model, optimizer, epoch=cfg.epochs - 1)
    logger.info("checkpoint saved: %s", ckpt_path)


if __name__ == "__main__":
    main()
