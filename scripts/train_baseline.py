from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.tusimple_baseline import get_config
from lanes.data.loader import build_test_loader, build_train_loader
from lanes.eval.postprocess import generate_tusimple_lines
from lanes.eval.tusimple_wrapper import build_submit_record, dump_submit_records, evaluate_submit_file
from lanes.losses.registry import build_losses, compute_total_loss
from lanes.models.parsing_baseline import ParsingBaseline
from lanes.utils.checkpoint import save_checkpoint
from lanes.utils.logger import build_logger
from lanes.utils.metrics import top1_accuracy
from lanes.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train M1 baseline (single GPU or torchrun DDP).")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None, help="Run root, e.g. outputs/baseline")
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=10,
        help="Save periodic checkpoints every N epochs on rank0 (<=0 disables periodic saves).",
    )
    parser.add_argument("--auto-eval", action="store_true", help="Run evaluation after training on rank0")
    parser.add_argument("--eval-gt-file", type=str, default=None, help="GT jsonl file for Accuracy/FP/FN")
    parser.add_argument("--eval-output", type=str, default=None, help="Prediction output path")
    parser.add_argument("--eval-max-batches", type=int, default=None, help="Optional debug cap for auto-eval")
    return parser.parse_args()


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _setup_distributed():
    if not _is_distributed():
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, local_rank, world_size


def _cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def _reduce_mean(value: float, device: torch.device, is_distributed: bool, world_size: int) -> float:
    if not is_distributed:
        return value
    tensor = torch.tensor(value, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / world_size).item())


def _build_optimizer(cfg, model):
    opt_name = str(cfg.optimizer).lower()
    if opt_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {cfg.optimizer}")


def _compute_lr_factor(cfg, cur_iter: int, total_iters: int, milestones: list[int]) -> float:
    scheduler_name = str(cfg.scheduler).lower()
    if scheduler_name == "cos":
        factor = 0.5 * (1.0 + math.cos(math.pi * cur_iter / max(total_iters, 1)))
    elif scheduler_name == "multi":
        passed = sum(cur_iter >= m for m in milestones)
        factor = cfg.gamma**passed
    else:
        raise ValueError(f"unsupported scheduler: {cfg.scheduler}")

    warmup_name = str(cfg.warmup).lower() if cfg.warmup is not None else "none"
    if warmup_name == "linear":
        if cfg.warmup_iters > 0 and cur_iter < cfg.warmup_iters:
            factor *= float(cur_iter + 1) / float(cfg.warmup_iters)
    elif warmup_name not in ("none", ""):
        raise ValueError(f"unsupported warmup strategy: {cfg.warmup}")
    return factor


def _set_learning_rate(optimizer, base_lr: float, factor: float):
    for param_group in optimizer.param_groups:
        lr_mult = float(param_group.get("lr_mult", 1.0))
        param_group["lr"] = base_lr * lr_mult * factor


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _normalize_metrics(metrics):
    out = {"Accuracy": float("nan"), "FP": float("nan"), "FN": float("nan")}
    for item in metrics:
        name = item.get("name")
        if name in out:
            out[name] = float(item.get("value", float("nan")))
    return out


def _run_auto_eval(model, cfg, device, output_path: Path, gt_file: str | None, max_batches: int | None, logger):
    model.eval()
    loader = build_test_loader(cfg)
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    records = []
    infer_frames = 0
    infer_seconds = 0.0
    with tqdm(total=total_batches, desc="Auto-eval", dynamic_ncols=True, leave=False) as pbar:
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
                if max_batches is not None and bidx + 1 >= max_batches:
                    break

    dump_submit_records(records, str(output_path))
    metrics_map = {"Accuracy": float("nan"), "FP": float("nan"), "FN": float("nan")}
    if gt_file is not None and Path(gt_file).exists():
        metrics = evaluate_submit_file(str(output_path), gt_file)
        metrics_map = _normalize_metrics(metrics)
    elif logger is not None:
        logger.info("auto_eval: skip metrics (gt file missing): %s", gt_file)

    fps = infer_frames / infer_seconds if infer_seconds > 0 else 0.0
    if logger is not None:
        logger.info(
            "auto_eval: Accuracy=%.6f FP=%.6f FN=%.6f",
            metrics_map["Accuracy"],
            metrics_map["FP"],
            metrics_map["FN"],
        )
        logger.info(
            "auto_eval: frames=%d time_s=%.4f FPS=%.2f pred_file=%s",
            infer_frames,
            infer_seconds,
            fps,
            output_path,
        )


def main():
    args = parse_args()
    is_distributed, rank, local_rank, world_size = _setup_distributed()
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

    model_to_save = None
    ckpt_path = None
    run_dir = None
    device = None
    logger = None
    try:
        seed = cfg.seed + rank if is_distributed else cfg.seed
        set_seed(seed)
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.output_dir) / run_ts
        ckpt_path = run_dir / f"baseline_{run_ts}.pth"

        logger = build_logger(str(run_dir), name=f"train_rank{rank}") if rank == 0 else None
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}") if is_distributed else torch.device("cuda")
        else:
            device = torch.device("cpu")

        if logger is not None:
            logger.info(
                "device=%s dry_run=%s run_dir=%s distributed=%s rank=%d world_size=%d",
                device,
                cfg.dry_run,
                run_dir,
                is_distributed,
                rank,
                world_size,
            )

        train_loader, train_sampler = build_train_loader(
            cfg,
            distributed=is_distributed,
            rank=rank,
            world_size=world_size,
        )
        steps_per_epoch = len(train_loader)
        if cfg.max_steps_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be > 0")
        total_iters = cfg.epochs * steps_per_epoch
        milestones = [int(step_epoch * steps_per_epoch) for step_epoch in cfg.steps]

        model = ParsingBaseline(cfg.griding_num, cfg.num_rows, cfg.num_lanes).to(device)
        if is_distributed:
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank} if device.type == "cuda" else {}
            model = DDP(model, **ddp_kwargs)
        optimizer = _build_optimizer(cfg, model)
        losses = build_losses()
        if logger is not None:
            model_for_stats = model.module if isinstance(model, DDP) else model
            params = sum(p.numel() for p in model_for_stats.parameters())
            logger.info("model: Params=%d", params)
            logger.info(
                "train_strategy: optimizer=%s lr=%.6g wd=%.6g momentum=%.3f scheduler=%s warmup=%s warmup_iters=%d",
                cfg.optimizer,
                cfg.learning_rate,
                cfg.weight_decay,
                cfg.momentum,
                cfg.scheduler,
                cfg.warmup,
                cfg.warmup_iters,
            )

        model.train()
        for epoch in range(cfg.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            pbar = None
            if rank == 0:
                pbar = tqdm(
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{cfg.epochs}",
                    dynamic_ncols=True,
                    leave=False,
                )

            for step, (images, cls_label) in enumerate(train_loader):
                global_step = epoch * steps_per_epoch + step
                lr_factor = _compute_lr_factor(cfg, global_step, total_iters, milestones)
                _set_learning_rate(optimizer, cfg.learning_rate, lr_factor)

                images = images.to(device, non_blocking=True)
                cls_label = cls_label.to(device, non_blocking=True)
                logits = model(images)
                total_loss, loss_items = compute_total_loss(losses, logits, cls_label)
                acc = top1_accuracy(logits, cls_label)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                avg_total = _reduce_mean(loss_items["total_loss"], device, is_distributed, world_size)
                avg_cls = _reduce_mean(loss_items["cls_loss"], device, is_distributed, world_size)
                avg_rel = _reduce_mean(loss_items["relation_loss"], device, is_distributed, world_size)
                avg_acc = _reduce_mean(acc, device, is_distributed, world_size)

                if logger is not None:
                    logger.info(
                        "train: epoch=%d step=%d lr=%.6g total=%.4f cls=%.4f rel=%.4f "
                        "Accuracy=%.4f FP=nan FN=nan",
                        epoch,
                        step,
                        _current_lr(optimizer),
                        avg_total,
                        avg_cls,
                        avg_rel,
                        avg_acc,
                    )
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        lr=f"{_current_lr(optimizer):.2e}",
                        loss=f"{avg_total:.4f}",
                        acc=f"{avg_acc:.4f}",
                    )
                if cfg.max_steps_per_epoch is not None and step + 1 >= cfg.max_steps_per_epoch:
                    break

            if pbar is not None:
                pbar.close()

            if rank == 0:
                model_to_save = model.module if isinstance(model, DDP) else model
                if args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0:
                    periodic_ckpt_path = run_dir / f"epoch_{epoch + 1:03d}.pth"
                    save_checkpoint(str(periodic_ckpt_path), model_to_save, optimizer, epoch=epoch)
                    if logger is not None:
                        logger.info("periodic checkpoint saved: %s", periodic_ckpt_path)

        if rank == 0:
            model_to_save = model.module if isinstance(model, DDP) else model
            save_checkpoint(str(ckpt_path), model_to_save, optimizer, epoch=cfg.epochs - 1)
            if logger is not None:
                logger.info("checkpoint saved: %s", ckpt_path)
    finally:
        _cleanup_distributed(is_distributed)

    if rank == 0 and args.auto_eval:
        if model_to_save is None or run_dir is None or ckpt_path is None or device is None:
            raise RuntimeError("auto-eval unavailable: training did not produce a model on rank0")
        eval_output = Path(args.eval_output) if args.eval_output else run_dir / "pred_eval.jsonl"
        _run_auto_eval(
            model=model_to_save,
            cfg=cfg,
            device=device,
            output_path=eval_output,
            gt_file=args.eval_gt_file,
            max_batches=args.eval_max_batches,
            logger=logger,
        )


if __name__ == "__main__":
    main()
