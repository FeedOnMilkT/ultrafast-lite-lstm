from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lanes.data.constant import tusimple_row_anchor
from lanes.data.transforms import build_test_transform, build_train_transform
from lanes.data.tusimple_sequence_dataset import TusimpleTestSequenceDataset, TusimpleTrainSequenceDataset


def _resolve_train_list(cfg) -> str:
    return cfg.sequence_train_list if cfg.sequence_train_list else cfg.train_list


def _resolve_test_list(cfg) -> str:
    return cfg.sequence_test_list if cfg.sequence_test_list else cfg.test_list


def build_train_sequence_loader(cfg, distributed: bool = False, rank: int = 0, world_size: int = 1):
    transform = build_train_transform(cfg.input_height, cfg.input_width)
    dataset = TusimpleTrainSequenceDataset(
        data_root=cfg.data_root,
        img_transform=transform,
        griding_num=cfg.griding_num,
        num_rows=cfg.num_rows,
        num_lanes=cfg.num_lanes,
        row_archor=tusimple_row_anchor,
        list_file=_resolve_train_list(cfg),
        sequence_length=cfg.sequence_length,
        sequence_stride=cfg.sequence_stride,
        dry_run=cfg.dry_run,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        drop_last=True,
    ), sampler


def build_test_sequence_loader(cfg):
    transform = build_test_transform(cfg.input_height, cfg.input_width)
    dataset = TusimpleTestSequenceDataset(
        data_root=cfg.data_root,
        img_transform=transform,
        list_file=_resolve_test_list(cfg),
        sequence_length=cfg.sequence_length,
        sequence_stride=cfg.sequence_stride,
        dry_run=cfg.dry_run,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
