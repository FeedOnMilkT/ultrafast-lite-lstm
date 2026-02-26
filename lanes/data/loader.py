from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lanes.data.transforms import build_test_transform, build_train_transform
from lanes.data.tusimple_dataset import TusimpleTestDataset, TusimpleTrainDataset

from lanes.data.constant import tusimple_row_anchor


def build_train_loader(cfg, distributed: bool = False, rank: int = 0, world_size: int = 1):
    transform = build_train_transform(cfg.input_height, cfg.input_width)
    dataset = TusimpleTrainDataset(
        data_root=cfg.data_root,
        img_transform=transform,
        griding_num=cfg.griding_num,
        num_rows=cfg.num_rows,
        num_lanes=cfg.num_lanes,
        dry_run=cfg.dry_run,
        row_archor=tusimple_row_anchor
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


def build_test_loader(cfg):
    transform = build_test_transform(cfg.input_height, cfg.input_width)
    dataset = TusimpleTestDataset(
        data_root=cfg.data_root,
        img_transform=transform,
        dry_run=cfg.dry_run,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
