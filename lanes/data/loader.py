from torch.utils.data import DataLoader

from lanes.data.transforms import build_image_transform
from lanes.data.tusimple_dataset import TusimpleTestDataset, TusimpleTrainDataset


def build_train_loader(cfg):
    transform = build_image_transform(cfg.input_height, cfg.input_width)
    dataset = TusimpleTrainDataset(
        data_root=cfg.data_root,
        img_transform=transform,
        griding_num=cfg.griding_num,
        num_rows=cfg.num_rows,
        num_lanes=cfg.num_lanes,
        dry_run=cfg.dry_run,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )


def build_test_loader(cfg):
    transform = build_image_transform(cfg.input_height, cfg.input_width)
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
