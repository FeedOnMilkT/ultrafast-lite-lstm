from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class TusimpleTrainDataset(Dataset):
    """
    Step-1 dataset skeleton.
    - dry_run=True: synthetic samples for end-to-end smoke tests.
    - dry_run=False: file list plumbing is ready; target parsing is TODO.
    """

    def __init__(
        self,
        data_root: str | None,
        img_transform,
        griding_num: int,
        num_rows: int,
        num_lanes: int,
        dry_run: bool = True,
        dry_run_len: int = 32,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.img_transform = img_transform
        self.griding_num = griding_num
        self.num_rows = num_rows
        self.num_lanes = num_lanes
        self.dry_run = dry_run
        self.dry_run_len = dry_run_len
        self.samples: List[Tuple[str, str]] = []

        if not self.dry_run:
            if self.data_root is None:
                raise ValueError("data_root is required when dry_run=False")
            list_file = self.data_root / "train_gt.txt"
            if not list_file.exists():
                raise FileNotFoundError(f"missing train list: {list_file}")
            with list_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fields = line.split()
                    if len(fields) < 2:
                        continue
                    self.samples.append((fields[0], fields[1]))

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_len
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.dry_run:
            image = torch.randn(3, 288, 800)
            cls_label = torch.randint(
                low=0,
                high=self.griding_num + 1,
                size=(self.num_rows, self.num_lanes),
                dtype=torch.long,
            )
            return image, cls_label

        rel_img_path, _ = self.samples[idx]
        rel_img_path = rel_img_path[1:] if rel_img_path.startswith("/") else rel_img_path
        image_path = self.data_root / rel_img_path
        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)

        # TODO(step-2): parse lane mask / lane json into cls label.
        cls_label = torch.randint(
            low=0,
            high=self.griding_num + 1,
            size=(self.num_rows, self.num_lanes),
            dtype=torch.long,
        )
        return image, cls_label


class TusimpleTestDataset(Dataset):
    def __init__(
        self,
        data_root: str | None,
        img_transform,
        dry_run: bool = True,
        dry_run_len: int = 8,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.img_transform = img_transform
        self.dry_run = dry_run
        self.dry_run_len = dry_run_len
        self.names: List[str] = []

        if not self.dry_run:
            if self.data_root is None:
                raise ValueError("data_root is required when dry_run=False")
            list_file = self.data_root / "test.txt"
            if not list_file.exists():
                raise FileNotFoundError(f"missing test list: {list_file}")
            with list_file.open("r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip().split()[0]
                    if name:
                        self.names.append(name)

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_len
        return len(self.names)

    def __getitem__(self, idx: int):
        if self.dry_run:
            image = torch.randn(3, 288, 800)
            name = f"clips/dry_run/{idx:06d}.jpg"
            return image, name

        rel_path = self.names[idx]
        rel_path = rel_path[1:] if rel_path.startswith("/") else rel_path
        image_path = self.data_root / rel_path
        image = Image.open(image_path).convert("RGB")
        image = self.img_transform(image)
        return image, rel_path
