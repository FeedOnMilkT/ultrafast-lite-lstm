from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np 
import os

def loader_func(path):
    return Image.open(path)

class TusimpleTrainDataset(Dataset):
    """
    Step-1 dataset skeleton.
    - dry_run=True: synthetic samples for end-to-end smoke tests.
    - dry_run=False: file list plumbing is ready; target parsing is TODO.
    """

    def __init__(
        self,
        data_root: str | None, #path
        img_transform,
        griding_num: int,
        num_rows: int,
        num_lanes: int,
        row_archor: None,
        dry_run: bool = True,
        dry_run_len: int = 32,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.img_transform = img_transform
        self.griding_num = griding_num
        self.num_rows = num_rows
        self.num_lanes = num_lanes
        # self.num_rows.sort()
        self.row_archor = row_archor
        self.row_archor.sort()

        self.dry_run = dry_run
        self.dry_run_len = dry_run_len
        # self.samples: List[Tuple[str, str]] = []

        if self.dry_run:
            return

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
        
        
        # with list_file.open("r", encoding="utf-8") as f:
        #   self.list = f.readlines()

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

        # rel_img_path, _ = self.samples[idx]
        # rel_img_path = rel_img_path[1:] if rel_img_path.startswith("/") else rel_img_path
        # image_path = self.data_root / rel_img_path
        # image = Image.open(image_path).convert("RGB")

        image_path, label_path  = self.samples[idx]

        if image_path[0] == "/":
            image_path = image_path[1:]
            label_path = label_path[1:]

        label_path = os.path.join(self.data_root, label_path)
        label = loader_func(label_path)

        img_path = os.path.join(self.data_root, img_path)
        img = loader_func(img_path)

        lane_pts = self._get_index(label)

        w, h = img.size

        cls_label = self._grid_pts(lane_pts, self.griding_num, w)

        image = self.img_transform(image)
        
        return image, cls_label
    
    def _get_index(self, label):
        w, h = label.size

        if h != 288:
            scale_f = lambda x: int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f, self.row_archor))
        else:
            sample_tmp = list(map(288, self.row_archor))
        
        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))

        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_index in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_index)[0]
                if len(pos) == 0:
                    all_idx[lane_index - 1, i, 0] = r
                    all_idx[lane_index - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_index - 1, i, 0] = r
                all_idx[lane_index - 1, i, 1] = pos

        return all_idx
    
    def _grid_pts(self, pts, num_cols, w):
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2

        to_pts = np.zeros((n, num_lane))

        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)        



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

        if self.dry_run:
            return

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
