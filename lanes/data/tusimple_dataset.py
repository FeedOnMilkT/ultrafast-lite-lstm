from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np

def loader_func(path: Path, mode: str | None = None):
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    with Image.open(path) as img:
        if mode is not None:
            img = img.convert(mode)
        return img.copy()

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
        self.row_archor = sorted(row_archor) if row_archor is not None else []

        self.dry_run = dry_run
        self.dry_run_len = dry_run_len
        self.samples: List[Tuple[str, str]] = []

        if self.dry_run:
            return

        if self.data_root is None:
            raise ValueError("data_root is required when dry_run=False")
        if not self.row_archor:
            raise ValueError("row_archor is required when dry_run=False")
        
        list_file = self.data_root / "train_gt.txt"

        if not list_file.exists():
            raise FileNotFoundError(f"missing train list: {list_file}")
        
        with list_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 2:
                    raise ValueError(f"invalid line in {list_file}:{line_idx} -> {line!r}")
                self.samples.append((fields[0], fields[1]))

        if not self.samples:
            raise ValueError(f"empty train list: {list_file}")

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

        image_path, label_path = self.samples[idx]
        if not image_path or not label_path:
            raise ValueError(f"invalid sample at idx={idx}: image={image_path!r}, label={label_path!r}")
        if image_path[0] == "/":
            image_path = image_path[1:]
            label_path = label_path[1:]

        label_path = self.data_root / label_path
        label = loader_func(label_path, mode="L")

        img_path = self.data_root / image_path
        img = loader_func(img_path, mode="RGB")

        lane_pts = self._get_index(label)

        w, _ = img.size

        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        cls_label = torch.from_numpy(cls_label).long()

        image = self.img_transform(img)
        
        return image, cls_label
    
    def _get_index(self, label):
        _, h = label.size
        label_np = np.asarray(label)

        if h != 288:
            scale_f = lambda x: int((x * 1.0/288) * h)
            sample_tmp = list(map(scale_f, self.row_archor))
        else:
            sample_tmp = list(self.row_archor)
        
        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))

        for i, r in enumerate(sample_tmp):
            r_idx = int(round(r))
            if r_idx < 0 or r_idx >= h:
                raise ValueError(f"row anchor out of range: {r_idx}, label height={h}")
            label_r = label_np[r_idx]
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
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split()
                if not parts:
                    continue
                name = parts[0]
                if name:
                    self.names.append(name)
                else:
                    raise ValueError(f"invalid line in {list_file}:{line_idx}")

        if not self.names:
            raise ValueError(f"empty test list: {list_file}")

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_len
        return len(self.names)

    def __getitem__(self, idx: int):
        if self.dry_run:
            image = torch.randn(3, 288, 800)
            name = f"clips/dry_run/{idx:06d}.jpg"
            return image, name

        # rel_path = self.names[idx]
        # rel_path = rel_path[1:] if rel_path.startswith("/") else rel_path
        # image_path = self.data_root / rel_path
        # image = Image.open(image_path).convert("RGB")
        # image = self.img_transform(image)

        name = self.names[idx]
        imag_path = self.data_root / name
        img = loader_func(imag_path, mode="RGB")

        if self.img_transform is not None:
            image = self.img_transform(img)
        else:
            image = img

        return image, name
