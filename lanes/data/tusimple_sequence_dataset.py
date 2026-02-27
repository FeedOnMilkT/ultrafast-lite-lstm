from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lanes.data.tusimple_dataset import loader_func


def _normalize_rel_path(path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("empty relative path")
    return path[1:] if path.startswith("/") else path


def _frame_sort_key(image_rel: str) -> Tuple[int, int, str]:
    stem = Path(image_rel).stem
    if stem.isdigit():
        return (0, int(stem), stem)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return (1, int(digits), stem)
    return (2, 0, stem)


def _make_sequence_windows(clip_indices: List[int], seq_len: int, stride: int) -> List[List[int]]:
    windows: List[List[int]] = []
    for end_pos in range(len(clip_indices)):
        win: List[int] = []
        for step in range(seq_len):
            src_pos = end_pos - (seq_len - 1 - step) * stride
            if src_pos < 0:
                src_pos = 0
            win.append(clip_indices[src_pos])
        windows.append(win)
    return windows


@dataclass(frozen=True)
class _FrameTrainSample:
    image_rel: str
    label_rel: str


@dataclass(frozen=True)
class _FrameTestSample:
    image_rel: str


class TusimpleTrainSequenceDataset(Dataset):
    """
    Sequence dataset for temporal training.
    Returns:
      images: [T, 3, H, W]
      cls_label: [num_rows, num_lanes] (only the last frame in the sequence)
    """

    def __init__(
        self,
        data_root: str | None,
        img_transform,
        griding_num: int,
        num_rows: int,
        num_lanes: int,
        row_archor,
        list_file: str = "train_gt.txt",
        sequence_length: int = 3,
        sequence_stride: int = 1,
        dry_run: bool = True,
        dry_run_len: int = 32,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.img_transform = img_transform
        self.griding_num = griding_num
        self.num_rows = num_rows
        self.num_lanes = num_lanes
        self.row_archor = sorted(row_archor) if row_archor is not None else []
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.dry_run = dry_run
        self.dry_run_len = dry_run_len

        self.frames: List[_FrameTrainSample] = []
        self.sequence_indices: List[List[int]] = []

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be > 0")
        if self.dry_run:
            return
        if self.data_root is None:
            raise ValueError("data_root is required when dry_run=False")
        if not self.row_archor:
            raise ValueError("row_archor is required when dry_run=False")

        list_path = Path(list_file)
        if not list_path.is_absolute():
            list_path = self.data_root / list_path
        if not list_path.exists():
            raise FileNotFoundError(f"missing train list: {list_path}")

        clip_frames: Dict[str, List[Tuple[Tuple[int, int, str], _FrameTrainSample]]] = defaultdict(list)
        with list_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                image_rel = _normalize_rel_path(fields[0])
                label_rel = _normalize_rel_path(fields[1])
                clip_key = str(Path(image_rel).parent)
                clip_frames[clip_key].append(
                    (_frame_sort_key(image_rel), _FrameTrainSample(image_rel=image_rel, label_rel=label_rel))
                )

        if not clip_frames:
            raise ValueError(f"empty train list: {list_path}")

        for clip_key in sorted(clip_frames.keys()):
            entries = sorted(clip_frames[clip_key], key=lambda x: x[0])
            clip_indices: List[int] = []
            for _, sample in entries:
                clip_indices.append(len(self.frames))
                self.frames.append(sample)
            self.sequence_indices.extend(
                _make_sequence_windows(clip_indices, seq_len=self.sequence_length, stride=self.sequence_stride)
            )

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_len
        return len(self.sequence_indices)

    def __getitem__(self, idx: int):
        if self.dry_run:
            image_seq = torch.randn(self.sequence_length, 3, 288, 800)
            cls_label = torch.randint(
                low=0,
                high=self.griding_num + 1,
                size=(self.num_rows, self.num_lanes),
                dtype=torch.long,
            )
            return image_seq, cls_label

        seq_frame_indices = self.sequence_indices[idx]
        images: List[torch.Tensor] = []
        last_width = None
        for frame_idx in seq_frame_indices:
            frame = self.frames[frame_idx]
            img_path = self.data_root / frame.image_rel
            img = loader_func(img_path, mode="RGB")
            if frame_idx == seq_frame_indices[-1]:
                last_width = img.size[0]
            image = self.img_transform(img) if self.img_transform is not None else img
            images.append(image)
        if last_width is None:
            raise RuntimeError("failed to read last frame width")

        last_frame = self.frames[seq_frame_indices[-1]]
        label_path = self.data_root / last_frame.label_rel
        label = loader_func(label_path, mode="L")
        lane_pts = self._get_index(label)
        cls_label = self._grid_pts(lane_pts, self.griding_num, last_width)
        cls_label = torch.from_numpy(cls_label).long()

        return torch.stack(images, dim=0), cls_label

    def _get_index(self, label):
        _, h = label.size
        label_np = np.asarray(label)
        if h != 288:
            scale_f = lambda x: int((x * 1.0 / 288) * h)
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
                all_idx[lane_index - 1, i, 0] = r
                all_idx[lane_index - 1, i, 1] = np.mean(pos)
        return all_idx

    def _grid_pts(self, pts, num_cols, w):
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)
        if n2 != 2:
            raise ValueError(f"unexpected pts shape: {pts.shape}")
        to_pts = np.zeros((n, num_lane))
        step = col_sample[1] - col_sample[0]
        for lane_id in range(num_lane):
            lane_pts = pts[lane_id, :, 1]
            to_pts[:, lane_id] = np.asarray(
                [int(pt // step) if pt != -1 else num_cols for pt in lane_pts]
            )
        return to_pts.astype(int)


class TusimpleTestSequenceDataset(Dataset):
    """
    Sequence dataset for temporal inference/eval.
    Returns:
      images: [T, 3, H, W]
      last_frame_name: relative image path for submit record
    """

    def __init__(
        self,
        data_root: str | None,
        img_transform,
        list_file: str = "test.txt",
        sequence_length: int = 3,
        sequence_stride: int = 1,
        dry_run: bool = True,
        dry_run_len: int = 8,
    ) -> None:
        self.data_root = Path(data_root) if data_root else None
        self.img_transform = img_transform
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.dry_run = dry_run
        self.dry_run_len = dry_run_len

        self.frames: List[_FrameTestSample] = []
        self.sequence_indices: List[List[int]] = []

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.sequence_stride <= 0:
            raise ValueError("sequence_stride must be > 0")
        if self.dry_run:
            return
        if self.data_root is None:
            raise ValueError("data_root is required when dry_run=False")

        list_path = Path(list_file)
        if not list_path.is_absolute():
            list_path = self.data_root / list_path
        if not list_path.exists():
            raise FileNotFoundError(f"missing test list: {list_path}")

        clip_frames: Dict[str, List[Tuple[Tuple[int, int, str], _FrameTestSample]]] = defaultdict(list)
        with list_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                image_rel = _normalize_rel_path(fields[0])
                clip_key = str(Path(image_rel).parent)
                clip_frames[clip_key].append((_frame_sort_key(image_rel), _FrameTestSample(image_rel=image_rel)))

        if not clip_frames:
            raise ValueError(f"empty test list: {list_path}")

        for clip_key in sorted(clip_frames.keys()):
            entries = sorted(clip_frames[clip_key], key=lambda x: x[0])
            clip_indices: List[int] = []
            for _, sample in entries:
                clip_indices.append(len(self.frames))
                self.frames.append(sample)
            self.sequence_indices.extend(
                _make_sequence_windows(clip_indices, seq_len=self.sequence_length, stride=self.sequence_stride)
            )

    def __len__(self) -> int:
        if self.dry_run:
            return self.dry_run_len
        return len(self.sequence_indices)

    def __getitem__(self, idx: int):
        if self.dry_run:
            image_seq = torch.randn(self.sequence_length, 3, 288, 800)
            name = f"clips/dry_run/{idx:06d}.jpg"
            return image_seq, name

        seq_frame_indices = self.sequence_indices[idx]
        images: List[torch.Tensor] = []
        for frame_idx in seq_frame_indices:
            frame = self.frames[frame_idx]
            img_path = self.data_root / frame.image_rel
            img = loader_func(img_path, mode="RGB")
            image = self.img_transform(img) if self.img_transform is not None else img
            images.append(image)

        last_name = self.frames[seq_frame_indices[-1]].image_rel
        return torch.stack(images, dim=0), last_name
