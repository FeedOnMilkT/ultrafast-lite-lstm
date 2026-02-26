from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tqdm


def calc_k(line: np.ndarray) -> float:
    """
    Estimate lane direction from interleaved xy points.
    Returns -10 for short lanes to keep original UFLD behavior.
    """
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 90:
        return -10.0
    p = np.polyfit(line_x, line_y, deg=1)
    return float(np.arctan(p[0]))


def draw_lane(mask: np.ndarray, line: np.ndarray, lane_id: int, thickness: int = 16):
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]), int(line_y[0]))
    for i in range(len(line_x) - 1):
        pt1 = (int(line_x[i + 1]), int(line_y[i + 1]))
        cv2.line(mask, pt0, pt1, (lane_id,), thickness=thickness)
        pt0 = pt1


def load_json_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def records_to_lines(records):
    names, all_line_txt = [], []
    for rec in records:
        raw_file = rec["raw_file"]
        h_samples = np.asarray(rec["h_samples"])
        lanes = np.asarray(rec["lanes"])
        names.append(raw_file)

        line_txt = []
        for lane in lanes:
            if np.all(lane == -2):
                continue
            valid = lane != -2
            xs = lane[valid]
            ys = h_samples[valid]
            if len(xs) < 2:
                continue
            interleaved = [None] * (len(xs) + len(ys))
            interleaved[::2] = list(map(str, xs))
            interleaved[1::2] = list(map(str, ys))
            line_txt.append(interleaved)
        all_line_txt.append(line_txt)
    return names, all_line_txt


def _pick_idx_by_value(values: np.ndarray, target: float) -> int:
    indices = np.where(values == target)[0]
    if len(indices) == 0:
        raise RuntimeError("failed to map slope value to lane index")
    return int(indices[0])


def generate_segmentation_and_train_list(
    root: Path,
    names,
    all_line_txt,
    train_gt_name: str = "train_gt.txt",
    line_thickness: int = 16,
):
    out_path = root / train_gt_name
    with out_path.open("w", encoding="utf-8") as train_gt_fp:
        for i in tqdm.tqdm(range(len(all_line_txt)), desc="convert train"):
            tmp_line = all_line_txt[i]
            lines = [list(map(float, one_line)) for one_line in tmp_line]
            ks = np.array([calc_k(np.asarray(line)) for line in lines], dtype=np.float32) if lines else np.array([])

            k_neg = ks[ks < 0].copy()
            k_pos = ks[ks > 0].copy()
            k_neg = k_neg[k_neg != -10]
            k_pos = k_pos[k_pos != -10]
            k_neg.sort()
            k_pos.sort()

            label_rel = names[i][:-3] + "png"
            label_abs = root / label_rel
            label_abs.parent.mkdir(parents=True, exist_ok=True)

            mask = np.zeros((720, 1280), dtype=np.uint8)
            bin_label = [0, 0, 0, 0]

            if len(k_neg) == 1:
                which_lane = _pick_idx_by_value(ks, k_neg[0])
                draw_lane(mask, np.asarray(lines[which_lane]), 2, thickness=line_thickness)
                bin_label[1] = 1
            elif len(k_neg) == 2:
                which_lane = _pick_idx_by_value(ks, k_neg[1])
                draw_lane(mask, np.asarray(lines[which_lane]), 1, thickness=line_thickness)
                which_lane = _pick_idx_by_value(ks, k_neg[0])
                draw_lane(mask, np.asarray(lines[which_lane]), 2, thickness=line_thickness)
                bin_label[0] = 1
                bin_label[1] = 1
            elif len(k_neg) > 2:
                which_lane = _pick_idx_by_value(ks, k_neg[1])
                draw_lane(mask, np.asarray(lines[which_lane]), 1, thickness=line_thickness)
                which_lane = _pick_idx_by_value(ks, k_neg[0])
                draw_lane(mask, np.asarray(lines[which_lane]), 2, thickness=line_thickness)
                bin_label[0] = 1
                bin_label[1] = 1

            if len(k_pos) == 1:
                which_lane = _pick_idx_by_value(ks, k_pos[0])
                draw_lane(mask, np.asarray(lines[which_lane]), 3, thickness=line_thickness)
                bin_label[2] = 1
            elif len(k_pos) == 2:
                which_lane = _pick_idx_by_value(ks, k_pos[1])
                draw_lane(mask, np.asarray(lines[which_lane]), 3, thickness=line_thickness)
                which_lane = _pick_idx_by_value(ks, k_pos[0])
                draw_lane(mask, np.asarray(lines[which_lane]), 4, thickness=line_thickness)
                bin_label[2] = 1
                bin_label[3] = 1
            elif len(k_pos) > 2:
                which_lane = _pick_idx_by_value(ks, k_pos[-1])
                draw_lane(mask, np.asarray(lines[which_lane]), 3, thickness=line_thickness)
                which_lane = _pick_idx_by_value(ks, k_pos[-2])
                draw_lane(mask, np.asarray(lines[which_lane]), 4, thickness=line_thickness)
                bin_label[2] = 1
                bin_label[3] = 1

            cv2.imwrite(str(label_abs), mask)
            train_gt_fp.write(f"{names[i]} {label_rel} {' '.join(map(str, bin_label))}\n")


def generate_test_list(root: Path, names, test_list_name: str = "test.txt"):
    with (root / test_list_name).open("w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def resolve_label_files(root: Path, names):
    files = []
    for name in names:
        p = root / name
        if not p.exists():
            raise FileNotFoundError(f"missing label file: {p}")
        files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Convert TuSimple JSON labels for UFLD-style training.")
    parser.add_argument("--root", required=True, help="TuSimple dataset root")
    parser.add_argument(
        "--train-label-files",
        nargs="+",
        default=["label_data_0601.json", "label_data_0531.json", "label_data_0313.json"],
        help="Training json files under root",
    )
    parser.add_argument(
        "--test-label-files",
        nargs="+",
        default=["test_tasks_0627.json"],
        help="Testing json files under root",
    )
    parser.add_argument("--train-gt-name", default="train_gt.txt")
    parser.add_argument("--test-list-name", default="test.txt")
    parser.add_argument("--line-thickness", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")

    train_files = resolve_label_files(root, args.train_label_files)
    test_files = resolve_label_files(root, args.test_label_files)

    train_records = []
    for p in train_files:
        train_records.extend(load_json_lines(p))
    train_names, train_line_txt = records_to_lines(train_records)
    generate_segmentation_and_train_list(
        root=root,
        names=train_names,
        all_line_txt=train_line_txt,
        train_gt_name=args.train_gt_name,
        line_thickness=args.line_thickness,
    )

    test_records = []
    for p in test_files:
        test_records.extend(load_json_lines(p))
    test_names, _ = records_to_lines(test_records)
    generate_test_list(root=root, names=test_names, test_list_name=args.test_list_name)

    print(f"Done. train list: {root / args.train_gt_name}")
    print(f"Done. test list:  {root / args.test_list_name}")


if __name__ == "__main__":
    main()
