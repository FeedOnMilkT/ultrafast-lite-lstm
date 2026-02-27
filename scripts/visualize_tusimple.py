from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LANE_COLORS = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 128, 255),  # orange
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize TuSimple lane predictions on images (single model or baseline-vs-lstm compare)."
    )
    parser.add_argument("--data-root", required=True, help="Root containing raw images, e.g. .../TUSimple/test_set")
    parser.add_argument("--out-dir", required=True, help="Output directory for rendered images")

    parser.add_argument("--gt-file", type=str, default=None, help="Optional TuSimple gt json/jsonl file")

    parser.add_argument("--pred-file", type=str, default=None, help="Prediction file #1 (jsonl)")
    parser.add_argument("--pred-label", type=str, default="pred", help="Label for prediction #1")

    parser.add_argument("--pred2-file", type=str, default=None, help="Optional prediction file #2 (jsonl)")
    parser.add_argument("--pred2-label", type=str, default="pred2", help="Label for prediction #2")

    parser.add_argument("--num-samples", type=int, default=100, help="Number of images to visualize")
    parser.add_argument("--sample-mode", type=str, default="random", choices=["random", "first"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.7)
    return parser.parse_args()


def _read_json_lines(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid json at {path}:{line_idx}: {e}") from e
    return records


def _normalize_rel_path(raw_file: str) -> str:
    raw_file = raw_file.strip()
    if not raw_file:
        raise ValueError("empty raw_file")
    return raw_file[1:] if raw_file.startswith("/") else raw_file


def _records_to_map(records: List[dict], source_name: str) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for rec in records:
        if "raw_file" not in rec:
            raise ValueError(f"{source_name}: missing raw_file field in record")
        key = _normalize_rel_path(str(rec["raw_file"]))
        out[key] = rec
    return out


def _draw_lanes(
    image_bgr: np.ndarray,
    lanes: List[List[int]],
    h_samples: List[int],
    label: str,
    line_thickness: int,
    point_radius: int,
    font_scale: float,
) -> np.ndarray:
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    for lane_idx, lane in enumerate(lanes):
        color = LANE_COLORS[lane_idx % len(LANE_COLORS)]
        points = []
        for y, x in zip(h_samples, lane):
            if x == -2:
                continue
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                points.append((xi, yi))

        if len(points) >= 2:
            for p0, p1 in zip(points[:-1], points[1:]):
                cv2.line(canvas, p0, p1, color, line_thickness, lineType=cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, p, point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (w, 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _stack_h(images: List[np.ndarray]) -> np.ndarray:
    heights = [img.shape[0] for img in images]
    target_h = min(heights)
    resized = []
    for img in images:
        h, w = img.shape[:2]
        if h != target_h:
            new_w = int(round(w * (target_h / h)))
            img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized.append(img)
    return np.concatenate(resized, axis=1)


def _select_keys(all_keys: List[str], num_samples: int, mode: str, seed: int) -> List[str]:
    if num_samples <= 0:
        return []
    if len(all_keys) <= num_samples:
        return list(all_keys)
    if mode == "first":
        return list(all_keys[:num_samples])
    rng = random.Random(seed)
    return rng.sample(list(all_keys), k=num_samples)


def main():
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    if args.pred_file is None and args.gt_file is None:
        raise ValueError("at least one of --pred-file or --gt-file must be provided")

    gt_map: Optional[Dict[str, dict]] = None
    pred_map: Optional[Dict[str, dict]] = None
    pred2_map: Optional[Dict[str, dict]] = None

    if args.gt_file is not None:
        gt_path = Path(args.gt_file).expanduser().resolve()
        gt_map = _records_to_map(_read_json_lines(gt_path), source_name="gt")

    if args.pred_file is not None:
        pred_path = Path(args.pred_file).expanduser().resolve()
        pred_map = _records_to_map(_read_json_lines(pred_path), source_name="pred1")

    if args.pred2_file is not None:
        pred2_path = Path(args.pred2_file).expanduser().resolve()
        pred2_map = _records_to_map(_read_json_lines(pred2_path), source_name="pred2")

    key_sets = []
    if gt_map is not None:
        key_sets.append(set(gt_map.keys()))
    if pred_map is not None:
        key_sets.append(set(pred_map.keys()))
    if pred2_map is not None:
        key_sets.append(set(pred2_map.keys()))
    if not key_sets:
        raise RuntimeError("no available keys")

    common_keys = sorted(set.intersection(*key_sets))
    if not common_keys:
        raise ValueError("no common raw_file between provided sources")

    selected_keys = _select_keys(common_keys, args.num_samples, args.sample_mode, args.seed)
    if not selected_keys:
        raise ValueError("selected key list is empty")

    meta_records = []
    for idx, raw_key in enumerate(selected_keys, start=1):
        img_path = data_root / raw_key
        if not img_path.exists():
            print(f"[skip] missing image: {img_path}")
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] failed reading image: {img_path}")
            continue

        panels = []
        if gt_map is not None:
            gt = gt_map[raw_key]
            panels.append(
                _draw_lanes(
                    img,
                    lanes=gt["lanes"],
                    h_samples=gt["h_samples"],
                    label="GT",
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius,
                    font_scale=args.font_scale,
                )
            )
        if pred_map is not None:
            pred = pred_map[raw_key]
            panels.append(
                _draw_lanes(
                    img,
                    lanes=pred["lanes"],
                    h_samples=pred["h_samples"],
                    label=args.pred_label,
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius,
                    font_scale=args.font_scale,
                )
            )
        if pred2_map is not None:
            pred2 = pred2_map[raw_key]
            panels.append(
                _draw_lanes(
                    img,
                    lanes=pred2["lanes"],
                    h_samples=pred2["h_samples"],
                    label=args.pred2_label,
                    line_thickness=args.line_thickness,
                    point_radius=args.point_radius,
                    font_scale=args.font_scale,
                )
            )

        merged = _stack_h(panels) if len(panels) > 1 else panels[0]
        out_name = f"{idx:04d}__{raw_key.replace('/', '__')}"
        out_path = out_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), merged)

        meta_records.append(
            {
                "index": idx,
                "raw_file": raw_key,
                "image_path": str(img_path),
                "output_path": str(out_path),
            }
        )

    meta_path = out_dir / "index.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in meta_records:
            f.write(json.dumps(rec) + "\n")

    print(f"done: rendered={len(meta_records)} out_dir={out_dir}")
    print(f"index: {meta_path}")


if __name__ == "__main__":
    main()
