import json
from pathlib import Path

from lanes.eval.lane_eval_adapter import LaneEval
from lanes.eval.postprocess import TUSIMPLE_H_SAMPLES


def build_submit_record(raw_file: str, lanes, run_time: int = 10):
    return {
        "lanes": lanes,
        "h_samples": TUSIMPLE_H_SAMPLES,
        "raw_file": raw_file,
        "run_time": run_time,
    }


def dump_submit_records(records, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def evaluate_submit_file(pred_path: str, gt_path: str):
    result = LaneEval.bench_one_submit(pred_path, gt_path)
    return json.loads(result)
