import json

import numpy as np
from sklearn.linear_model import LinearRegression


class LaneEval:
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0.0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise ValueError("Format of lanes error.")
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0.0, 0.0, 1.0

        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs, matched, fn = [], 0.0, 0.0

        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)

        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        score = sum(line_accs)
        if len(gt) > 4:
            score -= min(line_accs)
        return score / max(min(4.0, len(gt)), 1.0), fp / len(pred) if len(pred) > 0 else 0.0, fn / max(
            min(len(gt), 4.0), 1.0
        )

    @staticmethod
    def bench_one_submit(pred_file: str, gt_file: str) -> str:
        json_pred = [json.loads(line) for line in open(pred_file, "r", encoding="utf-8").readlines()]
        json_gt = [json.loads(line) for line in open(gt_file, "r", encoding="utf-8").readlines()]
        if len(json_gt) != len(json_pred):
            raise ValueError("prediction count does not match gt count")

        gts = {l["raw_file"]: l for l in json_gt}
        accuracy, fp, fn = 0.0, 0.0, 0.0
        for pred in json_pred:
            raw_file = pred["raw_file"]
            gt = gts[raw_file]
            a, p, n = LaneEval.bench(pred["lanes"], gt["lanes"], gt["h_samples"], pred.get("run_time", 10))
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        return json.dumps(
            [
                {"name": "Accuracy", "value": accuracy / num, "order": "desc"},
                {"name": "FP", "value": fp / num, "order": "asc"},
                {"name": "FN", "value": fn / num, "order": "asc"},
            ]
        )
