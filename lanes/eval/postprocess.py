import numpy as np
import torch

TUSIMPLE_H_SAMPLES = [
    160,
    170,
    180,
    190,
    200,
    210,
    220,
    230,
    240,
    250,
    260,
    270,
    280,
    290,
    300,
    310,
    320,
    330,
    340,
    350,
    360,
    370,
    380,
    390,
    400,
    410,
    420,
    430,
    440,
    450,
    460,
    470,
    480,
    490,
    500,
    510,
    520,
    530,
    540,
    550,
    560,
    570,
    580,
    590,
    600,
    610,
    620,
    630,
    640,
    650,
    660,
    670,
    680,
    690,
    700,
    710,
]


def _softmax_np(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def generate_tusimple_lines(logits_one: torch.Tensor, griding_num: int, localization_type: str = "rel"):
    """
    Args:
      logits_one: [C, R, L]
    Returns:
      list[list[int]] lanes with x in image scale (1280), missing value=-2.
    """
    out = logits_one.detach().cpu().numpy()
    out_loc = np.argmax(out, axis=0)

    if localization_type == "rel":
        prob = _softmax_np(out[:-1, :, :], axis=0)
        idx = np.arange(griding_num).reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        loc[out_loc == griding_num] = griding_num
        out_loc = loc

    lanes = []
    for i in range(out_loc.shape[1]):
        lane = []
        for loc in out_loc[:, i]:
            if loc == griding_num:
                lane.append(-2)
            else:
                x = int(round((loc + 0.5) * 1280.0 / (griding_num - 1)))
                lane.append(x)
        lanes.append(lane)
    return lanes
