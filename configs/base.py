from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BaselineConfig:
    dataset: str = "Tusimple"
    data_root: Optional[str] = None

    # model
    input_height: int = 288
    input_width: int = 800
    backbone: str = "resnet18"
    griding_num: int = 100
    num_rows: int = 56
    num_lanes: int = 4

    # train
    epochs: int = 100
    batch_size: int = 128
    optimizer: str = "Adam"
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cos"
    steps: Tuple[int, int] = (50, 75)
    gamma: float = 0.1
    warmup: str = "linear"
    warmup_iters: int = 100
    num_workers: int = 32
    max_steps_per_epoch: Optional[int] = None
    seed: int = 3407

    # runtime
    dry_run: bool = False
    output_dir: str = "outputs/baseline"
