from dataclasses import dataclass
from typing import Optional


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
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    max_steps_per_epoch: int = 5
    seed: int = 3407

    # runtime
    dry_run: bool = True
    output_dir: str = "outputs/baseline"
