from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class BaselineConfig:
    dataset: str = "Tusimple"
    data_root: Optional[str] = None
    train_list: str = "train_gt.txt"
    test_list: str = "test.txt"
    model_type: str = "baseline"  # baseline | lstm

    # model
    input_height: int = 288
    input_width: int = 800
    backbone: str = "resnet18"
    pretrained_backbone: bool = True
    griding_num: int = 100
    num_rows: int = 56
    num_lanes: int = 4
    cls_hidden_features: int = 2048

    # temporal model
    lstm_hidden_size: int = 512
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0

    # train
    epochs: int = 100
    batch_size: int = 32
    optimizer: str = "Adam"
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cos"
    steps: Tuple[int, int] = (50, 75)
    gamma: float = 0.1
    warmup: str = "linear"
    warmup_iters: int = 100
    num_workers: int = 16
    max_steps_per_epoch: Optional[int] = None
    seed: int = 3407

    # sequence dataloader (for lstm training path)
    sequence_length: int = 3
    sequence_stride: int = 1
    sequence_train_list: Optional[str] = None
    sequence_test_list: Optional[str] = None

    # runtime
    dry_run: bool = False
    output_dir: str = "outputs"
