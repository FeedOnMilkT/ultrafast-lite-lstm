from configs.base import BaselineConfig


def get_config() -> BaselineConfig:
    cfg = BaselineConfig()
    cfg.model_type = "lstm"
    cfg.sequence_length = 3
    cfg.sequence_stride = 1
    cfg.lstm_hidden_size = 512
    cfg.lstm_num_layers = 1
    cfg.lstm_dropout = 0.0
    cfg.cls_hidden_features = 1024
    return cfg
