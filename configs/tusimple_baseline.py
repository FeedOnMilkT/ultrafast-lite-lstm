from configs.base import BaselineConfig


def get_config() -> BaselineConfig:
    cfg = BaselineConfig()
    cfg.model_type = "baseline"
    return cfg
