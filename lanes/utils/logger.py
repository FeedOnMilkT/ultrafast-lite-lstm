import logging
from pathlib import Path


def build_logger(output_dir: str, name: str = "train"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(Path(output_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
