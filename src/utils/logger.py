import logging
import time
from pathlib import Path

def create_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(Path(log_dir) / 'log.txt')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_new_log_dir(root=Path('./logs'), postfix='', prefix=''):
    log_dir = root / (prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir