import logging
import time
from pathlib import Path


def create_logger(name: str, log_dir=None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file.

    Args:
        name: Logger name.
        log_dir: Optional directory to write ``log.txt`` into.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        fh = logging.FileHandler(Path(log_dir) / 'log.txt')
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_new_log_dir(root: Path = Path('./logs'), prefix: str = '', postfix: str = '') -> Path:
    """Create and return a timestamped log directory.

    Args:
        root: Parent directory for all log folders.
        prefix: Optional prefix string.
        postfix: Optional postfix string.

    Returns:
        Path to the newly created directory.
    """
    log_dir = root / (prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
