from __future__ import annotations

import logging
import time
from pathlib import Path


def create_logger(name: str, log_dir: Path | str | None = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file.

    Constructs a :class:`logging.Logger` with a consistent timestamp/name/level
    format.  A :class:`logging.StreamHandler` at DEBUG level is always attached.
    When *log_dir* is provided a :class:`logging.FileHandler` writing to
    ``<log_dir>/log.txt`` at INFO level is also attached.

    Args:
        name: Logger name, typically the calling module's ``__name__``.
        log_dir: Optional directory in which to create ``log.txt``.  If
            ``None`` no file handler is added.

    Returns:
        Fully configured :class:`logging.Logger` instance.
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


def get_new_log_dir(
    root: Path = Path('./logs'),
    prefix: str = '',
    postfix: str = '',
) -> Path:
    """Create and return a timestamped log directory.

    Builds a directory name from an optional *prefix*, the current local time
    formatted as ``YYYY_MM_DD__HH_MM_SS``, and an optional *postfix*.  The
    directory (and any missing parents) is created before the path is returned.

    Args:
        root: Parent directory under which all timestamped log folders are
            created.  Defaults to ``./logs``.
        prefix: String prepended to the timestamp component of the folder name.
        postfix: String appended to the timestamp component of the folder name.

    Returns:
        :class:`pathlib.Path` pointing to the newly created log directory.
    """
    log_dir = root / (prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
