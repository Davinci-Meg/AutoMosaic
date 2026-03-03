"""ロギングユーティリティ"""

import logging
import sys


def setup_logger(verbose: bool = False) -> logging.Logger:
    """automosaic ロガーをセットアップして返す。

    Args:
        verbose: True なら DEBUG レベル、False なら INFO レベル。
    """
    logger = logging.getLogger("automosaic")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # ハンドラが重複しないようにする
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger
