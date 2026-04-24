"""
Seed management for reproducibility.

Sets random, numpy, and torch seeds globally. Called once at the start of each stage.
"""

import random
import logging

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value (default 42 per spec §10).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seeds set: random={seed}, numpy={seed}, torch={seed}")
    except ImportError:
        logger.info(f"Seeds set: random={seed}, numpy={seed} (torch not available)")
