import random
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # The two lines below are known to slow down training, but ensure full reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config