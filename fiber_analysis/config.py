from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT = {
    "max_analysis_pixels": 500_000,
    "edge_crop_fraction": 0.01,
    "adaptive": {"background_sigma": 40, "contrast_sigma": 9, "z_threshold": 1.8},
    "ridge": {"sigmas": [1, 2, 3, 5], "low_percentile": 92, "high_percentile": 98},
    "lines": {"min_length": 25, "angle_degrees": 10, "gap": 20, "distance": 8},
    "postprocess": {"min_component_area": 20, "min_length_px": 20, "spur_length_px": 8},
    "unet": {"weights": None, "tile_size": 512, "overlap": 64, "threshold": 0.5},
}


def load(path: str | None) -> dict:
    config = deepcopy(DEFAULT)
    if not path:
        return config
    with Path(path).open() as f:
        supplied = yaml.safe_load(f) or {}
    for section, values in supplied.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    return config
