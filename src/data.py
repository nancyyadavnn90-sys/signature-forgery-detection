"""
Dataset indexing for the CEDAR signature database.

CEDAR layout after extraction:
    data/CEDAR/full_org/original_<writer>_<n>.png     (genuine,  label 0)
    data/CEDAR/full_forg/forgeries_<writer>_<n>.png   (forged,   label 1)
55 writers x 24 genuine + 55 writers x 24 skilled forgeries = 2,640 images.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

_PATTERN = re.compile(r"^(original|forgeries)_(\d+)_(\d+)\.png$", re.IGNORECASE)


def build_index(cedar_dir: Path = config.CEDAR_DIR) -> pd.DataFrame:
    """Return one row per image: path, writer id, sample number and label."""
    rows = []
    for sub, label in (("full_org", config.LABEL_GENUINE), ("full_forg", config.LABEL_FORGED)):
        folder = Path(cedar_dir) / sub
        if not folder.exists():
            raise FileNotFoundError(
                f"{folder} not found. Download CEDAR and extract it so that "
                f"'full_org' and 'full_forg' are inside {cedar_dir}."
            )
        for p in sorted(folder.iterdir()):
            m = _PATTERN.match(p.name)
            if m is None:            # skips Thumbs.db and any stray file
                continue
            rows.append({
                "path": str(p),
                "file": p.name,
                "writer": int(m.group(2)),
                "sample": int(m.group(3)),
                "label": label,
            })
    df = pd.DataFrame(rows).sort_values(["writer", "label", "sample"]).reset_index(drop=True)
    return df


def load_gray(path) -> np.ndarray:
    """Load any image as an 8-bit grayscale numpy array (0 = black, 255 = white)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


def image_metadata(path) -> dict:
    """Raw image properties used in EDA (size and grey-level statistics)."""
    g = load_gray(path)
    return {
        "img_height": g.shape[0],
        "img_width": g.shape[1],
        "gray_mean": float(g.mean()),
        "gray_median": float(np.median(g)),
        "gray_min": int(g.min()),
        "gray_p01": float(np.percentile(g, 1)),
    }
