"""
Inference for one questioned signature against a writer's reference signatures.
Used by the Streamlit demo app (app/streamlit_app.py).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .features import FEATURE_GROUPS, extract_features
from .reference import questioned_features, writer_profile

MODEL_PATH = config.MODEL_DIR / "signature_verifier.joblib"

# readable names for the explanation chart
PRETTY = {
    "stroke_width_mean": "Stroke thickness (mean)", "stroke_width_std": "Stroke thickness (variation)",
    "stroke_width_p95": "Stroke thickness (max)", "pressure_contrast": "Ink darkness / pressure",
    "ink_gray_std": "Ink tone variation", "aspect_ratio": "Aspect ratio", "height": "Height",
    "width": "Width", "ink_area": "Ink area", "ink_density": "Ink density", "solidity": "Solidity",
    "eccentricity": "Elongation", "principal_angle": "Slant of main axis", "centroid_x": "Centre (x)",
    "centroid_y": "Centre (y)", "hu1": "Shape moment 1", "hu2": "Shape moment 2",
    "curvature_mean": "Curvature (mean)", "curvature_std": "Curvature (variation)",
    "sharp_turn_ratio": "Sharp turns", "tremor_index": "Tremor / jaggedness",
    "dir_0": "Horizontal strokes", "dir_45": "Rising diagonal strokes", "dir_90": "Vertical strokes",
    "dir_135": "Falling diagonal strokes", "skeleton_length": "Total stroke length",
    "n_endpoints": "Stroke end points", "n_junctions": "Stroke junctions",
    "n_components": "Pen lifts (components)", "n_holes": "Loops",
    "hproj_std": "Vertical spread", "hproj_skew": "Vertical skew", "hproj_kurt": "Vertical peakedness",
    "hproj_peak_pos": "Baseline position", "vproj_std": "Horizontal spread",
    "vproj_skew": "Horizontal skew", "vproj_kurt": "Horizontal peakedness",
    "vproj_peaks": "Letter-like peaks", "vproj_gap_ratio": "Gaps between parts",
    "ref_dist_mean": "Overall distance to references", "ref_dist_min": "Distance to closest reference",
    "ref_dist_ratio": "Distance vs writer's own variation",
}
for _r in range(1, 4):
    for _c in range(1, 5):
        PRETTY[f"density_r{_r}c{_c}"] = f"Ink share, grid cell ({_r},{_c})"


def pretty(name: str) -> str:
    return PRETTY.get(name.replace("dev_", ""), name)


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    return joblib.load(path)


def verify(bundle: dict, reference_grays: list[np.ndarray], questioned_gray: np.ndarray,
           threshold: float | None = None) -> dict:
    """Return P(forged), the decision and the writer-normalised deviations."""
    if len(reference_grays) < 2:
        raise ValueError("At least 2 reference signatures are needed (5 recommended).")
    F = bundle["raw_features"]
    ref = pd.DataFrame([extract_features(g) for g in reference_grays])[F]
    q_raw, pp = extract_features(questioned_gray, return_intermediate=True)
    profile = writer_profile(ref, bundle["scale"])
    q = questioned_features(np.array([q_raw[f] for f in F], dtype=float), profile, bundle["scale"], F)
    X = pd.DataFrame([q])[bundle["selected_features"]]
    p = float(bundle["model"].predict_proba(X)[0, 1])
    thr = bundle["thresholds"]["default"] if threshold is None else threshold
    dev = pd.Series({k: v for k, v in q.items() if k in bundle["selected_features"]})
    return {
        "prob_forged": p,
        "threshold": thr,
        "decision": "Forged" if p >= thr else "Genuine",
        "deviations": dev.sort_values(ascending=False),
        "questioned_raw": pd.Series(q_raw)[F],
        "reference_mean": pd.Series(profile["mean"], index=F),
        "preprocessing": pp,
    }


# ------------------------------------------------------------------ uploads (phone photos / other scanners)
# The model was trained on 300-dpi CEDAR scans with even lighting. User uploads are often phone photos:
# uneven lighting and a different (or varying) scale. Two corrections bring them to CEDAR conditions;
# both were validated on the unseen test writers (see README, "Robustness checks").
CEDAR_SIGNATURE_HEIGHT = 196.0      # median signature height (px) in CEDAR


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Remove shadows / uneven lighting: divide by a smooth estimate of the paper brightness."""
    from scipy import ndimage as ndi
    from skimage.transform import resize

    g = np.asarray(gray, dtype=float)
    step = max(1, int(round(max(g.shape) / 400)))          # estimate on a ~400 px copy (fast)
    small = g[::step, ::step]
    k = max(15, int(min(small.shape) / 8))
    bg = ndi.grey_closing(small, size=(k, k))               # removes dark ink strokes thinner than k
    bg = ndi.gaussian_filter(bg, k / 2)
    bg = resize(bg, g.shape, order=1, preserve_range=True)
    return np.clip(g / np.maximum(bg, 1.0) * 255.0, 0, 255).astype(np.uint8)


def ink_height(gray: np.ndarray) -> float:
    """Height (px) of the signature's ink bounding box, measured quickly on a reduced copy."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    from .features import normalize_intensity

    g = np.asarray(gray, dtype=np.uint8)
    step = max(1, int(np.ceil(max(g.shape) / 1000)))
    small = normalize_intensity(g[::step, ::step])
    small = ndi.median_filter(small, size=3)
    ink = small <= threshold_otsu(small)
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n == 0:
        return float(g.shape[0])
    sizes = np.bincount(lab.ravel())
    keep = sizes >= max(4, 30 / step ** 2)
    keep[0] = False
    rows = np.flatnonzero(keep[lab].any(axis=1))
    return float((rows[-1] - rows[0] + 1) * step) if rows.size else float(g.shape[0])


def _rescale(gray: np.ndarray, factor: float) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(np.asarray(gray, dtype=np.uint8))
    w, h = max(8, int(round(im.width * factor))), max(8, int(round(im.height * factor)))
    return np.asarray(im.resize((w, h), Image.LANCZOS if factor < 1 else Image.BILINEAR))


def prepare_uploads(reference_grays, questioned_gray, flatten: bool = True, match_scale: bool = True):
    """Bring user images to CEDAR-like conditions before `verify`.

    flatten     - remove uneven lighting / shadows from every image;
    match_scale - rescale ALL images with ONE common factor so that the references' median signature height
                  equals the CEDAR median (handles large phone photos and different camera resolutions).
                  The questioned image is NOT rescaled individually: that would distort its stroke
                  thickness, one of the most important features. Validation on the unseen test writers:
                  normal scans 89.9 % -> 91.1 % accuracy, uneven lighting 71.5 % -> 92.0 %,
                  phone-size images with uneven lighting 90.1 %.
    info["size_ratio"] = questioned height / median reference height; values far from 1 usually mean the
    photos were taken from different distances (results then become unreliable).
    """
    refs = [np.asarray(g, dtype=np.uint8) for g in reference_grays]
    q = np.asarray(questioned_gray, dtype=np.uint8)
    if flatten:
        refs = [flatten_background(g) for g in refs]
        q = flatten_background(q)
    ref_h = float(np.median([ink_height(g) for g in refs]))
    info = {"flatten": flatten, "match_scale": match_scale, "size_ratio": ink_height(q) / max(ref_h, 1.0)}
    if match_scale:
        f = CEDAR_SIGNATURE_HEIGHT / max(ref_h, 1.0)
        refs = [_rescale(g, f) for g in refs]
        q = _rescale(q, f)
        info["scale_factor"] = f
    return refs, q, info


def feature_group(name: str) -> str:
    base = name.replace("dev_", "")
    for g, fs in FEATURE_GROUPS.items():
        if base in fs:
            return g
    return "Distance to references"
