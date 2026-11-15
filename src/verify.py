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


def feature_group(name: str) -> str:
    base = name.replace("dev_", "")
    for g, fs in FEATURE_GROUPS.items():
        if base in fs:
            return g
    return "Distance to references"
