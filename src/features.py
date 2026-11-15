"""
Pre-processing and hand-crafted feature extraction for offline signatures.

Pipeline for one image
----------------------
1. Grayscale            : every scan is read as 8-bit grey (0 = black ink, 255 = white paper).
1b. Intensity normalisation: grey levels are stretched so the paper is 255 and the darkest ink 0,
                          which removes the scanner-brightness difference found in the EDA.
2. Noise removal      : 3x3 median filter removes salt-and-pepper scanner noise
                          without blurring stroke edges (unlike a mean/Gaussian filter).
3. Thresholding         : Otsu's method picks the grey level that best separates ink from
                          paper for *this* image, so scans of different brightness all work.
4. Speck removal        : connected ink blobs smaller than MIN_COMPONENT_SIZE pixels are dust.
5. Cropping             : the image is cut to the bounding box of the ink, which removes the
                          large blank margins (translation invariance).
6. Size normalisation   : projection profiles are computed on a resized fixed canvas, and every
                          density / position feature is expressed as a fraction, so signatures
                          of different sizes are comparable.

Stroke-width and grey-level (pressure) features are measured on the cropped image at the
original 300 dpi resolution, because resizing would change the measured stroke thickness.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, regionprops
from skimage.morphology import convex_hull_image, skeletonize
from skimage.transform import resize

from . import config
from .data import load_gray

EIGHT_CONN = np.ones((3, 3), dtype=bool)

# Feature groups (used for documentation, EDA and the report)
FEATURE_GROUPS = {
    "Size & shape": [
        "height", "width", "aspect_ratio", "ink_area", "ink_density", "solidity",
        "eccentricity", "principal_angle", "centroid_x", "centroid_y", "hu1", "hu2",
    ],
    "Stroke thickness & pressure": [
        "stroke_width_mean", "stroke_width_std", "stroke_width_p95",
        "pressure_contrast", "ink_gray_std",
    ],
    "Curvature & stroke structure": [
        "curvature_mean", "curvature_std", "sharp_turn_ratio", "tremor_index",
        "dir_0", "dir_45", "dir_90", "dir_135",
        "skeleton_length", "n_endpoints", "n_junctions", "n_components", "n_holes",
    ],
    "Pixel-density distribution (3x4 grid)": [
        f"density_r{r}c{c}" for r in range(1, 4) for c in range(1, 5)
    ],
    "Projection profiles": [
        "hproj_std", "hproj_skew", "hproj_kurt", "hproj_peak_pos",
        "vproj_std", "vproj_skew", "vproj_kurt", "vproj_peaks", "vproj_gap_ratio",
    ],
}
FEATURE_NAMES = [f for group in FEATURE_GROUPS.values() for f in group]

# Grey-level features are extracted for the EDA but NOT used by the models: in CEDAR the two
# classes were scanned with different paper brightness, so grey levels leak the label through
# an acquisition artefact rather than through the signature itself (see notebook Section 3).
INTENSITY_FEATURES = ["pressure_contrast", "ink_gray_std"]
MODEL_FEATURES = [f for f in FEATURE_NAMES if f not in INTENSITY_FEATURES]


# =============================================================== pre-processing
def normalize_intensity(gray: np.ndarray) -> np.ndarray:
    """Stretch grey levels so that the paper becomes white (255) and the darkest ink black (0).

    EDA showed that in CEDAR the genuine and the forged signatures were scanned with a
    different paper brightness (median grey 234-245 vs 249-255). Without this step the
    binarisation - and every feature computed from it - would partly encode the scanner
    setting instead of the signature.
    """
    g = np.asarray(gray, dtype=float)
    paper = np.median(g)                       # most pixels are paper
    ink = np.percentile(g, 0.5)                # darkest ink
    if paper - ink < 1:
        return np.asarray(gray, dtype=np.uint8)
    out = (g - ink) / (paper - ink) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(gray: np.ndarray,
               median_size: int = config.MEDIAN_FILTER_SIZE,
               min_size: int = config.MIN_COMPONENT_SIZE) -> dict:
    """Clean, binarise and crop one grayscale signature image.

    Returns a dict with the intermediate images (useful for plots and the app):
        normalized - intensity-normalised grey image (paper = 255)
        smooth   - median-filtered grey image (full size)
        binary   - Otsu binary mask after speck removal (full size, True = ink)
        gray     - original grey image cropped to the signature bounding box
        ink      - binary mask cropped to the bounding box
        skeleton - 1-pixel-wide skeleton of the cropped mask
        threshold, bbox
    """
    gray = normalize_intensity(gray)
    smooth = ndi.median_filter(gray, size=median_size)
    t = threshold_otsu(smooth)
    binary = smooth <= t                                 # Otsu: values > t are paper, <= t are ink

    labels, n = ndi.label(binary, structure=EIGHT_CONN)
    if n == 0:
        raise ValueError("No ink found in the image.")
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False                                      # label 0 is background
    if not keep.any():                                   # very faint/small image: keep largest blob
        keep[np.argmax(sizes[1:]) + 1] = True
    binary = keep[labels]

    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    ink = binary[r0:r1, c0:c1]
    return {
        "normalized": gray,
        "smooth": smooth,
        "binary": binary,
        "gray": gray[r0:r1, c0:c1],
        "ink": ink,
        "skeleton": skeletonize(ink),
        "threshold": float(t),
        "bbox": (int(r0), int(r1), int(c0), int(c1)),
    }


# =============================================================== helpers
def _profile_moments(profile: np.ndarray) -> tuple[float, float, float]:
    """Std, skewness and excess kurtosis of a projection profile treated as a
    probability distribution over normalised position [0, 1]."""
    p = profile.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = p / total
    pos = (np.arange(len(p)) + 0.5) / len(p)
    mean = (p * pos).sum()
    var = (p * (pos - mean) ** 2).sum()
    std = np.sqrt(var)
    if std < 1e-9:
        return 0.0, 0.0, 0.0
    skew = (p * (pos - mean) ** 3).sum() / std ** 3
    kurt = (p * (pos - mean) ** 4).sum() / std ** 4 - 3.0
    return float(std), float(skew), float(kurt)


def _contour_features(ink: np.ndarray, step: int = 4) -> dict:
    """Curvature, tremor and stroke-direction features from the ink outline.

    The outline is traced with marching squares and sampled every `step` pixels.
    For consecutive samples we measure the turning angle (curvature) and the
    direction of each segment (0, 45, 90 or 135 degrees, length weighted).
    """
    contours = find_contours(np.pad(ink, 1).astype(float), 0.5)
    turns, fine_len, coarse_len = [], 0.0, 0.0
    dir_hist = np.zeros(4)
    for c in contours:
        if len(c) < 3 * step:
            continue
        d_fine = np.diff(c, axis=0)
        fine_len += np.hypot(d_fine[:, 0], d_fine[:, 1]).sum()
        s = c[::step]
        v = np.diff(s, axis=0)
        seg = np.hypot(v[:, 0], v[:, 1])
        coarse_len += seg.sum()
        ang = np.arctan2(-v[:, 0], v[:, 1])              # image rows point down -> flip
        turn = np.diff(ang)
        turn = (turn + np.pi) % (2 * np.pi) - np.pi      # wrap to [-pi, pi]
        turns.append(np.abs(turn))
        orient = np.degrees(ang) % 180.0
        bins = (((orient + 22.5) // 45).astype(int)) % 4
        dir_hist += np.bincount(bins, weights=seg, minlength=4)

    if turns:
        t = np.concatenate(turns)
        curv_mean, curv_std = float(t.mean()), float(t.std())
        sharp = float((t > np.pi / 4).mean())
    else:
        curv_mean = curv_std = sharp = 0.0
    dir_frac = dir_hist / dir_hist.sum() if dir_hist.sum() > 0 else np.full(4, 0.25)
    return {
        "curvature_mean": curv_mean,
        "curvature_std": curv_std,
        "sharp_turn_ratio": sharp,
        "tremor_index": float(fine_len / coarse_len) if coarse_len > 0 else 1.0,
        "dir_0": float(dir_frac[0]), "dir_45": float(dir_frac[1]),
        "dir_90": float(dir_frac[2]), "dir_135": float(dir_frac[3]),
    }


def _count_holes(ink: np.ndarray, min_hole: int = 10) -> int:
    """Enclosed background regions (loops such as in 'a', 'o', 'e')."""
    bg_labels, n = ndi.label(~ink)                       # 4-connectivity for background
    if n == 0:
        return 0
    border = np.unique(np.concatenate([bg_labels[0], bg_labels[-1],
                                       bg_labels[:, 0], bg_labels[:, -1]]))
    sizes = np.bincount(bg_labels.ravel())
    holes = [lab for lab in range(1, n + 1) if lab not in border and sizes[lab] >= min_hole]
    return len(holes)


# =============================================================== main extractor
def extract_features(gray: np.ndarray, return_intermediate: bool = False):
    """Compute the 51 hand-crafted features of one signature image."""
    pp = preprocess(gray)
    ink, g, skel = pp["ink"], pp["gray"], pp["skeleton"]
    H, W = ink.shape
    area = float(ink.sum())
    f: dict[str, float] = {}

    # ---------------- size & shape --------------------------------------
    f["height"] = float(H)
    f["width"] = float(W)
    f["aspect_ratio"] = W / H
    f["ink_area"] = area
    f["ink_density"] = area / (H * W)
    hull = convex_hull_image(ink)
    f["solidity"] = area / max(hull.sum(), 1)

    ys, xs = np.nonzero(ink)
    cx, cy = xs.mean(), ys.mean()
    mu20 = ((xs - cx) ** 2).mean()
    mu02 = ((ys - cy) ** 2).mean()
    mu11 = ((xs - cx) * (cy - ys)).mean()                # y axis pointing up
    common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    lam1 = (mu20 + mu02) / 2 + common
    lam2 = (mu20 + mu02) / 2 - common
    f["eccentricity"] = float(np.sqrt(1 - lam2 / lam1)) if lam1 > 0 else 0.0
    f["principal_angle"] = float(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))
    f["centroid_x"] = (cx + 0.5) / W
    f["centroid_y"] = (cy + 0.5) / H
    hu = regionprops(ink.astype(np.uint8))[0].moments_hu
    f["hu1"] = float(-np.sign(hu[0]) * np.log10(abs(hu[0]) + 1e-30))
    f["hu2"] = float(-np.sign(hu[1]) * np.log10(abs(hu[1]) + 1e-30))

    # ---------------- stroke thickness & pressure -----------------------
    dist = ndi.distance_transform_edt(ink)
    widths = 2.0 * dist[skel] - 1.0                      # local stroke width on the medial axis
    if widths.size == 0:
        widths = np.array([1.0])
    f["stroke_width_mean"] = float(widths.mean())
    f["stroke_width_std"] = float(widths.std())
    f["stroke_width_p95"] = float(np.percentile(widths, 95))
    ink_vals = g[ink].astype(float)
    bg_vals = g[~ink]
    bg_level = float(np.median(bg_vals)) if bg_vals.size else 255.0
    f["pressure_contrast"] = bg_level - float(ink_vals.mean())   # darker ink = more pressure
    f["ink_gray_std"] = float(ink_vals.std())

    # ---------------- curvature & stroke structure ----------------------
    f.update(_contour_features(ink))
    f["skeleton_length"] = float(skel.sum())
    nb = ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - 1
    f["n_endpoints"] = float((skel & (nb == 1)).sum())
    _, n_junc = ndi.label(skel & (nb >= 3), structure=EIGHT_CONN)
    f["n_junctions"] = float(n_junc)
    _, n_comp = ndi.label(ink, structure=EIGHT_CONN)
    f["n_components"] = float(n_comp)
    f["n_holes"] = float(_count_holes(ink))

    # ---------------- pixel-density distribution (3 x 4 grid) ----------
    row_parts = np.array_split(np.arange(H), 3)
    col_parts = np.array_split(np.arange(W), 4)
    for i, rr in enumerate(row_parts, start=1):
        for j, cc in enumerate(col_parts, start=1):
            cell = ink[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]
            f[f"density_r{i}c{j}"] = float(cell.sum() / area)

    # ---------------- projection profiles (size-normalised canvas) ------
    canvas = resize(ink.astype(float), (config.NORM_HEIGHT, config.NORM_WIDTH),
                    anti_aliasing=True, preserve_range=True)
    hp = canvas.sum(axis=1)                              # horizontal projection (ink per row)
    vp = canvas.sum(axis=0)                              # vertical projection (ink per column)
    f["hproj_std"], f["hproj_skew"], f["hproj_kurt"] = _profile_moments(hp)
    hp_s = ndi.gaussian_filter1d(hp, 2)
    f["hproj_peak_pos"] = float((np.argmax(hp_s) + 0.5) / len(hp_s))   # ~ baseline height
    f["vproj_std"], f["vproj_skew"], f["vproj_kurt"] = _profile_moments(vp)
    vp_s = ndi.gaussian_filter1d(vp, 3)
    peaks, _ = find_peaks(vp_s, prominence=0.15 * vp_s.max()) if vp_s.max() > 0 else ([], None)
    f["vproj_peaks"] = float(len(peaks))
    f["vproj_gap_ratio"] = float((ink.sum(axis=0) == 0).mean())         # empty columns (gaps)

    feats = {k: float(f[k]) for k in FEATURE_NAMES}      # fixed column order
    if return_intermediate:
        return feats, pp
    return feats


def extract_from_path(path) -> dict:
    """Convenience wrapper: features of one image file."""
    return extract_features(load_gray(path))


def extract_record(path) -> dict:
    """Raw-image metadata (for EDA / data-quality checks) + all features of one file.

    Used for the parallel pass over the whole dataset. Errors are recorded instead
    of raised so that one unreadable file cannot stop the whole extraction.
    """
    import hashlib
    from pathlib import Path

    raw = Path(path).read_bytes()
    gray = load_gray(path)
    rec = {
        "md5": hashlib.md5(raw).hexdigest(),          # duplicate detection
        "img_height": int(gray.shape[0]),
        "img_width": int(gray.shape[1]),
        "gray_mean": float(gray.mean()),
        "background_level": float(np.median(gray)),   # paper brightness
        "dark_pixel_frac": float((gray < 128).mean()),
        "error": "",
    }
    try:
        feats, pp = extract_features(gray, return_intermediate=True)
        rec["otsu_threshold"] = pp["threshold"]
        rec["noise_pixels_removed"] = int((pp["smooth"] <= pp["threshold"]).sum() - pp["binary"].sum())
        rec.update(feats)
    except Exception as exc:                          # pragma: no cover
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec
