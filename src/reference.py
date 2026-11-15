"""
Writer-normalised ("reference-based") features.

Why?
----
EDA shows that the difference *between two people's signatures* is much larger than
the difference between a person's genuine signature and a skilled forgery of it.
A raw feature such as stroke width = 4.1 px means nothing on its own: it is normal for
one writer and suspicious for another. Banks never judge a signature in isolation either -
they compare it with the specimen signatures kept on file.

So for every writer we keep N_REFERENCES genuine signatures "on file" and describe a
questioned signature q by how far each of its features lies from that writer's normal:

    dev_j = | x_qj - mean_ref_j | / sigma_j          (a z-score, in "writer standard deviations")
    sigma_j = sqrt( (std_ref_j^2 + s_j^2) / 2 )

std_ref_j is the spread of the writer's own references; because 5 signatures give a noisy
estimate, it is blended with s_j, the average within-writer spread measured on the
TRAINING writers only (no test information is used). log1p() then compresses very large
deviations so they do not dominate distance-based models (KNN, SVM).

Three summary "distance statistics" (in the spirit of Kalera, Srihari & Xu, 2004) are added:
    ref_dist_mean  - mean distance from q to the references
    ref_dist_min   - distance from q to the closest reference
    ref_dist_ratio - ref_dist_mean divided by the typical distance between the references
                     themselves (how unusual q is compared with the writer's own variation)

The resulting model is WRITER-INDEPENDENT: one classifier works for any writer, including
writers never seen in training, as long as a few of their genuine signatures are available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

DIST_FEATURES = ["ref_dist_mean", "ref_dist_min", "ref_dist_ratio"]


def select_references(index_df: pd.DataFrame, n_ref: int = config.N_REFERENCES,
                      seed: int = config.RANDOM_STATE) -> pd.Series:
    """Boolean mask: which genuine signatures of each writer act as references.

    The references are chosen at random (fixed seed) among the writer's 24 genuine
    signatures. They are never used as questioned samples.
    """
    rng = np.random.default_rng(seed)
    is_ref = pd.Series(False, index=index_df.index)
    genuine = index_df[index_df["label"] == config.LABEL_GENUINE]
    for _, grp in genuine.groupby("writer"):
        chosen = rng.choice(grp.index.to_numpy(), size=min(n_ref, len(grp) - 1), replace=False)
        is_ref.loc[chosen] = True
    return is_ref


def pooled_within_writer_std(ref_feats: pd.DataFrame, writers: pd.Series) -> pd.Series:
    """Average within-writer standard deviation of each feature (s_j above)."""
    var = ref_feats.groupby(writers.to_numpy()).var(ddof=1).mean(axis=0)
    s = np.sqrt(var)
    return s.replace(0, 1e-6).fillna(1e-6)


def writer_profile(ref_feats: pd.DataFrame, scale: pd.Series) -> dict:
    """Statistics of one writer's reference signatures (what the bank 'has on file')."""
    ref = ref_feats.to_numpy(dtype=float)
    s = scale.to_numpy(dtype=float)
    mean = ref.mean(axis=0)
    std = ref.std(axis=0, ddof=1) if len(ref) > 1 else np.zeros_like(mean)
    z_ref = ref / s
    k = len(ref)
    if k > 1:
        pair = [np.linalg.norm(z_ref[i] - z_ref[j]) for i in range(k) for j in range(i + 1, k)]
        spread = float(np.mean(pair)) / np.sqrt(ref.shape[1])
    else:
        spread = 1.0
    return {"mean": mean, "std": std, "z_ref": z_ref, "spread": max(spread, 1e-6)}


def questioned_features(x: np.ndarray, profile: dict, scale: pd.Series,
                        feature_names: list[str]) -> dict:
    """Writer-normalised feature vector of one questioned signature."""
    s = scale.to_numpy(dtype=float)
    sigma = np.sqrt((profile["std"] ** 2 + s ** 2) / 2.0)
    dev = np.log1p(np.abs(x - profile["mean"]) / sigma)
    out = {f"dev_{n}": float(v) for n, v in zip(feature_names, dev)}
    d = np.linalg.norm(profile["z_ref"] - x / s, axis=1) / np.sqrt(len(s))
    out["ref_dist_mean"] = float(d.mean())
    out["ref_dist_min"] = float(d.min())
    out["ref_dist_ratio"] = float(d.mean() / profile["spread"])
    return out


def build_reference_dataset(feat_df: pd.DataFrame, feature_names: list[str],
                            is_ref: pd.Series, scale: pd.Series) -> pd.DataFrame:
    """Turn the per-image feature table into the questioned-signature table.

    feat_df must contain the columns: writer, label, sample, file + feature_names.
    Every non-reference image becomes one row (remaining genuine + all forgeries).
    """
    rows = []
    for writer, grp in feat_df.groupby("writer"):
        ref = grp.loc[is_ref.loc[grp.index], feature_names]
        profile = writer_profile(ref, scale)
        questioned = grp.loc[~is_ref.loc[grp.index]]
        for idx, r in questioned.iterrows():
            q = questioned_features(r[feature_names].to_numpy(dtype=float), profile,
                                    scale, feature_names)
            q.update({"writer": int(writer), "label": int(r["label"]),
                      "sample": int(r["sample"]), "file": r["file"], "row_id": idx})
            rows.append(q)
    out = pd.DataFrame(rows)
    meta = ["row_id", "file", "writer", "sample", "label"]
    return out[meta + [c for c in out.columns if c not in meta]]
