"""
Signature Forgery Detection - live demo (Streamlit).

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.verify import MODEL_PATH, load_bundle, verify, pretty, feature_group  # noqa: E402
from src import plotting as P  # noqa: E402

SAMPLES = ROOT / "app" / "samples"

st.set_page_config(page_title="Signature Forgery Detection", page_icon="✍️", layout="wide")
P.set_style()


# ------------------------------------------------------------------ helpers
@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


def to_gray(file_or_path) -> np.ndarray:
    with Image.open(file_or_path) as im:
        return np.asarray(im.convert("L"), dtype=np.uint8)


@st.cache_data(show_spinner=False)
def cached_gray(path: str) -> np.ndarray:
    return to_gray(path)


def sample_writers():
    if not SAMPLES.exists():
        return []
    return sorted(p.name for p in SAMPLES.iterdir() if p.is_dir())


def true_label(fname: str) -> str:
    return "Forged" if fname.lower().startswith("forg") else "Genuine"


def deviation_chart(dev: pd.Series, top: int = 10):
    d = dev.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.26 * len(d) + 0.8), dpi=100)
    ax.barh([pretty(n) for n in d.index], d.values, color=P.MAIN, height=0.6)
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("Deviation from the writer's reference signatures (log scale units)", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def show_result(res: dict, truth: str | None = None):
    p = res["prob_forged"]
    decision = res["decision"]
    color = P.FORGED if decision == "Forged" else P.GENUINE
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:10px;background:{color}1A;"
            f"border:1px solid {color}66'>"
            f"<div style='font-size:14px;color:#52514e'>Decision</div>"
            f"<div style='font-size:30px;font-weight:700;color:{color}'>{decision.upper()}</div>"
            f"</div>", unsafe_allow_html=True)
    c2.metric("P(forged)", f"{p:.1%}")
    c3.metric("Decision threshold", f"{res['threshold']:.2f}")
    st.progress(min(max(p, 0.0), 1.0), text=f"Forgery probability {p:.1%}")
    if truth is not None:
        ok = truth == decision
        (st.success if ok else st.error)(
            f"Ground truth: **{truth}** -> the model is {'correct' if ok else 'wrong'} for this signature.")
    st.markdown("**Why?** Features of the questioned signature that deviate most from this writer's references:")
    st.pyplot(deviation_chart(res["deviations"]), clear_figure=True, width="content")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("Signature Verifier")
if not MODEL_PATH.exists():
    st.error("Model file not found. Run the notebook `notebooks/Signature_Forgery_Detection.ipynb` first "
             "(it saves `models/signature_verifier.joblib`).")
    st.stop()
bundle = get_bundle()
thr = bundle["thresholds"]
mode = st.sidebar.radio(
    "Operating point",
    ["Balanced (EER threshold)", "High security (low FAR)", "Default (0.5)"],
    help="High security flags more signatures as forged so that fewer forgeries slip through.")
threshold = {"Balanced (EER threshold)": thr["eer"], "High security (low FAR)": thr["high_security"],
             "Default (0.5)": thr["default"]}[mode]
m = bundle["test_metrics"]
st.sidebar.markdown(f"**Model:** {bundle['model_name']}")
st.sidebar.markdown(
    f"Test on 11 unseen writers  \n"
    f"Accuracy **{m['Accuracy']:.1%}** · ROC-AUC **{m['ROC_AUC']:.3f}**  \n"
    f"FAR **{m['FAR']:.1%}** · FRR **{m['FRR']:.1%}** · EER **{m['EER']:.1%}**")
st.sidebar.caption("FAR = forged accepted as genuine; FRR = genuine rejected as forged (threshold 0.5).")
st.sidebar.divider()
st.sidebar.caption("CSE2032 Machine Learning · BML Munjal University  \n"
                   "Nancy Yadav · Purva Joshi · Jiya Joshi")

# ------------------------------------------------------------------ header
st.title("✍️ Signature Forgery Detection")
st.write("Compare a **questioned signature** with a writer's **reference (genuine) signatures**. "
         "49 hand-crafted features (stroke thickness, curvature, tremor, aspect ratio, ink density, "
         "projection profiles ...) are extracted, expressed as deviations from the writer's normal "
         "signature, and classified by the trained model.")

TAB_NAMES = ["Demo: unseen CEDAR writers", "Verify your own images", "How it works"]
_tab_q = {"demo": TAB_NAMES[0], "upload": TAB_NAMES[1], "how": TAB_NAMES[2]}.get(st.query_params.get("tab", "demo"))
tab_demo, tab_upload, tab_how = st.tabs(TAB_NAMES, default=_tab_q)

# ------------------------------------------------------------------ tab 1
with tab_demo:
    writers = sample_writers()
    if not writers:
        st.info("No demo samples found in app/samples. Run the notebook to create them.")
    else:
        # optional URL parameters (?writer=writer_05&q=3&run=1) make a result shareable / reproducible
        qp = st.query_params
        w_default = writers.index(qp["writer"]) if qp.get("writer") in writers else 0
        wsel = st.selectbox("Writer (never seen during training)", writers, index=w_default,
                            format_func=lambda s: s.replace("writer_", "Writer "))
        ref_paths = sorted((SAMPLES / wsel / "references").glob("*.png"))
        q_paths = sorted((SAMPLES / wsel / "questioned").glob("*.png"))
        st.markdown(f"**Reference signatures on file ({len(ref_paths)})**")
        cols = st.columns(len(ref_paths))
        for c, pth in zip(cols, ref_paths):
            c.image(cached_gray(str(pth)), width="stretch")
        labels = [f"Questioned signature {i + 1}" for i in range(len(q_paths))]
        q_default = int(qp["q"]) if str(qp.get("q", "")).isdigit() and int(qp["q"]) < len(q_paths) else 0
        qi = st.selectbox("Questioned signature", range(len(q_paths)), index=q_default,
                          format_func=lambda i: labels[i])
        st.image(cached_gray(str(q_paths[qi])), caption="Questioned signature", width=420)
        if st.button("Verify signature", type="primary", key="demo_btn") or qp.get("run") == "1":
            with st.spinner("Extracting features and classifying ..."):
                res = verify(bundle, [cached_gray(str(p)) for p in ref_paths],
                             cached_gray(str(q_paths[qi])), threshold)
            show_result(res, truth=true_label(q_paths[qi].name))

# ------------------------------------------------------------------ tab 2
with tab_upload:
    st.write("Upload **3-10 genuine reference signatures** of one person and **one questioned signature**. "
             "Use clean scans on white paper (PNG / JPG).")
    refs_up = st.file_uploader("Reference signatures (genuine)", type=["png", "jpg", "jpeg", "bmp", "tif"],
                               accept_multiple_files=True)
    q_up = st.file_uploader("Questioned signature", type=["png", "jpg", "jpeg", "bmp", "tif"])
    if refs_up:
        cols = st.columns(min(len(refs_up), 6))
        for i, f in enumerate(refs_up[:6]):
            cols[i].image(to_gray(f), width="stretch")
    if q_up:
        st.image(to_gray(q_up), caption="Questioned signature", width=420)
    if st.button("Verify uploaded signature", type="primary", key="up_btn"):
        if not refs_up or len(refs_up) < 3 or q_up is None:
            st.warning("Please upload at least 3 reference signatures and 1 questioned signature.")
        else:
            with st.spinner("Extracting features and classifying ..."):
                res = verify(bundle, [to_gray(f) for f in refs_up], to_gray(q_up), threshold)
            show_result(res)

# ------------------------------------------------------------------ tab 3
with tab_how:
    st.subheader("Pipeline")
    st.markdown(
        "1. **Pre-processing** - grayscale, intensity normalisation (paper -> white), 3x3 median filter, "
        "Otsu threshold, removal of specks < 30 px, crop to the signature.\n"
        "2. **Feature derivation** - 49 features in 5 groups (stroke thickness, size & shape, "
        "curvature & structure, 3x4 pixel-density grid, projection profiles).\n"
        "3. **Writer normalisation** - each feature becomes |x - mean(ref)| / sigma, so the model judges "
        "*how unusual the signature is for this writer*.\n"
        "4. **Classification** - the model tuned with writer-grouped cross-validation outputs P(forged); "
        "the operating threshold controls the FAR / FRR trade-off.")
    writers = sample_writers()
    if writers:
        w0 = writers[0]
        q0 = sorted((SAMPLES / w0 / "questioned").glob("*.png"))[0]
        refs0 = sorted((SAMPLES / w0 / "references").glob("*.png"))
        res0 = verify(bundle, [cached_gray(str(p)) for p in refs0], cached_gray(str(q0)), threshold)
        pp = res0["preprocessing"]
        c1, c2, c3 = st.columns(3)
        c1.image(cached_gray(str(q0)), caption="1. Original scan", width="stretch")
        c2.image((~pp["ink"]).astype(np.uint8) * 255, caption="2. Binarised + cropped", width="stretch")
        c3.image((~pp["skeleton"]).astype(np.uint8) * 255, caption="3. Skeleton", width="stretch")
        tbl = pd.DataFrame({
            "Feature": [pretty(f) for f in res0["questioned_raw"].index],
            "Group": [feature_group(f) for f in res0["questioned_raw"].index],
            "Questioned": res0["questioned_raw"].values,
            "Writer's reference mean": res0["reference_mean"].values,
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
