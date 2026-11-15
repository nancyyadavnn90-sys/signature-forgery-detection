"""
Static diagrams for the report and slides: the ML pipeline and the deployment architecture.

    python -m src.diagrams
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import plotting as P

FILL = {"data": "#e8f1fc", "proc": "#eeecfb", "model": "#e6f6f0", "eval": "#fdf3df"}
EDGE = {"data": "#2a78d6", "proc": "#6250d6", "model": "#1baf7a", "eval": "#c98500"}


def _box(ax, x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 0.03, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=P.INK)
    ax.text(x + w / 2, y + h - 0.075, body, ha="center", va="top", fontsize=8.6,
            color=P.INK_2, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color=P.INK_2, lw=1.3))


def pipeline(path):
    P.set_style()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 1.225); ax.set_ylim(0, 0.5); ax.axis("off")
    w, h = 0.235, 0.19
    top, bot = 0.29, 0.03
    xs = [0.02, 0.335, 0.65, 0.965]
    row1 = [
        ("1. CEDAR dataset", "55 writers\n24 genuine + 24 forged each\n2,640 grey-scale scans", "data"),
        ("2. EDA", "class balance, image size,\npaper brightness, noise,\nwriter vs class variance", "data"),
        ("3. Pre-processing", "grayscale -> median filter\n-> Otsu threshold -> remove\nspecks -> crop -> skeleton", "proc"),
        ("4. Feature derivation", "51 derived (49 used): stroke\nwidth, aspect ratio, curvature,\ndensity, projection profiles", "proc"),
    ]
    row2 = [
        ("8. Evaluation", "11 unseen writers: Accuracy,\nPrecision, Recall, F1, ROC-AUC,\nFAR, FRR, EER + threshold", "eval"),
        ("7. Models + tuning", "LogReg (L1/L2), SVM, KNN,\nDecision Tree, RF, GB, AdaBoost,\nVoting; GridSearchCV", "model"),
        ("6. Writer-wise split", "44 train writers (GroupKFold-5)\n11 test writers never seen\n+ correlation filter", "model"),
        ("5. Writer normalisation", "deviation of each feature from\nthe writer's 5 reference\nsignatures + distance stats", "proc"),
    ]
    for x, (t, b, k) in zip(xs, row1):
        _box(ax, x, top, w, h, t, b, k)
    for x, (t, b, k) in zip(xs, row2):
        _box(ax, x, bot, w, h, t, b, k)
    for i in range(3):
        _arrow(ax, xs[i] + w + 0.004, top + h / 2, xs[i + 1] - 0.004, top + h / 2)
        _arrow(ax, xs[i + 1] - 0.004, bot + h / 2, xs[i] + w + 0.004, bot + h / 2)
    _arrow(ax, xs[3] + w / 2, top - 0.004, xs[3] + w / 2, bot + h + 0.004)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def deployment(path):
    P.set_style()
    fig, ax = plt.subplots(figsize=(12.5, 3.9))
    ax.set_xlim(-0.02, 1.25); ax.set_ylim(0, 0.42); ax.axis("off")
    w, h = 0.215, 0.19
    y = 0.11
    boxes = [
        (0.01, "User (browser)", "uploads reference +\nquestioned signatures\nor picks a demo writer", "data"),
        (0.325, "Streamlit app", "app/streamlit_app.py\nUI, threshold choice,\nexplanation chart", "eval"),
        (0.64, "Inference (src/)", "pre-processing ->\n49 features -> writer\nnormalisation", "proc"),
        (0.955, "Trained model", "models/signature_\nverifier.joblib\n(+ scale, features)", "model"),
    ]
    for x, t, b, k in boxes:
        _box(ax, x, y, w, h, t, b, k)
    for i in range(3):
        x0 = boxes[i][0] + w
        x1 = boxes[i + 1][0]
        _arrow(ax, x0 + 0.004, y + h * 0.62, x1 - 0.004, y + h * 0.62)
        _arrow(ax, x1 - 0.004, y + h * 0.32, x0 + 0.004, y + h * 0.32)
    ax.text(0.5 * (boxes[1][0] + boxes[3][0] + w), y + h + 0.075,
            "Hosted on Streamlit Community Cloud (or run locally: streamlit run app/streamlit_app.py)",
            ha="center", fontsize=9.5, color=P.INK_2)
    ax.add_patch(FancyBboxPatch((0.305, y - 0.04), 0.93, h + 0.095, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc="none", ec=P.AXIS, lw=1.2, ls="--"))
    ax.text(0.5 * (boxes[0][0] * 2 + w), y - 0.07, "Response: decision, P(forged),\nmost deviating features",
            ha="center", va="top", fontsize=8.5, color=P.INK_2)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    from . import config
    pipeline(config.FIG_DIR / "fig00_pipeline.png")
    deployment(config.FIG_DIR / "fig25_deployment.png")
    print("diagrams saved")
