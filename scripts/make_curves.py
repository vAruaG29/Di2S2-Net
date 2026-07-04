#!/usr/bin/env python3
"""Generate training-curve figures for the presentation from Lightning
csv_logs/metrics.csv. Dependency: matplotlib only. Saves PNGs into docs/figures/.

Regeneration record — reads training run logs (logs/, logs_2val/) which are
gitignored and NOT shipped, so this only reproduces the committed figures on a
machine that still has those run directories. Run from the repo root:
    python scripts/make_curves.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ → repo root
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

ACCENT = "#ff5600"; BLUE = "#1565c0"; GREEN = "#2e7d32"; GREY = "#9aa0a6"
PRESENT = ["background", "Built_Up_Area", "Road", "Water_Body", "Utility"]  # no Bridge/Railway

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(path, split):
    """epoch -> {metric: float} using last non-empty value per epoch."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            ep = r.get("epoch", "")
            if ep in ("", "epoch", None):
                continue
            ep = int(float(ep))
            d = rows.setdefault(ep, {})
            for k, v in r.items():
                if v not in ("", None) and k not in ("epoch", "step"):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        pass
    eps = sorted(rows)
    def series(metric):
        xs, ys = [], []
        for e in eps:
            if metric in rows[e]:
                xs.append(e); ys.append(rows[e][metric])
        return xs, ys
    def miou_present(prefix):  # mean IoU over the 5 present classes
        xs, ys = [], []
        for e in eps:
            vals = [rows[e].get(f"{prefix}/iou/{c}") for c in PRESENT]
            if all(v is not None for v in vals):
                xs.append(e); ys.append(sum(vals) / len(vals))
        return xs, ys
    return rows, eps, series, miou_present


def curve(path, split, title, best_ep, fname, has_val=True):
    rows, eps, series, miou_present = load(path, split)
    pre = "val" if has_val else "train"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Panel 1 — loss
    if has_val:
        x, y = series("val/loss"); ax1.plot(x, y, color=ACCENT, lw=2.2, label="val loss")
    xt, yt = series("train/loss"); ax1.plot(xt, yt, color=GREY, lw=1.8, ls="--", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss"); ax1.legend()

    # Panel 2 — mIoU (7-class official + 5-class present)
    x7, y7 = series(f"{pre}/mIoU"); ax2.plot(x7, y7, color=BLUE, lw=1.8, ls=":",
                                             label=f"{pre} mIoU (7-class, incl. Bridge/Railway=0)")
    xp, yp = miou_present(pre); ax2.plot(xp, yp, color=GREEN, lw=2.4,
                                         label=f"{pre} mIoU (5-class, present only)")
    if has_val and best_ep is not None and best_ep in dict(zip(*miou_present(pre))):
        by = dict(zip(*miou_present(pre)))[best_ep]
        ax2.scatter([best_ep], [by], color=ACCENT, zorder=5, s=60)
        ax2.annotate(f"best ep {best_ep}\n5-cls mIoU={by:.3f}", (best_ep, by),
                     textcoords="offset points", xytext=(8, -28), color=ACCENT, fontweight="bold")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("mIoU"); ax2.set_ylim(0, 1)
    ax2.set_title("mean IoU"); ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle(title, fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, fname); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def comparison():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    runs = [
        ("logs_2val/run_20260605_091158/csv_logs/version_0/metrics.csv",
         "8:2 split (val = TIMMOWAL + NAGUL)", ACCENT),
        ("dinov3_hrdecoder_pipeline/logs/run_20260321_234449/csv_logs/version_0/metrics.csv",
         "9:1 split (val = NAGUL)", BLUE),
    ]
    for rel, lab, col in runs:
        _, _, _, miou_present = load(os.path.join(ROOT, rel), lab)
        x, y = miou_present("val")
        ax.plot(x, y, color=col, lw=2.4, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("val mIoU (5-class, present only)"); ax.set_ylim(0, 0.85)
    ax.set_title("Held-out generalization: val mIoU vs epoch")
    ax.legend(fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_val_miou_comparison.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


curve(os.path.join(ROOT, "logs_2val/run_20260605_091158/csv_logs/version_0/metrics.csv"),
      "8:2", "8:2 Split — held-out validation (TIMMOWAL + NAGUL)", 19, "fig_8_2_curves.png", has_val=True)
curve(os.path.join(ROOT, "dinov3_hrdecoder_pipeline/logs/run_20260321_234449/csv_logs/version_0/metrics.csv"),
      "9:1", "9:1 Split — held-out validation (NAGUL)", 14, "fig_9_1_curves.png", has_val=True)
curve(os.path.join(ROOT, "dinov3_hrdecoder_pipeline/logs/full_train_20260323_014307/csv_logs/version_0/metrics.csv"),
      "full", "Full Training — all 10 annotated villages (no held-out val)", None, "fig_full_train_curves.png", has_val=False)
comparison()
print("\nAll figures in", OUT)
