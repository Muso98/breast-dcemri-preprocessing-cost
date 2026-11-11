"""Figure 2 of the paper: (a) resolution, (b) by lesion size, (c) truncation.
All values are case-level (seed-averaged), identical to stats_final.py."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

res = pd.read_csv(ROOT / "results" / "ressweep_percase.csv")
mag = pd.read_csv(ROOT / "results" / "magsweep_percase.csv")
res = (res.groupby(["held_out", "case", "spacing_mm"], as_index=False)
          [["dice", "gt_vox"]].mean())
mag = (mag.groupby(["held_out", "case", "k"], as_index=False)
          [["dice", "gt_vox"]].mean())

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
STYLE = {"NACT": dict(color=ORANGE, marker="o", ls="-"),
         "DUKE": dict(color=BLUE, marker="s", ls="--")}

# Times-like serif font (TeX Gyre Termes converted to TrueType) if present.
import glob as _g
from matplotlib import font_manager as _fm
for _f in _g.glob(str(ROOT / "fonts" / "*.ttf")):   # optional Times clone; falls back to default serif
    _fm.fontManager.addfont(_f)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TermesTT", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "text.color": INK, "axes.labelcolor": INK,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def tidy(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def ci(v, n=10000):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    bs = np.random.default_rng(0).choice(v, (n, len(v)), replace=True).mean(1)
    return np.percentile(bs, [2.5, 97.5])


b = res[res.spacing_mm == 1.5].set_index(["held_out", "case"])
r = res.join(b[["dice", "gt_vox"]].rename(columns=lambda c: c + "_1"),
             on=["held_out", "case"])
r["dd"] = r.dice - r.dice_1

fig, (axA, axB, axC) = plt.subplots(
    1, 3, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]})

# ---------------- (a) ----------------
XS = sorted(r.spacing_mm.unique())          # actual values: 1.5, 2.0, 3.0, 4.545...
for ho in ("NACT", "DUKE"):
    s = r[r.held_out == ho]
    ys = np.array([s[s.spacing_mm == x].dice.mean() for x in XS])
    cis = np.array([ci(s[s.spacing_mm == x].dice) for x in XS])
    axA.errorbar(XS, ys, yerr=[ys - cis[:, 0], cis[:, 1] - ys], lw=1.6, ms=4,
                 capsize=2, elinewidth=0.7, zorder=3, **STYLE[ho])
    axA.annotate(ho, (XS[-1], ys[-1]), xytext=(5, 0), textcoords="offset points",
                 color=STYLE[ho]["color"], fontsize=7.5, va="center")
axA.set_xticks(XS); axA.set_xticklabels(["1.5", "2", "3", "4.55"])
axA.set_xlim(1.25, 5.25)
axA.set_xlabel(r"Effective in-plane spacing (mm)")
axA.set_ylabel(r"Dice")
axA.set_title("(a) Resolution", loc="left")
tidy(axA)

# ---------------- (b) ----------------
W = 0.38
x = np.arange(4)
for i, ho in enumerate(("NACT", "DUKE")):
    s = r[(r.held_out == ho) & (r.spacing_mm == max(XS))].dropna(subset=["dd"])
    q = pd.qcut(s.gt_vox_1, 4, labels=False)
    axB.bar(x + (i - 0.5) * W, [s.dd[q == k].mean() for k in range(4)],
            W * 0.9, color=STYLE[ho]["color"], zorder=3, edgecolor="none")
axB.axhline(0, color=MUTED, lw=0.6, zorder=2)
axB.set_xticks(x); axB.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
axB.set_xlabel("Tumor volume quartile (Q1 smallest)")
axB.set_ylabel(r"$\Delta$Dice at 4.55 mm")
axB.set_title("(b) By lesion size", loc="left")
tidy(axB)

# ---------------- (c) ----------------
b1 = mag[mag.k == 1.0].set_index(["held_out", "case"])
m = mag.join(b1[["dice", "gt_vox"]].rename(columns=lambda c: c + "_1"),
             on=["held_out", "case"])
m["keep"] = m.gt_vox / m.k ** 2 / m.gt_vox_1
m["dd"] = m.dice - m.dice_1
d = m[(m.held_out == "DUKE") & (m.k == 2.0)].dropna(subset=["keep", "dd"])

axC.axhline(0, color=MUTED, lw=0.6, zorder=2)
axC.scatter(d.keep, d.dd, s=6, color=BLUE, alpha=0.3, edgecolors="none",
            zorder=3)
cx, cy = [], []
for lo, hi in [(0, .25), (.25, .75), (.75, .95), (.95, 9)]:
    q = d[(d.keep >= lo) & (d.keep < hi)]
    cx.append(q.keep.median()); cy.append(q.dd.mean())
axC.plot(cx, cy, color=INK, marker="o", ms=3.5, lw=1.4, zorder=4)
rho, p = stats.spearmanr(d.keep, d.dd)
axC.annotate(rf"$\rho = {rho:.2f}$" "\n" rf"$n$ = {len(d)}", (0.04, 0.95),
             xycoords="axes fraction", va="top", fontsize=7.5, color=INK)
axC.set_xlim(-0.04, 1.1)
axC.set_xlabel(r"Lesion retention in frame (DUKE, $k=2$)")
axC.set_ylabel(r"$\Delta$Dice")
axC.set_title("(c) Truncation", loc="left")
tidy(axC)

# ------------- shared legend -------------
h = [Line2D([], [], lw=1.6, ms=4, **STYLE["NACT"]),
     Line2D([], [], lw=1.6, ms=4, **STYLE["DUKE"]),
     Line2D([], [], color=INK, marker="o", ms=3.5, lw=1.4)]
fig.legend(h, ["held-out NACT", "held-out DUKE", "binned mean (c)"],
           loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.07), handlelength=2.2)

fig.tight_layout(w_pad=1.4)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_resolution.pdf")
print(f"figure written:  n={len(d)}  rho={rho:.3f}")
