
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom as ndzoom

NPZ = ROOT / "results" / "figure_slices.npz"   # produced by notebooks/06_figure_slices.ipynb (not distributed)
z = np.load(NPZ)
meta = json.loads(str(z["meta"]))

IMG = 192
BASE = 1.5
ORANGE, INK, MUTED = "#eb6834", "#0b0b0b", "#52514e"

PICK = {"DUKE": 1, "ISPY1": 2, "ISPY2": 2, "NACT": 2}
ROW1 = ["DUKE", "ISPY1", "ISPY2", "NACT"]
RES = [1.0, 0.75, 0.5, 0.33]          # -> 1.5, 2.0, 3.0, 4.55 mm

import glob as _g
from matplotlib import font_manager as _fm
for _f in _g.glob(str(ROOT / "fonts" / "*.ttf")):   # optional Times clone; falls back to default serif
    _fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["TermesTT", "Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 9,
    "figure.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def fit_center(a, pad):
    H, W = a.shape
    if (H, W) == (IMG, IMG):
        return a
    out = np.full((IMG, IMG), pad, a.dtype)
    hs, ws = min(H, IMG), min(W, IMG)
    sy, sx = (H - hs) // 2, (W - ws) // 2
    dy, dx = (IMG - hs) // 2, (IMG - ws) // 2
    out[dy:dy+hs, dx:dx+ws] = a[sy:sy+hs, sx:sx+ws]
    return out


def degrade(img, r):
    """192 -> 192r -> 192, identical to notebooks/04_resolution_sweep.ipynb."""
    if r >= 0.999:
        return img
    small = ndzoom(img, r, order=1, mode="nearest")
    back = ndzoom(small, (IMG / small.shape[0], IMG / small.shape[1]),
                  order=1, mode="nearest")
    return fit_center(back, 0.0)


def show(ax, img, msk=None, lw=0.9, up=2):
    """Nearest-neighbour upsampling by `up` for >=300 ppi print; data unchanged."""
    lo, hi = np.percentile(img, [1, 99.5])
    H, W = img.shape
    big = np.kron(img, np.ones((up, up), img.dtype))
    ax.imshow(big, cmap="gray", vmin=lo, vmax=hi, interpolation="none",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    if msk is not None and msk.any():
        ax.contour(msk, levels=[0.5], colors=ORANGE, linewidths=lw)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.75))

# ---- row (a): four sources at the same physical scale ----
for ax, s in zip(axes[0], ROW1):
    i = PICK[s]
    show(ax, z[f"{s}_{i}_img"], z[f"{s}_{i}_msk"], up=2)
    ax.set_title(s.replace("ISPY", "I-SPY"), color=INK, pad=3)

# ---- row (b): one case at four effective resolutions ----
s, i = "DUKE", PICK["DUKE"]
img, msk = z[f"{s}_{i}_img"], z[f"{s}_{i}_msk"]

# Crop around the lesion for display only; degrade() is always applied
# to the full 192x192 frame.
HALF = 44
cy, cx = (np.argwhere(msk > 0).mean(axis=0)).astype(int)
cy = int(np.clip(cy, HALF, IMG - HALF)); cx = int(np.clip(cx, HALF, IMG - HALF))
sl = (slice(cy - HALF, cy + HALF), slice(cx - HALF, cx + HALF))

for ax, r in zip(axes[1], RES):
    show(ax, degrade(img, r)[sl], msk[sl], lw=1.1, up=4)
    ax.set_title(f"{BASE/r:.2f} mm".replace("1.50", "1.5").replace("2.00", "2.0")
                 .replace("3.00", "3.0"), color=INK, pad=3)

axes[0, 0].set_ylabel("(a)", rotation=0, labelpad=12, color=INK,
                      fontsize=9, va="center")
axes[1, 0].set_ylabel("(b)", rotation=0, labelpad=12, color=INK,
                      fontsize=9, va="center")

fig.tight_layout(h_pad=0.9, w_pad=0.4)
(ROOT / "figures").mkdir(exist_ok=True)
fig.savefig(ROOT / "figures" / "fig_qualitative.pdf")
print("figure written")
for s in ROW1:
    print(f"  {s:<6} {meta[s][PICK[s]]['key']:<18} A={meta[s][PICK[s]]['area']}")
