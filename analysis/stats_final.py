"""Every number reported in the paper comes from this script.
Each case was evaluated with two seeds; to avoid pseudo-replication the seeds
are averaged per case first, and statistics are computed over cases
(n = 64 NACT, n = 290 DUKE). Output: results/numbers.json."""
import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from scipy import stats

res = pd.read_csv(ROOT / "results" / "ressweep_percase.csv")
mag = pd.read_csv(ROOT / "results" / "magsweep_percase.csv")
OUT = {}

def boot(v, n=10000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    bs = np.random.default_rng(seed).choice(v, (n, len(v)), replace=True).mean(1)
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]

# ================= 1) RESOLUTION =================
r = (res.groupby(["held_out", "case", "spacing_mm"], as_index=False)
        [["dice", "assd_mm", "gt_vox", "sharp"]].mean())
base = r[r.spacing_mm == 1.5].set_index(["held_out", "case"])
r = r.join(base[["dice", "assd_mm", "gt_vox"]].rename(columns=lambda c: c + "_1"),
           on=["held_out", "case"])
r["dd"] = r.dice - r.dice_1
OUT["res"] = {}
for ho in ("NACT", "DUKE"):
    s = r[r.held_out == ho]
    o = {"n_cases": int(s.case.nunique()), "levels": {}}
    for sp in sorted(s.spacing_mm.unique()):
        t = s[s.spacing_mm == sp]
        e = {"dice": float(t.dice.mean()), "assd": float(t.assd_mm.mean())}
        if sp != 1.5:
            e["dd"] = float(t.dd.mean()); e["ci"] = boot(t.dd)
            e["p"] = float(stats.wilcoxon(t.dice, t.dice_1).pvalue)
            e["worse"] = float((t.dd < 0).mean())
        o["levels"][f"{sp:.2f}"] = e
    t = s[s.spacing_mm == s.spacing_mm.max()].dropna(subset=["dd"])
    q = pd.qcut(t.gt_vox_1, 4, labels=False)
    o["quart"] = [float(t.dd[q == i].mean()) for i in range(4)]
    rho, p = stats.spearmanr(t.gt_vox_1, t.dd)
    o["size_rho"], o["size_p"] = float(rho), float(p)
    OUT["res"][ho] = o
OUT["res"]["sharp"] = [float(x) for x in
                       r.groupby("spacing_mm").sharp.mean().sort_index().values]
# validity: reference masks unchanged across levels
OUT["res"]["gt_invariant"] = bool(
    (res.groupby(["held_out", "seed", "case"]).gt_vox.nunique() == 1).all())

# ================= 2) TRUNCATION (DUKE, k=2) =================
m = (mag.groupby(["held_out", "case", "k"], as_index=False)
        [["dice", "assd_px", "assd_mm", "gt_vox"]].mean())
b1 = m[m.k == 1.0].set_index(["held_out", "case"])
m = m.join(b1[["dice", "assd_px", "assd_mm", "gt_vox"]].rename(columns=lambda c: c + "_1"),
           on=["held_out", "case"])
m["keep"] = m.gt_vox / m.k ** 2 / m.gt_vox_1
m["dd"] = m.dice - m.dice_1
d = m[(m.held_out == "DUKE") & (m.k == 2.0)].dropna(subset=["keep", "dd"])
rho, p = stats.spearmanr(d.keep, d.dd)
bins = []
for lo, hi in [(0, .25), (.25, .75), (.75, .95), (.95, 9)]:
    q = d[(d.keep >= lo) & (d.keep < hi)]
    bins.append({"lo": lo, "hi": hi, "dd": float(q.dd.mean()), "n": int(len(q))})
OUT["fov"] = {"n": int(len(d)), "rho": float(rho), "p": float(p),
              "frac_half_lost": float((d.keep < 0.5).mean()), "bins": bins}

# ================= 3) APPARENT SCALE (untruncated cases) =================
clean = m.groupby(["held_out", "case"]).keep.min() >= 0.90
ok = set(clean[clean].index)
c = m[[(h, cs) in ok for h, cs in zip(m.held_out, m.case)]]
OUT["scale"] = {}
for ho in ("NACT", "DUKE"):
    s = c[c.held_out == ho]
    n_all = m[(m.held_out == ho) & (m.k == 1.0)].shape[0]
    o = {"n_clean": int(s.case.nunique()), "n_all": int(n_all), "k": {}}
    for k in (0.5, 1.0, 2.0):
        t = s[s.k == k]
        e = {"dice": float(t.dice.mean()), "px": float(t.assd_px.mean()),
             "mm": float(t.assd_mm.mean())}
        if k != 1.0:
            rp = (t.assd_px / t.assd_px_1).dropna(); rm = (t.assd_mm / t.assd_mm_1).dropna()
            e["ratio_px_med"] = float(rp.median()); e["ratio_mm_med"] = float(rm.median())
            e["p_px"] = float(stats.wilcoxon(t.assd_px, t.assd_px_1, nan_policy="omit").pvalue)
            e["p_mm"] = float(stats.wilcoxon(t.assd_mm, t.assd_mm_1, nan_policy="omit").pvalue)
        o["k"][str(k)] = e
    OUT["scale"][ho] = o

json.dump(OUT, open(ROOT / "results" / "numbers.json", "w"), indent=1)

# ---------- print ----------
for ho in ("NACT", "DUKE"):
    o = OUT["res"][ho]
    print(f"\nRESOLUTION {ho} (n={o['n_cases']} cases)")
    for sp, e in o["levels"].items():
        x = f"  {sp} mm  Dice {e['dice']:.3f}  ASSD {e['assd']:.2f}"
        if "dd" in e:
            x += (f"  dd {e['dd']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
                  f"  p={e['p']:.1e}  worse={e['worse']:.0%}")
        print(x)
    print(f"  quartile dd: {[round(v,3) for v in o['quart']]}  rho={o['size_rho']:+.3f} p={o['size_p']:.1e}")
print("  sharpness:", [round(v, 3) for v in OUT["res"]["sharp"]],
      " GT invariant:", OUT["res"]["gt_invariant"])
f = OUT["fov"]
print(f"\nFOV DUKE k=2: n={f['n']} cases, rho={f['rho']:+.3f} p={f['p']:.1e}, "
      f"more than half lost {f['frac_half_lost']:.1%}")
for b_ in f["bins"]:
    print(f"   {b_['lo']}-{b_['hi']}: dd {b_['dd']:+.3f}  n={b_['n']}")
for ho in ("NACT", "DUKE"):
    o = OUT["scale"][ho]
    print(f"\nSCALE {ho}: untruncated {o['n_clean']}/{o['n_all']}")
    for k, e in o["k"].items():
        x = f"  k={k}: Dice {e['dice']:.3f}  px {e['px']:.2f}  mm {e['mm']:.2f}"
        if "ratio_px_med" in e:
            x += (f"  | ratio px {e['ratio_px_med']:.2f} (p={e['p_px']:.1e})"
                  f"  mm {e['ratio_mm_med']:.2f} (p={e['p_mm']:.2g})")
        print(x)
