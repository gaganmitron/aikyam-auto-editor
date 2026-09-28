"""EXP-028 / E1 analysis: within-video Spearman of DOVER (technical, aesthetic, fused) vs the 1-5 clip ratings (tools/bench_gate_ratings.json), next to the signals we already have, on the SAME windows
(those whose source video is still on disk). usage: python tools/e1_score.py [dover|mobile ...]"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
R = json.load(open("tools/bench_gate_ratings.json")); W = json.load(open("results/bench_gate/windows_usable.json")); S = json.load(open("results/bench_gate/signals.json"))
sig = {w["id"]: s for w, s in zip(W, S)}; names = sys.argv[1:] or ["dover", "mobile"]
sc = {n: json.load(open(f"results/e1/dover_{n}.json")) for n in names if os.path.exists(f"results/e1/dover_{n}.json")}
ids = sorted(set.intersection(*[set(v) for v in sc.values()])) if sc else []
vid = {i: i.split("_")[0] for i in ids}; y = {i: R[i] for i in ids}
cands = {}
for n, d in sc.items():
    for k in ("technical", "aesthetic", "fused"): cands[f"DOVER{'-Mobile' if n == 'mobile' else ''} {k}"] = {i: d[i][k] for i in ids}
for k in ("pipeline quality (mean slot q)", "min slot q", "luma (min)", "sharpness (log median)", "SigLIP aesthetic"): cands[f"existing: {k}"] = {i: sig[i][k] for i in ids}
print(f"{len(ids)} windows: " + ", ".join(f"{v} n={sum(1 for i in ids if vid[i] == v)}" for v in sorted(set(vid.values()))) + f"; ratings {sorted(set(y.values()))}\n")
print(f"{'signal':42s} " + " ".join(f"{v:>9s}" for v in sorted(set(vid.values()))) + "     mean rho   AUC(bad<=2 vs ok)")
for name, d in cands.items():
    rho = []; 
    for v in sorted(set(vid.values())):
        m = [i for i in ids if vid[i] == v]
        rho.append(spearmanr([d[i] for i in m], [y[i] for i in m])[0] if len({y[i] for i in m}) > 1 else float("nan"))
    bad = np.array([y[i] <= 2 for i in ids]); x = np.array([d[i] for i in ids]); pos, neg = x[~bad], x[bad]
    auc = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg]) if len(pos) and len(neg) else float("nan")
    print(f"{name:42s} " + " ".join(f"{r:+9.2f}" for r in rho) + f"     {np.nanmean(rho):+.2f}      {auc:.2f}")

# ---- uncertainty: bootstrap the windows inside each video (2000 resamples): mean within-video rho, and the paired DIFFERENCE between each DOVER score and our current pipeline quality
rng = np.random.RandomState(28); vids = sorted(set(vid.values())); byv = {v: [i for i in ids if vid[i] == v] for v in vids}


def boot(f, g=None, n=2000):
    out = []
    for _ in range(n):
        r = []
        for v in vids:
            m = [byv[v][k] for k in rng.randint(0, len(byv[v]), len(byv[v]))]
            yy = [y[i] for i in m]
            if len(set(yy)) < 2: continue
            a = spearmanr([f[i] for i in m], yy)[0]; b = spearmanr([g[i] for i in m], yy)[0] if g else 0.0
            if not (np.isnan(a) or np.isnan(b)): r.append(a - b)
        if r: out.append(np.mean(r))
    return np.percentile(out, [2.5, 50, 97.5])


base = cands["existing: pipeline quality (mean slot q)"]
print("\nbootstrap 95% interval of the mean within-video rho, and of the improvement over the current pipeline quality measure:")
for name in [k for k in cands if k.startswith("DOVER")]:
    lo, md, hi = boot(cands[name]); dlo, dmd, dhi = boot(cands[name], base)
    print(f"  {name:28s} rho {md:+.2f} [{lo:+.2f}, {hi:+.2f}]   improvement over current quality {dmd:+.2f} [{dlo:+.2f}, {dhi:+.2f}]" + ("   <- interval excludes 0" if dlo > 0 else ""))
