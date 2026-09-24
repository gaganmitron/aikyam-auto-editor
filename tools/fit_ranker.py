"""EXP-016: fit the learned clip ranker (aikyam_video/ranker.py) on the rated benchmark frames and VALIDATE it before anything ships.
Features come from ClipVision.analyze -- exactly what the pipeline stores per scene. Validation: hold out a whole video, fit on the others, and score
the within-video Spearman rank correlation with the rating (LOVO, strict) and one frame at a time (LOO, optimistic: neighbours of the same video leak).
usage: python tools/fit_ranker.py [--write]     (--write saves aikyam_video/data/ranker.json fitted on ALL frames)"""
import json, os, sys
import cv2, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import ranker
from aikyam_video.vision import ClipVision

OUT = "results/bench_labels"
frames = json.load(open(f"{OUT}/frames.json")); names = [os.path.basename(f)[:-4] for f in frames]
y = np.array([json.load(open("tools/bench_clips_ratings.json"))[n] for n in names], float); src = np.array([n.split("_")[0] for n in names])
cache = f"{OUT}/rank_features.npz"
if os.path.exists(cache) and len(np.load(cache)["X"]) == len(frames):
    X = np.load(cache)["X"]
else:
    cv = ClipVision(); X = np.array([ranker.featurize(cv.analyze(cv2.imread(f))) for f in frames]); np.savez(cache, X=X)
print(f"{len(frames)} frames, features {ranker.FEATURES}\n")

prior = np.array([ranker.PRIOR[f] for f in ranker.FEATURES])
Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
baselines = {"prior weights (no data)": lambda tr, te: Z[te] @ prior,
             "aesthetic only": lambda tr, te: Z[te][:, 0],
             "devotional only (pipeline signal)": lambda tr, te: Z[te][:, 2]}


def within_video_rho(pred_by_frame):
    out = {}
    for s in sorted(set(src)):
        m = src == s
        if len(set(y[m])) > 1 and len(set(np.round(pred_by_frame[m], 9))) > 1:
            out[s] = float(spearmanr(pred_by_frame[m], y[m])[0])
    return out


def lovo(predict):
    pred = np.zeros(len(y))
    for s in sorted(set(src)):
        te = np.where(src == s)[0]; tr = np.where(src != s)[0]; pred[te] = predict(tr, te)
    return within_video_rho(pred)


def loo(predict):
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.array([j for j in range(len(y)) if j != i]); pred[i] = predict(tr, np.array([i]))[0]
    return within_video_rho(pred)


def learned(lam):
    def p(tr, te):
        rk = ranker.fit(X[tr], y[tr], lam=lam); return np.array([rk.raw(X[i]) for i in te])
    return p


methods = dict(baselines); [methods.__setitem__(f"learned, lam={l:g}", learned(l)) for l in (1, 4, 16, 64)]
print(f"{'method':38s}  LOVO rho per video (golden / janakpur / kolkata)   mean   |  LOO mean")
best = None
for name, fn in methods.items():
    a, b = lovo(fn), loo(fn)
    ma, mb = np.mean(list(a.values())), np.mean(list(b.values()))
    print(f"{name:38s}  " + " / ".join(f"{a.get(s, float('nan')):+.2f}" for s in ("golden", "janakpur", "kolkata")) + f"   {ma:+.2f}   |  {mb:+.2f}")
    if name.startswith("learned") and (best is None or ma > best[0]): best = (ma, name)
print(f"\nbest learned variant by LOVO: {best[1]} ({best[0]:+.2f}); frames per video:", {s: int((src == s).sum()) for s in sorted(set(src))})
rk_all = ranker.fit(X, y, lam=float(best[1].split('=')[1]))
print("weights fitted on all frames (standardised):", {f: round(w, 2) for f, w in zip(ranker.FEATURES, rk_all.w)}, " prior:", ranker.PRIOR)
if "--write" in sys.argv:
    from aikyam_video.vision import model_name
    rk_all.meta.update({"source": "tools/fit_ranker.py on tools/bench_clips_ratings.json", "lovo_mean": round(best[0], 3), "vision_model": model_name()}); rk_all.save(); print("wrote", ranker.PATH)
