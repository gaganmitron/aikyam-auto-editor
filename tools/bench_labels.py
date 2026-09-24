"""Scored label benchmark: how well do different ways of turning SigLIP logits into label confidences rank the labels
that are really in a frame? Ground truth: tools/bench_labels_truth.json. Frames: results/bench_labels/<source>_<sec>.jpg
(made by the extraction block below from the three source videos in inputs/).
usage: python tools/bench_labels.py"""
import json, os, subprocess, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.vision import ClipVision, LABEL_PROMPTS, KEEP

OUT = "results/bench_labels"
JOBS = [("golden", "inputs/golden_temple_full.webm", [15 + 30 * i for i in range(22)]),
        ("janakpur", "inputs/bench_ganga_aarti_janakpur.webm", [3, 10, 17, 24, 31, 38, 45]),
        ("kolkata", "inputs/bench_temple_aarti_kolkata.webm", [2, 7, 12, 17, 22, 27])]
truth = {k: set(v) for k, v in json.load(open("tools/bench_labels_truth.json")).items() if not k.startswith("_")}
os.makedirs(OUT, exist_ok=True)
frames = []
for name, path, ts in JOBS:
    for t in ts:
        fn = f"{OUT}/{name}_{t:03d}.jpg"
        if not os.path.exists(fn):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", path, "-frames:v", "1", "-vf",
                            "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2", "-q:v", "3", fn], check=True)
        frames.append((name, os.path.basename(fn)[:-4], fn))

keys = list(LABEL_PROMPTS)
cache = f"{OUT}/logits.npz"                                                       # delete after changing LABEL_PROMPTS / the model
if os.path.exists(cache) and list(np.load(cache)["keys"]) == keys and len(np.load(cache)["Z"]) == len(frames):
    d = np.load(cache); Z, ZN = d["Z"], d["ZN"]
else:
    cv = ClipVision(); g = cv.labels
    scale, bias = float(cv.model.logit_scale.exp()), float(cv.model.logit_bias)
    Z, ZN = [], []
    for _, _, fn in frames:
        e = cv.embed_image(cv2.imread(fn))
        zp = scale * (g.emb @ e) + bias
        Z.append([zp[g.owner == i].max() for i in range(len(keys))]); ZN.append(scale * (cv._neg @ e) + bias)
    Z, ZN = np.array(Z), np.array(ZN)
    np.savez(cache, Z=Z, ZN=ZN, keys=np.array(keys))
src = np.array([s for s, _, _ in frames])


def video_relative(Z):
    out = Z.copy()
    for s in set(src):
        m = src == s
        out[m] = Z[m] - np.median(Z[m], axis=0, keepdims=True)
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


sig = lambda z: 1 / (1 + np.exp(-z))
zc = Z - Z.mean(axis=1, keepdims=True)
VR = video_relative(Z)
METHODS = {
    "A raw sigmoid (current)": sig(Z),
    "B softmax vs other labels+negatives": softmax(np.hstack([Z, ZN]))[:, :len(keys)],
    "C frame-centred logit": zc,
    "D video-relative logit": VR,
    "E video-relative + frame-centred": VR - VR.mean(axis=1, keepdims=True),
}


def evaluate(S):
    ap, top1, rec3, n = [], 0, [], 0
    for i, (_, name, _) in enumerate(frames):
        T = truth[name]
        if not T:
            continue
        order = [keys[j] for j in np.argsort(-S[i])]
        hits = [r + 1 for r, k in enumerate(order) if k in T]
        ap.append(np.mean([(h_i + 1) / h for h_i, h in enumerate(hits)])); top1 += order[0] in T; n += 1
        rec3.append(len(T & set(order[:3])) / min(3, len(T)))
    return np.mean(ap), top1 / n, np.mean(rec3)


print(f"{len(frames)} frames, {len(keys)} labels\n{'method':40s}  mAP   top1  recall@3")
for name, S in METHODS.items():
    a, t, r = evaluate(S); print(f"{name:40s} {a:.3f}  {t:.2f}  {r:.2f}")

P = METHODS["A raw sigmoid (current)"] >= KEEP                                     # what the pipeline keeps today (threshold KEEP)
tp = fp = fn_ = 0
for i, (_, name, _) in enumerate(frames):
    pred = {keys[j] for j in np.where(P[i])[0]}; T = truth[name]
    tp += len(pred & T); fp += len(pred - T); fn_ += len(T - pred)
print(f"\ncurrent behaviour at KEEP={KEEP}: precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn_, 1):.2f}  ({fp} false labels kept over {len(frames)} frames)")
print("\nper-label (raw sigmoid): times predicted / times true / true-positive")
for j, k in enumerate(keys):
    pr = sum(P[i][j] for i in range(len(frames))); tr = sum(k in truth[n] for _, n, _ in frames); hit = sum(P[i][j] and k in truth[n] for i, (_, n, _) in enumerate(frames))
    print(f"  {k:22s} pred {pr:2d}  true {tr:2d}  tp {hit:2d}")


# ---- what clip selection actually needs: for each label, does the score rank the frames that truly contain it above the rest?
def cross_frame_ap(S):
    aps = {}
    for j, k in enumerate(keys):
        y = np.array([k in truth[n] for _, n, _ in frames])
        if y.sum() < 2 or y.sum() == len(y):
            continue
        order = np.argsort(-S[:, j]); hits = np.where(y[order])[0]
        aps[k] = np.mean([(r + 1) / (h + 1) for r, h in enumerate(hits)])
    return aps


print("\ncross-frame AP per label (does the score rank frames that truly contain the label first?)")
allaps = {m: cross_frame_ap(S) for m, S in METHODS.items()}
common = sorted(set.intersection(*[set(a) for a in allaps.values()]))
print(f"{'label':22s}" + "".join(f"{m[:1]:>7s}" for m in METHODS))
for k in common:
    print(f"{k:22s}" + "".join(f"{allaps[m][k]:7.2f}" for m in METHODS))
print(f"{'MEAN':22s}" + "".join(f"{np.mean([allaps[m][k] for k in common]):7.2f}" for m in METHODS))


# ---- the fair test: clip selection ranks frames WITHIN one video, so score (video, label) pairs separately
def within_video_ap(S):
    aps = {}
    for s in sorted(set(src)):
        idx = np.where(src == s)[0]
        for j, k in enumerate(keys):
            y = np.array([k in truth[frames[i][1]] for i in idx])
            if y.sum() < 1 or y.sum() == len(y):
                continue
            order = np.argsort(-S[idx, j]); hits = np.where(y[order])[0]
            aps[(s, k)] = np.mean([(r + 1) / (h + 1) for r, h in enumerate(hits)])
    return aps


print("\nWITHIN-video AP (video, label) pairs with at least one true and one false frame")
W = {m: within_video_ap(S) for m, S in METHODS.items()}
pairs = sorted(set.intersection(*[set(a) for a in W.values()]))
print(f"{'video/label':30s}" + "".join(f"{m[:1]:>7s}" for m in METHODS))
for p in pairs:
    print(f"{p[0] + '/' + p[1]:30s}" + "".join(f"{W[m][p]:7.2f}" for m in METHODS))
print(f"{'MEAN over ' + str(len(pairs)) + ' pairs':30s}" + "".join(f"{np.mean([W[m][p] for p in pairs]):7.2f}" for m in METHODS))


print("\nglobal threshold sweep on raw sigmoid (what the pipeline keeps): precision / recall / false labels per frame")
for th in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
    Pm = sig(Z) >= th; tp = fp = fn_ = 0
    for i, (_, name, _) in enumerate(frames):
        pred = {keys[j] for j in np.where(Pm[i])[0]}; T = truth[name]
        tp += len(pred & T); fp += len(pred - T); fn_ += len(T - pred)
    print(f"  >= {th:4.2f}: precision {tp / max(tp + fp, 1):.2f}  recall {tp / max(tp + fn_, 1):.2f}  false/frame {fp / len(frames):.1f}")
print("\nbest single threshold per label (raw sigmoid, max F1) -- diagnostic only, 35 frames is too few to ship these")
for j, k in enumerate(keys):
    y = np.array([k in truth[n] for _, n, _ in frames])
    if y.sum() == 0:
        continue
    best = max(((2 * ((sig(Z[:, j]) >= t) & y).sum()) / max((sig(Z[:, j]) >= t).sum() + y.sum(), 1), t) for t in np.unique(np.round(sig(Z[:, j]), 3)))
    print(f"  {k:22s} true {int(y.sum()):2d}  best F1 {best[0]:.2f} at {best[1]:.3f}")


# ---- label-free corpus calibration: standardise each label's logit against its own distribution over ALL frames seen (no human labels)
def zscore(Z):
    return (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-6)


def robust_z(Z):
    med = np.median(Z, axis=0, keepdims=True); mad = np.median(np.abs(Z - med), axis=0, keepdims=True) * 1.4826
    return (Z - med) / (mad + 1e-6)


def best_global_f1(S):
    """best F1 over ONE threshold shared by all labels -- what a pipeline with a single KEEP can do"""
    best = (0, 0)
    for t in np.unique(np.round(S.ravel(), 2)):
        tp = fp = fn_ = 0
        for i, (_, name, _) in enumerate(frames):
            pred = {keys[j] for j in np.where(S[i] >= t)[0]}; T = truth[name]
            tp += len(pred & T); fp += len(pred - T); fn_ += len(T - pred)
        f1 = 2 * tp / max(2 * tp + fp + fn_, 1)
        best = max(best, (f1, t))
    return best


print("\nlabel-free corpus calibration (per-label standardisation over all frames)")
CAL = {"A raw sigmoid (current)": sig(Z), "F corpus z-score": zscore(Z), "G corpus robust z (median/MAD)": robust_z(Z)}
print(f"{'method':34s} within-frame mAP  top1  recall@3 | best single-threshold F1 (threshold) | within-video AP")
for name, S in CAL.items():
    a, t, r = evaluate(S); f1, th = best_global_f1(S); Wn = within_video_ap(S)
    print(f"{name:34s} {a:.3f}          {t:.2f}  {r:.2f}     | {f1:.2f} ({th:.2f})                          | {np.mean([Wn[p] for p in pairs]):.2f}")
