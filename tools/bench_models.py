"""EXP-015: which image-text model should the pipeline use? Same 35-frame benchmark as bench_labels / bench_clips / bench_vlm, per model:
  within-video AP (labels)  |  best single-threshold F1  |  8-way scene-type accuracy  |  aesthetic-prior rho vs reel-worthiness  |  seconds per frame on this CPU
usage: python tools/bench_models.py ViT-B-16-SigLIP ViT-B-16-SigLIP-256 ...      (results/bench_labels/model_<name>.json is cached; delete to redo)"""
import json, os, sys, time
import cv2, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.vision import ClipVision, LABEL_PROMPTS, AESTHETIC_PAIRS

OUT = "results/bench_labels"
frames = json.load(open(f"{OUT}/frames.json")); names = [os.path.basename(f)[:-4] for f in frames]
truth = {k: set(v) for k, v in json.load(open("tools/bench_labels_truth.json")).items() if not k.startswith("_")}
rating = np.array([json.load(open("tools/bench_clips_ratings.json"))[n] for n in names], float)
src = np.array([n.split("_")[0] for n in names]); keys = list(LABEL_PROMPTS)
TYPES = {"A": "a priest waving a burning aarti lamp at night", "B": "a decorated deity idol in a shrine", "C": "a temple building seen from outside, across a lake or at night",
         "D": "devotees praying with folded hands in a crowd", "E": "people carrying a decorated palanquin or canopy in a procession",
         "F": "attendants performing a ritual inside a temple sanctum", "G": "people sitting in rows being served food", "H": "a title card, text or an unrelated close-up"}


def scene_type(labels):
    L = set(labels)
    if not L: return "H"
    if "aarti" in L or ("lamps" in L and "priest" in L and "deity" not in L): return "A"
    if "deity" in L or "idol" in L: return "B"
    if "food_service" in L: return "G"
    if "procession" in L: return "E"
    if L & {"temple_architecture", "temple_tank", "temple_night"}: return "C"
    if "priest" in L: return "F"
    if "devotees" in L or "crowd" in L: return "D"
    return "H"


def evaluate(model):
    cv = ClipVision(model=f"hf-hub:timm/{model}"); g = cv.labels
    scale, bias = float(cv.model.logit_scale.exp()), float(cv.model.logit_bias)
    imgs = [cv2.imread(f) for f in frames]; cv.embed_image(imgs[0])                       # warm-up (first call pays one-off costs)
    t0 = time.time(); E = np.array([cv.embed_image(im) for im in imgs]); sec = (time.time() - t0) / len(imgs)
    Z = np.array([[ (scale * (g.emb @ e) + bias)[g.owner == i].max() for i in range(len(keys))] for e in E]); P = 1 / (1 + np.exp(-Z))
    aps = []                                                                              # within-video AP over (video, label) pairs with variation
    for s in sorted(set(src)):
        idx = np.where(src == s)[0]
        for j, k in enumerate(keys):
            y = np.array([k in truth[names[i]] for i in idx])
            if 1 <= y.sum() < len(y):
                order = np.argsort(-P[idx, j]); hits = np.where(y[order])[0]; aps.append(np.mean([(r + 1) / (h + 1) for r, h in enumerate(hits)]))
    best = 0.0
    for t in np.unique(np.round(P.ravel(), 2)):
        tp = fp = fn = 0
        for i, n in enumerate(names):
            pred = {keys[j] for j in np.where(P[i] >= t)[0]}; tp += len(pred & truth[n]); fp += len(pred - truth[n]); fn += len(truth[n] - pred)
        best = max(best, 2 * tp / max(2 * tp + fp + fn, 1))
    T = cv.embed_text(list(TYPES.values())); pred = [list(TYPES)[int(np.argmax(e @ T.T))] for e in E]
    acc = float(np.mean([p == scene_type(truth[n]) for p, n in zip(pred, names)]))
    tx = cv.embed_text([t for pair in AESTHETIC_PAIRS for t in pair]); lg = scale * (E @ tx.T); aes = np.mean(lg[:, 0::2] - lg[:, 1::2], axis=1)
    g_ = src == "golden"
    r = {"model": model, "within_video_AP": float(np.mean(aps)), "n_pairs": len(aps), "best_F1": best, "scene_type_acc": acc, "aesthetic_rho": float(spearmanr(aes, rating)[0]),
         "aesthetic_rho_golden": float(spearmanr(aes[g_], rating[g_])[0]), "sec_per_frame": sec, "params_M": sum(p.numel() for p in cv.model.parameters()) / 1e6}
    return r


rows = []
for m in sys.argv[1:]:
    cache = f"{OUT}/model_{m}.json"
    if not os.path.exists(cache):
        try:
            json.dump(evaluate(m), open(cache, "w"))
        except Exception as e:                                                             # noqa: BLE001 -- a model that will not load (RAM, missing weights) is a result too
            json.dump({"model": m, "error": f"{type(e).__name__}: {str(e)[:120]}"}, open(cache, "w"))
    rows.append(json.load(open(cache)))
print(f"{'model':28s} {'params':>7s} {'AP':>6s} {'F1':>5s} {'scene acc':>9s} {'aes rho':>8s} {'aes rho (golden)':>16s} {'s/frame':>8s}")
for r in rows:
    if "error" in r: print(f"{r['model']:28s} FAILED: {r['error']}"); continue
    print(f"{r['model']:28s} {r['params_M']:6.0f}M {r['within_video_AP']:6.3f} {r['best_F1']:5.2f} {r['scene_type_acc']:9.2f} {r['aesthetic_rho']:+8.2f} {r['aesthetic_rho_golden']:+16.2f} {r['sec_per_frame']:8.2f}")
