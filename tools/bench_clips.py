"""Which cheap signals predict reel-worthiness? Spearman vs tools/bench_clips_ratings.json on the bench_labels frames.
Needs results/bench_labels/frames.json (run tools/bench_labels.py first). usage: python tools/bench_clips.py"""
import json, os, sys
import cv2, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.vision import ClipVision, HeuristicVision, LABEL_PROMPTS
from aikyam_video.scoring import DEVOTIONAL

OUT = "results/bench_labels"
frames = json.load(open(f"{OUT}/frames.json"))
names = [os.path.basename(f)[:-4] for f in frames]
R = {k: v for k, v in json.load(open("tools/bench_clips_ratings.json")).items() if not k.startswith("_")}
y = np.array([R[n] for n in names], float); src = np.array([n.split("_")[0] for n in names])
imgs = [cv2.imread(f) for f in frames]

PAIRS = {  # CLIP-IQA style antonym pairs: score = logit(positive) - logit(negative)
    "iqa quality": ("a high quality, beautiful, well composed photo", "a low quality, ugly, badly composed photo"),
    "iqa sharp": ("a sharp, detailed photo", "a blurry, out of focus photo"),
    "iqa bright": ("a bright, well lit photo", "a dark, underexposed photo"),
    "iqa colourful": ("a vivid, colourful photo", "a dull, washed out photo"),
    "iqa cinematic": ("a striking cinematic photo of a temple ritual", "a boring, cluttered snapshot"),
    "iqa clean frame": ("a photo with a clear single subject", "a photo of a messy crowded scene"),
    "iqa no text": ("a photo with no text", "a screenshot with subtitles and text overlay"),
}
cache = f"{OUT}/emb.npz"
if os.path.exists(cache) and len(np.load(cache)["E"]) == len(frames):
    E = np.load(cache)["E"]
    cv = None
else:
    cv = ClipVision(); E = np.array([cv.embed_image(im) for im in imgs]); np.savez(cache, E=E)
cv = cv or ClipVision()
scale, bias = float(cv.model.logit_scale.exp()), float(cv.model.logit_bias)
S = {}
for name, (pos, neg) in PAIRS.items():
    t = cv.embed_text([pos, neg]); S[name] = (scale * (E @ t[0]) - scale * (E @ t[1]))

heur = HeuristicVision(); H = [heur.analyze(im) for im in imgs]
S["sharpness"] = np.array([h.sharpness for h in H])
S["exposure (not too dark/bright)"] = np.array([1 - abs(h.brightness - 0.5) * 2 for h in H])
S["colourfulness"] = np.array([float(np.std(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1])) for im in imgs])
S["face area (pipeline)"] = np.array([min(1.0, sum(f[2] * f[3] for f in h.faces) * 8) for h in H])
Zc = np.load(f"{OUT}/logits.npz")["Z"]; keys = list(np.load(f"{OUT}/logits.npz")["keys"]); P = 1 / (1 + np.exp(-Zc))
S["devotional relevance (pipeline)"] = np.array([max(DEVOTIONAL.get(k, 0) * P[i, j] for j, k in enumerate(keys) if P[i, j] >= 0.2) if (P[i] >= 0.2).any() else 0.0 for i in range(len(frames))])
vis = np.array([0.4 * min(1.0, sum(f[2] * f[3] for f in h.faces) * 8) + 0.25 * P[i, keys.index("lamps")] + 0.35 * max(0.0, h.sharpness * (1 - abs(h.brightness - 0.5) * 1.2)) for i, h in enumerate(H)])
S["visual importance (pipeline)"] = vis
S["pipeline visual+devotional (0.25/0.20)"] = 0.25 * vis + 0.20 * S["devotional relevance (pipeline)"]

base = (y >= 4).mean()
print(f"{len(frames)} frames; base rate of rating>=4: {base:.2f}\n{'signal':42s} rho(all)  rho(golden)  precision@8 (>=4)")
rows = []
for name, s in S.items():
    rho = spearmanr(s, y)[0]; g = src == "golden"; rg = spearmanr(s[g], y[g])[0]
    top = np.argsort(-s)[:8]; rows.append((rho, name, rg, (y[top] >= 4).mean()))
for rho, name, rg, p8 in sorted(rows, reverse=True):
    print(f"{name:42s} {rho:+.2f}    {rg:+.2f}       {p8:.2f}")
