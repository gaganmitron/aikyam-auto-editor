"""Moment-level check: does window scoring (scoring.py) track reel-worthiness? Uses the analysis artifacts of a finished run over the 11-min
Golden Temple video (results/golden_full) and the 22 rated frames (tools/bench_clips_ratings.json, golden_* keys: t = 15 + 30 i).
usage: python tools/bench_scoring.py [artifact_dir]"""
import json, os, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import stages, ffmpeg as ff
from aikyam_video.highlights import build_ctx
from aikyam_video.models import Transcript
from aikyam_video.scoring import SCORERS, DEFAULT_WEIGHTS

art = sys.argv[1] if len(sys.argv) > 1 else "results/golden_full"
R = {k: v for k, v in json.load(open("tools/bench_clips_ratings.json")).items() if k.startswith("golden_")}
ts = sorted(int(k.split("_")[1]) for k in R); y = np.array([R[f"golden_{t:03d}"] for t in ts], float)
src = "inputs/golden_temple_full.webm"
scenes, vis, audio, black = stages._load_analysis(art)
ctx = build_ctx(src, ff.probe(src), scenes, vis, Transcript(**stages._load(art, stages.TRANSCRIPT)), audio, stages._provider("entity", "gazetteer"), black=black)
if any(sv.vision.aesthetic is None for sv in ctx.vision):     # artifacts from before the aesthetic field existed: derive it from the saved scene embeddings (approximation)
    from aikyam_video.vision import ClipVision, AESTHETIC_PAIRS
    cv = ClipVision(); tx = cv.embed_text([t for pair in AESTHETIC_PAIRS for t in pair]); sc = float(cv.model.logit_scale.exp())
    for sv in ctx.vision:
        lg = sc * (tx @ np.asarray(sv.vision.embedding)); sv.vision.aesthetic = float(np.mean(lg[0::2] - lg[1::2]))
scene_at = lambda t: next(sv for sv in ctx.vision if sv.start <= t < sv.end)                # the pipeline scores whole scenes / scene-aligned windows
wins = [(scene_at(t).start, scene_at(t).end) for t in ts]
comp = {k: np.array([SCORERS[k](ctx, a, b) for a, b in wins]) for k in DEFAULT_WEIGHTS}
HALF = float(np.mean([(b - a) / 2 for a, b in wins]))


def combo(w):
    return sum(w[k] * comp[k] for k in w) / sum(w.values())


cfgs = {"default weights (aesthetic 0)": dict(DEFAULT_WEIGHTS),
        "default + aesthetic 0.15": {**DEFAULT_WEIGHTS, "aesthetic": 0.15},
        "default + aesthetic 0.30": {**DEFAULT_WEIGHTS, "aesthetic": 0.30},
        "visual 0.25->0.10, aesthetic 0.15": {**DEFAULT_WEIGHTS, "visualImportance": 0.10, "aesthetic": 0.15},
        "aesthetic only": {"aesthetic": 1.0}}
print(f"{len(ts)} rated windows (scene windows, mean {2*HALF:.0f}s) in {art}\n\ncomponent alone               rho vs rating")
for k in DEFAULT_WEIGHTS:
    print(f"  {k:26s} {spearmanr(comp[k], y)[0]:+.2f}")
print("\nweight configuration                   rho    precision@6 (rating>=4; base %.2f)" % (y >= 4).mean())
for n, w in cfgs.items():
    s = combo(w); top = np.argsort(-s)[:6]
    print(f"  {n:36s} {spearmanr(s, y)[0]:+.2f}   {(y[top] >= 4).mean():.2f}")
