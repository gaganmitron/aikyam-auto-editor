"""EXP-018: how well does zero-shot story-beat classification work (beats.py) on the benchmark frames? Compared with the coarse 8-way scene-type baseline (0.69).
usage: python tools/bench_beats.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import beats
from aikyam_video.vision import ClipVision

OUT = "results/bench_labels"
names = [os.path.basename(f)[:-4] for f in json.load(open(f"{OUT}/frames.json"))]; E = np.load(f"{OUT}/emb.npz")["E"]
truth = [t for k, t in json.load(open("tools/bench_beats_truth.json")).items() if not k.startswith("_")]
truth = [json.load(open("tools/bench_beats_truth.json"))[n] for n in names]
cv = ClipVision(); T, owner = beats.text_embeddings(cv.embed_text); scale = float(cv.model.logit_scale.exp())
P = [beats.score(e, T, owner, scale) for e in E]; pred = [max(p, key=p.get) for p in P]
acc = np.mean([a == b for a, b in zip(pred, truth)]); top2 = np.mean([t in sorted(p, key=p.get)[-2:] for p, t in zip(P, truth)])
print(f"beat accuracy {acc:.2f}  (top-2 {top2:.2f}) on {len(names)} frames, {len(beats.KEYS)} beat types\n")
print("truth counts:", {k: truth.count(k) for k in beats.KEYS if truth.count(k)})
print("wrong (frame: truth -> predicted, confidence):")
for n, t, p, d in zip(names, truth, pred, P):
    if t != p: print(f"   {n:14s} {t:14s} -> {p:14s} {d[p]:.2f}   (truth got {d[t]:.2f})")
per = {k: (sum(1 for t, p in zip(truth, pred) if t == k and p == k), truth.count(k)) for k in beats.KEYS if truth.count(k)}
print("\nper beat (correct / true):", per)
