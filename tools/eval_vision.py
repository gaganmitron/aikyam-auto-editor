"""Eyeball harness: per-clip frames + provider labels, plus a contact sheet you can open.
usage: python tools/eval_vision.py heuristic|clip samples/*.webm  -> tools/eval_out/<clip>.jpg + printed labels"""
import os, sys, tempfile
import cv2, numpy as np
from aikyam_video import ffmpeg as ff, providers, vision  # noqa

kind, clips = sys.argv[1], sys.argv[2:]
vp = providers.get("vision", kind)
os.makedirs("tools/eval_out", exist_ok=True)
for c in clips:
    d = ff.probe(c).duration; tiles = []
    print(f"\n== {os.path.basename(c)} ({d:.0f}s)")
    for j in range(6):
        t = d * (j + 0.5) / 6; f = tempfile.mktemp(suffix=".jpg"); ff.frame_at(c, t, f)
        img = cv2.imread(f); r = vp.analyze(img)
        top = sorted(r.labels.items(), key=lambda kv: -kv[1])[:4]
        extra = (f" | deity={r.deities}" if r.deities else "") + (f" | MOD={ {k: round(v, 2) for k, v in r.moderation.items()} }" if r.moderation else "")
        print(f"  t={t:5.1f}s " + ", ".join(f"{k}:{v:.2f}" for k, v in top) + f" | faces={len(r.faces)}" + extra)
        tile = cv2.resize(img, (320, 240)); cv2.putText(tile, f"{top[0][0]}:{top[0][1]:.2f}" if top else "-", (5, 20), 0, .6, (0, 255, 255), 2)
        tiles.append(tile)
    cv2.imwrite(f"tools/eval_out/{os.path.basename(c)}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
