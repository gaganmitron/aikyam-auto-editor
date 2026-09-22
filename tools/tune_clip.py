import os, sys, tempfile, numpy as np, cv2
from aikyam_video import ffmpeg as ff
from aikyam_video.vision import ClipVision, LABEL_PROMPTS
cv = ClipVision()
print(cv.model.__class__.__name__, cv.sigmoid)
clips = {"madan": "samples/madan-mohan-aarti.webm", "durga": "samples/durga-aarti.webm", "sudalai": "samples/sudalai-madan-aarti.webm"}
E = {}
for n, c in clips.items():
    d = ff.probe(c).duration
    E[n] = []
    for j in range(6):
        f = tempfile.mktemp(suffix=".jpg"); ff.frame_at(c, d * (j + .5) / 6, f); E[n].append(cv.embed_image(cv2.imread(f)))
np.save("tools/eval_out/emb.npy", {k: np.array(v) for k, v in E.items()}, allow_pickle=True)
np.save("tools/eval_out/text.npy", {"pos": cv._pos, "neg": cv._neg, "owner": cv._owner, "keys": cv.keys}, allow_pickle=True)
