"""E3 (subject locator), extra candidate: Google MediaPipe Face Detection (Apache-2.0, pip-installable, CPU) on the SAME 22-frame truth set as
tools/bench_subject.py, so it is directly comparable to the centre-crop / pipeline-tracker / SigLIP-crop / OWL-ViT numbers already in EXP-023.
Only meaningful on the 12 compact-subject frames; a face-only detector says nothing about a gopuram or a lamp (reported anyway, honestly).
usage: python tools/e3_mediapipe.py"""
import json
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

T = json.load(open("tools/bench_subject_truth.json")); D = "results/bench_subject"
det = mp_vision.FaceDetector.create_from_options(mp_vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path="results/mp_models/blaze_face_short_range.tflite"), min_detection_confidence=0.3))


def center(img):
    mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = det.detect(mpi)
    if not res.detections:
        return 0.5, 0.0
    h, w = img.shape[:2]
    best = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bb = best.bounding_box
    return float(np.clip((bb.origin_x + bb.width / 2) / w, 0, 1)), float(bb.width / w)


def raw(c, W, sub):
    a = min(max(c - W / 2, 0), 1 - W); b = a + W
    return max(0.0, min(b, sub[1]) - max(a, sub[0])) / (sub[1] - sub[0])


def main():
    hit = miss = 0; rows = []
    for it in T:
        img = cv2.imread(f"{D}/f{it['n']:02d}.png"); c, w = center(img)
        rows.append((it["n"], it["subject"], c, w > 0))
        if it["subject"] is not None: hit += w > 0; miss += w == 0
    print(f"face found on {hit} of {hit + miss} compact-subject frames (a person is not always the subject: a gopuram, a lamp, an idol have no face)")
    for W in (0.45, 0.60):
        r = [raw(c, W, sub) for n, sub, c, found in rows if sub is not None]
        full = np.mean([x >= 0.97 for x in r])
        print(f"  window {W:.2f}: mean fraction of subject held {np.mean(r):.2f}, fully inside {full:.2f}  (centre-crop from EXP-023: {'0.75/0.17' if W == 0.45 else '0.88/0.50'}; tracker: {'0.77/0.42' if W == 0.45 else '0.87/0.50'})")
    only_faces = [(raw(c, 0.45, sub), raw(c, 0.60, sub)) for n, sub, c, found in rows if sub is not None and found]
    if only_faces:
        a45, a60 = zip(*only_faces)
        print(f"  ON THE {len(only_faces)} FRAMES WHERE A FACE WAS FOUND ONLY: 0.45 -> {np.mean(a45):.2f}, 0.60 -> {np.mean(a60):.2f}")


if __name__ == "__main__":
    main()
