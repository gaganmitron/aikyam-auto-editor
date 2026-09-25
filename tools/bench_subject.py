"""EXP-023: where is the subject horizontally? Ground truth `tools/bench_subject_truth.json` = the x-range (fraction of width) a viewer must see, drawn by eye on 22 frames
(12 with a compact subject, 10 wide scenes with none). A locator predicts a window centre; a window of width W there is scored by how much of the subject it holds:
coverage = |window & subject| / min(|subject|, W)   (1.0 = as much of the subject as any window of that width could hold).
Methods: centre crop, the pipeline's tracker (reframe.track_subject_ex), SigLIP crop scoring (devotional relevance of each candidate window), OWL-ViT boxes for a prompt list.
usage: python tools/bench_subject.py [--owl]"""
import json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = json.load(open("tools/bench_subject_truth.json")); D = "results/bench_subject"
WS = (0.45, 0.60)
QUERIES = ["a temple tower gopuram", "a deity idol", "a person", "a group of people", "a flame lamp", "a palanquin", "a golden temple building"]


def cov(c, W, sub):
    a, b = min(max(c - W / 2, 0.0), 1 - W), min(max(c - W / 2, 0.0), 1 - W) + W
    return max(0.0, min(b, sub[1]) - max(a, sub[0])) / min(sub[1] - sub[0], W)


def siglip_center(img, W=0.5, step=0.05):
    from aikyam_video.vision import ClipVision
    from aikyam_video.scoring import DEVOTIONAL
    global _cv
    _cv = globals().get("_cv") or ClipVision(); h, w = img.shape[:2]; best, bc = -1, 0.5
    for x0 in np.arange(0, 1 - W + 1e-6, step):
        v = _cv.analyze(img[:, int(x0 * w):int((x0 + W) * w)]); s = max([DEVOTIONAL.get(k, 0.0) * p for k, p in v.labels.items()] or [0.0])
        if s > best + 1e-9: best, bc = s, x0 + W / 2
    return bc


def owl_center(img):
    import torch
    from transformers import OwlViTProcessor, OwlViTForObjectDetection
    global _owl
    if "_owl" not in globals() or _owl is None:
        _owl = (OwlViTProcessor.from_pretrained("google/owlvit-base-patch32"), OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").eval())
    proc, m = _owl; h, w = img.shape[:2]
    inp = proc(text=[QUERIES], images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt")
    with torch.no_grad(): out = m(**inp)
    r = proc.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[h, w]]))[0]
    if not len(r["scores"]): return 0.5
    b = r["boxes"].numpy(); s = r["scores"].numpy(); cx = (b[:, 0] + b[:, 2]) / 2 / w; ar = (b[:, 2] - b[:, 0]) / w
    wt = s * np.clip(1 - ar, 0.05, 1)            # a box spanning the whole frame says nothing about WHERE the subject is
    return float((cx * wt).sum() / wt.sum())


def main():
    from aikyam_video.reframe import track_subject_ex
    use_owl = "--owl" in sys.argv; res = {"centre": [], "tracker": [], "siglip crops": []}
    if use_owl: res["owlvit"] = []
    for it in T:
        if it["subject"] is None: continue
        img = cv2.imread(f"{D}/f{it['n']:02d}.png"); sub = it["subject"]; centres = {"centre": 0.5}
        tr = track_subject_ex(it["path"], it["t"] - 1.0, it["t"] + 1.0); centres["tracker"] = float(np.median([x for _, x, _ in tr])) if tr else 0.5
        centres["siglip crops"] = siglip_center(img)
        if use_owl: centres["owlvit"] = owl_center(img)
        for k, c in centres.items(): res[k].append((it["n"], c, [cov(c, W, sub) for W in WS], abs(c - (sub[0] + sub[1]) / 2)))
        print(it["n"], sub, {k: round(v, 2) for k, v in centres.items()}, flush=True)
    print(f"\n{len(res['centre'])} compact-subject frames.   method            coverage@W=" + " / ".join(map(str, WS)) + "    mean centre error")
    for k, rows in res.items():
        cv_ = np.mean([r[2] for r in rows], axis=0); print(f"  {k:16s} {cv_[0]:.2f} / {cv_[1]:.2f}          {np.mean([r[3] for r in rows]):.2f}")
    json.dump({k: [(n, c) for n, c, _, _ in v] for k, v in res.items()}, open(f"{D}/subject_results.json", "w"))


if __name__ == "__main__":
    main()
