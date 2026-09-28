"""EXP-021: which measurable signals predict whether a candidate clip is worth putting in a reel? (the clip quality gate)
step 1  python tools/bench_gate.py sample   -> results/bench_gate/windows.json (+ per-window slot features) and sheet_XX.jpg contact sheets to rate by eye
step 2  write tools/bench_gate_ratings.json  {window id: 1..5}  (1 unusable, 3 acceptable filler, 5 a clip you would build a reel around)
step 3  python tools/bench_gate.py fit      -> Spearman (within video, and pooled) of every signal vs the rating, leave-one-video-out
Windows are random 4 s stretches inside one shot (no cut), stratified over each video's length, NOT the clips the pipeline picked: a gate has to reject candidates the ranking might still propose."""
import json, os, random, subprocess, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = "results/bench_gate"; L = 4.0; PER = 14; SEED = 21
SOURCES = {"darshan": "inputs/tirumala_darshan_2021.mp4", "drone": "inputs/tirumala_drone_2025.mp4",
           "tourist": "inputs/tirumala_tourist_2024_trimmed.mp4", "padmavathi": "inputs/tirupati_padmavathi_abhishekam_2020.mp4"}      # updated 2026-09-27 (EXP-034): the current 4-video Tirumala/Tirupati set, golden/janakpur/kolkata were deleted
FEATS = ["sharpness", "jitter", "motion", "luma", "contrast", "clipped", "concentration"]


def frame_at(path, t, w=240):
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", f"scale={w}:-2", "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True)
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def sample():
    from aikyam_video.creative.shots import analyze_window
    from aikyam_video.creative import edges
    os.makedirs(OUT, exist_ok=True); rng = random.Random(SEED); wins = []
    for name, path in SOURCES.items():
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout)
        bin_w = (dur - 10 - L) / PER
        for i in range(PER):
            for _try in range(40):                                                                      # a window must lie inside ONE shot: the pipeline's clips do, and cuts would dominate every motion/jitter feature
                a = round(5 + i * bin_w + rng.random() * (bin_w - 0.1), 1)
                if not edges.detect_cuts(path, a, a + L): break
            sl = analyze_window(path, a, a + L, None)
            wins.append({"id": f"{name}_{a:07.1f}", "video": name, "path": path, "start": a, "end": a + L, **{k: [float(x) for x in getattr(sl, k)] for k in FEATS}})
            print(len(wins), wins[-1]["id"], flush=True)
    order = list(range(len(wins))); rng.shuffle(order); wins = [wins[i] for i in order]          # shuffled: the rater sees no video order
    json.dump(wins, open(f"{OUT}/windows.json", "w"))
    for s in range(0, len(wins), 5):
        rows = []
        for w in wins[s:s + 5]:
            fr = [frame_at(w["path"], w["start"] + f * L) for f in (0.02, 0.35, 0.68, 0.98)]
            h = min(f.shape[0] for f in fr); fr = [cv2.resize(f, (int(f.shape[1] * 200 / f.shape[0]), 200)) for f in fr]
            strip = np.hstack([cv2.resize(f, (356, 200)) for f in fr]); cv2.putText(strip, w["id"].split("_")[1] + f" #{wins.index(w)}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); rows.append(strip)
        cv2.imwrite(f"{OUT}/sheet_{s // 5:02d}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("wrote", len(wins), "windows,", (len(wins) + 4) // 5, "sheets")


def add(only=None, per=18, seed=34):
    """EXP-034: append NEW random windows for `only` (a list of SOURCES keys, default: all) WITHOUT touching existing windows.json entries -- previously
    rated windows keep their id and rating. Writes new_sheet_*.jpg for ONLY the new windows (existing sheets are untouched)."""
    from aikyam_video.creative.shots import analyze_window
    from aikyam_video.creative import edges
    os.makedirs(OUT, exist_ok=True)
    existing = json.load(open(f"{OUT}/windows.json")) if os.path.exists(f"{OUT}/windows.json") else []
    have_ids = {w["id"] for w in existing}
    rng = random.Random(seed); new = []
    for name, path in {k: SOURCES[k] for k in (only or SOURCES)}.items():
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True).stdout)
        bin_w = (dur - 10 - L) / per
        for i in range(per):
            for _try in range(40):
                a = round(5 + i * bin_w + rng.random() * (bin_w - 0.1), 1)
                if not edges.detect_cuts(path, a, a + L): break
            wid = f"{name}_{a:07.1f}"
            if wid in have_ids: continue
            sl = analyze_window(path, a, a + L, None)
            w = {"id": wid, "video": name, "path": path, "start": a, "end": a + L, **{k2: [float(x) for x in getattr(sl, k2)] for k2 in FEATS}}
            new.append(w); have_ids.add(wid); print(len(existing) + len(new), w["id"], flush=True)
    rng.shuffle(new); all_wins = existing + new
    json.dump(all_wins, open(f"{OUT}/windows.json", "w"))
    for s in range(0, len(new), 5):
        rows = []
        for w in new[s:s + 5]:
            fr = [frame_at(w["path"], w["start"] + f * L) for f in (0.02, 0.35, 0.68, 0.98)]
            fr = [cv2.resize(f, (int(f.shape[1] * 200 / f.shape[0]), 200)) for f in fr]
            strip = np.hstack([cv2.resize(f, (356, 200)) for f in fr])
            cv2.putText(strip, w["id"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.putText(strip, f"#{all_wins.index(w)}", (6, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rows.append(strip)
        cv2.imwrite(f"{OUT}/new_sheet_{s // 5:02d}.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f"added {len(new)} new windows, {(len(new) + 4) // 5} new sheets (new_sheet_*.jpg); total windows now {len(all_wins)}")


def signals(wins):
    """One row of candidate signals per window. Population-relative ones use the windows of the SAME video as the pool, like the pipeline's Population over a video's candidate shots."""
    from types import SimpleNamespace
    from aikyam_video.creative.model import SlotFeatures
    from aikyam_video.creative.shots import Population, dead_slots
    from aikyam_video.vision import ClipVision
    cv = ClipVision(); rows = []
    mk = lambda w: SimpleNamespace(slots=SlotFeatures(0.5, w["sharpness"], w["jitter"], w["motion"], w["luma"], w["contrast"], w["clipped"], w["concentration"], [0.0] * len(w["luma"]), [-30.0] * len(w["luma"]), [0.0] * len(w["luma"])), start=0.0)
    pops = {v: Population([mk(w) for w in wins if w["video"] == v]) for v in SOURCES}
    for w in wins:
        sh = mk(w); pop = pops[w["video"]]; q = pop.slot_quality(sh.slots); dead = dead_slots(sh, pop); sharp = np.asarray(w["sharpness"]); mid = frame_at(w["path"], w["start"] + L / 2, 480)
        vr = cv.analyze(mid)
        rows.append({"pipeline quality (mean slot q)": float(q.mean()), "min slot q": float(q.min()), "dead-slot share": -float(dead.mean()), "sharpness (log median)": float(np.log(np.median(sharp) + 1)),
                     "sharpness dip (min/median)": float(sharp.min() / (np.median(sharp) + 1e-9)), "jitter (mean, lower better)": -float(np.mean(w["jitter"])), "jitter (max, lower better)": -float(np.max(w["jitter"])),
                     "motion (mean, lower better)": -float(np.mean(w["motion"])), "luma (mean)": float(np.mean(w["luma"])), "luma (min)": float(np.min(w["luma"])), "contrast (mean)": float(np.mean(w["contrast"])),
                     "blown-out (lower better)": -float(np.mean(w["clipped"])), "concentration": float(np.mean(w["concentration"])), "SigLIP aesthetic": float(vr.aesthetic), "SigLIP no_text": float(vr.no_text)})
    return rows


def auc(x, bad):
    """P(a random OK window scores above a random bad one): 0.5 = no information."""
    x = np.asarray(x, float); pos, neg = x[~bad], x[bad]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else float("nan")


def fit():
    from scipy.stats import spearmanr
    wins = json.load(open(f"{OUT}/windows.json")); R = json.load(open("tools/bench_gate_ratings.json")); y = np.array([R[w["id"]] for w in wins], float)
    cache = f"{OUT}/signals.json"
    rows = json.load(open(cache)) if os.path.exists(cache) else signals(wins)
    json.dump(rows, open(cache, "w")); names = list(rows[0]); X = np.array([[r[n] for n in names] for r in rows]); vid = np.array([w["video"] for w in wins]); bad = y <= 2
    print(f"{len(y)} windows, ratings {dict(zip(*np.unique(y, return_counts=True)))}, bad (<=2): {bad.sum()}\n")
    print(f"{'signal':32s} pooled rho  within-video rho (mean)   AUC bad-vs-ok (pooled / mean within)")
    for j, n in enumerate(names):
        pooled = spearmanr(X[:, j], y)[0]
        within = [spearmanr(X[vid == v, j], y[vid == v])[0] for v in SOURCES if len(set(y[vid == v])) > 1 and len(set(X[vid == v, j])) > 1]
        aw = [auc(X[vid == v, j], bad[vid == v]) for v in SOURCES if bad[vid == v].any() and (~bad[vid == v]).any()]
        print(f"{n:32s} {pooled:+.2f}       {np.mean(within):+.2f}                      {auc(X[:, j], bad):.2f} / {np.nanmean(aw):.2f}")


import argparse


if __name__ == "__main__":
    if sys.argv[1] == "add":
        ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--only", nargs="*"); ap.add_argument("--per", type=int, default=18); ap.add_argument("--seed", type=int, default=34); a = ap.parse_args()
        add(a.only, a.per, a.seed)
    else:
        {"sample": sample, "fit": fit}[sys.argv[1]]()
